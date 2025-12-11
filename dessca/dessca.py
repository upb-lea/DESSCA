import numpy as np
import kalepy as kale
import pyswarms as ps
import matplotlib.pyplot as plt


class DesscaModel:
    def __init__(self,
                 box_constraints,
                 bandwidth=1e-1,
                 reference_pdf=None,
                 render_online=False,
                 pso_options=None,
                 state_names=None,
                 buffer_size=None,
                 disc_resolution=None,
                 epsilon=1e-5):

        ### one time initialization
        # a bandwidth of 0.1 yielded empirically good results for box constraints of [-1, 1]
        self.render_online = render_online
        self.dim = len(box_constraints)
        self.lower_bound = np.array(box_constraints)[:, 0]
        self.upper_bound = np.array(box_constraints)[:, -1]
        self.disc_resolution = disc_resolution

        if np.isscalar(bandwidth):
            self.bandwidth = [bandwidth for _ in range(self.dim)]  # set the bandwidth for this estimator
        else:
            self.bandwidth = bandwidth
        uniform_variance = np.diag((self.upper_bound - self.lower_bound) ** 2) / 12.0
        self._disc_covariance = np.diag(self.bandwidth) ** 2 @ uniform_variance
        self._disc_det_cov = np.linalg.det(self._disc_covariance)
        self._disc_inv_cov = np.linalg.inv(self._disc_covariance)

        if pso_options is None:
            pso_options = {'c1': 2, 'c2': 2, 'w': 0.6}
        if state_names is None:
            self.state_names = [f"$x_{i}$" for i in range(self.dim)]
        else:
            self.state_names = state_names

        # instantiate a particle swarm optimizer
        self.optimizer = ps.single.GlobalBestPSO(n_particles=self.dim * 10,
                                                 dimensions=self.dim,
                                                 options=pso_options,
                                                 bounds=(self.lower_bound, self.upper_bound))

        if reference_pdf is None:
            # if no reference_pdf has been defined we assume uniform distribution on the axes
            _state_space_volume = np.prod([_con[-1] - _con[0] for _con in box_constraints])

            def uniform_pdf(x):
                return np.ones_like(x[0]) / _state_space_volume

            self.reference_pdf = uniform_pdf
        else:
            self.reference_pdf = reference_pdf

        # ring buffer for the collected states
        if buffer_size is None:
            self.buffer_size = np.inf
            self.buffer_idx = None
            self.coverage_data = np.empty((self.dim, 0))
        else:
            self.buffer_size = buffer_size
            self.buffer_idx = 0
            self.coverage_data = np.empty((self.dim, buffer_size)) * np.nan

        # initializing
        self.nb_datapoints = 0
        self.suggested_sample = None
        self._last_sample = None

        if self.disc_resolution is None:
            # continuous mode
            self.coverage_pdf = None
            self._bins_volume = 0.0  # infinite resolution -> infinitesimal bin volume

        elif isinstance(self.disc_resolution, int):
            # discrete mode, TODO: allow different resolution for each dimension
            self.resolution_per_dimension = [self.disc_resolution for _ in range(self.dim)]

            self.coverage_pdf = np.zeros(self.resolution_per_dimension)
            self._bins_size = (self.upper_bound - self.lower_bound) / self.disc_resolution
            self._bins_volume = np.prod(self._bins_size)
            self._coordinate_mesh = np.meshgrid(*[np.linspace(self.lower_bound[i] + self._bins_size[i] / 2,
                                                              self.upper_bound[i] - self._bins_size[i] / 2,
                                                              self.disc_resolution,
                                                              endpoint=True) for i in range(self.dim)])
            self._coordinate_list = np.stack([_mesh.flatten() for _mesh in self._coordinate_mesh])
            self.reference_pdf = np.reshape(self.reference_pdf(self._coordinate_list),
                                            shape=self.resolution_per_dimension)

        else:
            raise Exception(f"Discretization resolution must be an integer or None, but is {self.disc_resolution}.")

        # regularization constat
        self.epsilon = epsilon

        # Plotting
        self.scatter_fig = None
        self.scatter_axes = None
        self.heatmap_fig = None
        self.heatmap_axes = None

    def update_coverage_pdf(self, data):
        if self.render_online:
            self.render_scatter(online_data=data)
        # append the newly acquired data

        if self.buffer_idx is None:
            self.coverage_data = np.append(self.coverage_data, np.array(data), axis=1)
            coverage_data = np.copy(self.coverage_data)
        else:
            self.coverage_data[:, self.buffer_idx] = np.reshape(data, (-1))
            self.buffer_idx = (self.buffer_idx + 1) % self.buffer_size
            first_nan_idx = np.where(np.isnan(self.coverage_data))
            if len(first_nan_idx[0]) > 0:
                coverage_data = self.coverage_data[:, 0:first_nan_idx[1][0]]
            else:
                coverage_data = np.copy(self.coverage_data)



        if self.disc_resolution is None:
        # use kalepy KDE in continuous mode
            if (np.shape(coverage_data)[0] >= np.shape(coverage_data)[1] or
                    np.linalg.det(np.cov(coverage_data)) <= self.epsilon):
                # for small no. of samples, the KDE might produce infeasible results
                # hence we work around this by duplicating and adding noise to the few available samples
                _tiled = np.tile(coverage_data, (1, np.max([self.dim, 2]) + 1))
                _noisy = _tiled + np.random.normal(0, 1, np.shape(_tiled))
                self.coverage_pdf = kale.KDE(dataset=_noisy, bandwidth=self.bandwidth, diagonal=True)
            else:
                self.coverage_pdf = kale.KDE(dataset=coverage_data, bandwidth=self.bandwidth, diagonal=True)
            self.nb_datapoints = np.clip(self.nb_datapoints + np.shape(data)[1], 0, self.buffer_size)

        else:
            # use own meshgrid KDE in discrete mode
            self.nb_datapoints = np.clip(self.nb_datapoints + np.shape(data)[1], 0, self.buffer_size)
            _kernels = self._discretized_kernels(data)
            self.coverage_pdf = (
                    ((self.nb_datapoints - np.shape(data)[1]) / self.nb_datapoints) * self.coverage_pdf +
                    _kernels / self.nb_datapoints
            )


    def sample_optimally(self):

        if self.disc_resolution is None:
            # continuous mode

            # check if this is the first sample
            if self.coverage_pdf is not None:
                def _optim_problem(x):
                    result = (self.coverage_pdf.density(np.transpose(x), probability=True)[1] -
                              self.reference_pdf(np.transpose(x)))
                    return result

            else:
                # if this is the first sample, the optimal sample is solely based on the reference coverage
                def _optim_problem(x):
                    result = - self.reference_pdf(np.transpose(x))
                    return result

            _, self.suggested_sample = self.optimizer.optimize(_optim_problem, iters=self.dim * 10 + 5, verbose=False)
            self.optimizer.reset()

        else:
            # discrete mode
            if self.coverage_pdf is not None:
                _min_idx = np.argmin(self.coverage_pdf - self.reference_pdf)

            else:
                # if this is the first sample, the optimal sample is solely based on the reference coverage
                _min_idx = np.argmax(self.reference_pdf)

            _min_idx = np.unravel_index(_min_idx, self.coverage_pdf.shape)
            self.suggested_sample = np.array([_mesh[_min_idx] for _mesh in self._coordinate_mesh])

        return self.suggested_sample

    def downsample(self, data, target_size):
        # this function samples down a large dataset while preserving the original distribution
        if self.render_online:
            print("The render_online feature is not yet available for this function.")

        self.coverage_data = np.copy(data)
        dataset_size = np.shape(self.coverage_data)[1]
        self.reference_pdf = kale.KDE(dataset=self.coverage_data, bandwidth=self.bandwidth)

        def _optim_problem(x):
            result = - self.reference_pdf.density(np.transpose(x), probability=True)[1]
            return result

        # at first, remove the one sample where coverage is largest
        _, suggested_sample = self.optimizer.optimize(_optim_problem, iters=self.dim * 10 + 5, verbose=False)
        distances = np.linalg.norm(np.add(np.transpose([suggested_sample]), -self.coverage_data), axis=0)
        removal_idx = np.argmin(distances)
        self.coverage_data = np.delete(self.coverage_data, removal_idx, 1)
        dataset_size -= 1
        self.coverage_pdf = kale.KDE(dataset=self.coverage_data, bandwidth=self.bandwidth)
        self.optimizer.reset()

        # then, proceed to remove samples that cause biggest deviation from original distribution
        while dataset_size > target_size:
            def _optim_problem(x):
                result = (self.reference_pdf.density(np.transpose(x), probability=True)[1] -
                          self.coverage_pdf.density(np.transpose(x), probability=True)[1])
                return result

            _, suggested_sample = self.optimizer.optimize(_optim_problem, iters=self.dim * 10 + 5, verbose=False)
            distances = np.linalg.norm(np.add(np.transpose([suggested_sample]), -self.coverage_data), axis=0)
            removal_idx = np.argmin(distances)
            self.coverage_data = np.delete(self.coverage_data, removal_idx, 1)
            dataset_size -= 1
            self.coverage_pdf = kale.KDE(dataset=self.coverage_data, bandwidth=self.bandwidth)
            self.optimizer.reset()

    def update_and_sample(self, data=None):
        if data is not None:
            self.update_coverage_pdf(data=data)

        self.sample_optimally()

        return self.suggested_sample

    def _continuous_kernel(self, center):
        def placed_kernel(x):
            diff = x[:, :, np.newaxis] - center[:, np.newaxis, :]
            kernel_eval = (np.exp(-0.5 * np.einsum("dnm, dj, dnm -> nm",
                                                   diff, self._disc_inv_cov, diff)) /
                           np.sqrt((2 * np.pi) ** self.dim * self._disc_det_cov))

            return np.mean(kernel_eval, axis=1)

        return placed_kernel

    def _discretized_kernels(self, data):
        placed_kernels = self._continuous_kernel(data)
        disc_kernels = placed_kernels(self._coordinate_list)
        disc_kernels = np.reshape(disc_kernels, shape=self.resolution_per_dimension) * self._bins_volume

        return disc_kernels

    def plot_heatmap(self, resolution=100, **kwargs):
        if self.dim == 1:
            print("Heatmap plot is not available for dim < 2")
            return None

        else:
            kwargs.setdefault("cmap", "inferno")
            kwargs.setdefault("vmin", 0)
            kwargs.setdefault("aspect", "equal")
            kwargs.setdefault("origin", "lower")

            self.heatmap_fig, self.heatmap_axes = plt.subplots(self.dim, self.dim)
            self.heatmap_axes = np.reshape(self.heatmap_axes, (self.dim, self.dim)).tolist()

            for i in range(self.dim):
                for j in range(self.dim):
                    if j < i:
                        _xj = np.linspace(self.lower_bound[j], self.upper_bound[j], resolution)
                        _xi = np.linspace(self.lower_bound[i], self.upper_bound[i], resolution)
                        grid = np.meshgrid(_xj, _xi, indexing="ij")
                        positions = np.vstack([_grid.ravel() for _grid in grid])
                        kde = kale.KDE(np.concatenate(([self.coverage_data[j]], [self.coverage_data[i]]), axis=0))
                        _, _disc_coverage = kde.density(positions, probability=True)
                        _disc_coverage = np.reshape(_disc_coverage, grid[0].shape)
                        img = self.heatmap_axes[i][j].imshow(np.transpose(_disc_coverage), **kwargs)
                        self.heatmap_axes[i][j].set_xticks([])
                        self.heatmap_axes[i][j].set_yticks([])
                        if i == self.dim - 1:
                            self.heatmap_axes[i][j].set_xlabel(self.state_names[j])
                        if j == 0:
                            self.heatmap_axes[i][j].set_ylabel(self.state_names[i])

            self.heatmap_fig.colorbar(img, ax=np.ravel(self.heatmap_axes).tolist())
            for _ax in np.array(self.heatmap_axes).flatten():
                if not _ax.images:
                    _ax.remove()
            plt.show()
            return self.heatmap_fig, self.heatmap_axes

    def render_scatter(self, online_data=None):

        if self.scatter_fig is None:
            self.scatter_fig, self.scatter_axes = plt.subplots(self.dim, self.dim)
            self.scatter_axes = np.reshape(self.scatter_axes, (self.dim, self.dim)).tolist()
            for i in range(self.dim):
                for j in range(self.dim):
                    _j_margin = (self.upper_bound[j] - self.lower_bound[j]) * 0.1
                    _i_margin = (self.upper_bound[i] - self.lower_bound[i]) * 0.1
                    if j < i:
                        self.scatter_axes[i][j].set_xlim(
                            [self.lower_bound[j] - _j_margin, self.upper_bound[j] + _j_margin])
                        self.scatter_axes[i][j].set_ylim(
                            [self.lower_bound[i] - _i_margin, self.upper_bound[i] + _i_margin])
                        if i == self.dim - 1:
                            self.scatter_axes[i][j].set_xlabel(self.state_names[j])
                        if j == 0:
                            self.scatter_axes[i][j].set_ylabel(self.state_names[i])
                        self.scatter_axes[i][j].grid(True)
                        self.scatter_axes[i][j].set_aspect((self.upper_bound[j]
                                                            - self.lower_bound[j]
                                                            + 2 * _j_margin)
                                                           / (self.upper_bound[i]
                                                              - self.lower_bound[i]
                                                              + 2 * _i_margin))
                        if np.shape(online_data)[1] > 1:
                            self.scatter_axes[i][j].plot(online_data[j], online_data[i], ".", color="blue")

                    elif i == j:
                        self.scatter_axes[i][j].set_xlim(
                            [self.lower_bound[j] - _j_margin, self.upper_bound[j] + _j_margin])
                        self.scatter_axes[i][j].set_ylim([-0.1, 1.1])
                        if i == self.dim - 1:
                            plt.xlabel(self.state_names[i])
                        self.scatter_axes[i][j].grid(True)
                        self.scatter_axes[i][j].set_aspect(
                            (self.upper_bound[j] - self.lower_bound[j] + 2 * _j_margin) / 1.2
                        )
                    elif j > i:
                        self.scatter_axes[i][j].remove()

            if np.shape(online_data)[1] == 1:
                self.suggested_sample = online_data
            elif np.shape(online_data)[1] > 1:
                self.suggested_sample = np.full((np.shape(online_data)[0], 1), None, dtype=object)
                online_data = np.full((np.shape(online_data)[0], 1), None, dtype=object)

        for i in range(self.dim):
            for j in range(self.dim):
                if j < i:
                    if self._last_sample is not None:
                        self.scatter_axes[i][j].plot(self._last_sample[j], self._last_sample[i], ".", color="blue")
                    self.scatter_axes[i][j].plot(online_data[j], online_data[i], ".", color="orange")

                if j == i:
                    self.scatter_axes[i][j].plot(self.coverage_data[j], np.zeros_like(self.coverage_data[i]), ".",
                                                 color="blue")
                    self.scatter_axes[i][j].plot(self.suggested_sample[i], 0, ".", color="orange")

        self._last_sample = self.suggested_sample

        plt.pause(0.001)

    def plot_scatter(self, scatter_kwargs=None):
        if scatter_kwargs is None:
            scatter_kwargs = {}

        if not hasattr(self, "scatter_fig") or not self.render_online:
            self.scatter_fig, self.scatter_axes = plt.subplots(self.dim, self.dim)
            self.scatter_axes = np.reshape(self.scatter_axes, (self.dim, self.dim)).tolist()

            for i in range(self.dim):
                for j in range(self.dim):
                    _j_margin = (self.upper_bound[j] - self.lower_bound[j]) * 0.1
                    _i_margin = (self.upper_bound[i] - self.lower_bound[i]) * 0.1

                    if j < i:
                        self.scatter_axes[i][j].set_xlim([self.lower_bound[j] - _j_margin,
                                                          self.upper_bound[j] + _j_margin])
                        self.scatter_axes[i][j].set_ylim([self.lower_bound[i] - _i_margin,
                                                          self.upper_bound[i] + _i_margin])
                        if i == self.dim - 1:
                            self.scatter_axes[i][j].set_xlabel(self.state_names[j])
                        if j == 0:
                            self.scatter_axes[i][j].set_ylabel(self.state_names[i])

                        self.scatter_axes[i][j].grid(True)
                        self.scatter_axes[i][j].set_aspect((self.upper_bound[j]
                                                            - self.lower_bound[j]
                                                            + 2 * _j_margin)
                                                           / (self.upper_bound[i]
                                                              - self.lower_bound[i]
                                                              + 2 * _i_margin))
                    elif i == j:
                        self.scatter_axes[i][j].set_xlim(
                            [self.lower_bound[j] - _j_margin, self.upper_bound[j] + _j_margin])
                        self.scatter_axes[i][j].set_ylim([-0.1, 1.1])
                        self.scatter_axes[i][j].grid(True)
                        self.scatter_axes[i][j].set_aspect(
                            (self.upper_bound[j] - self.lower_bound[j] + 2 * _j_margin) / 1.2
                        )
                    elif j > i:
                        self.scatter_axes[i][j].remove()

        for i in range(self.dim):
            for j in range(self.dim):
                if j < i:
                    scatter_kwargs.setdefault("color", "blue")
                    scatter_kwargs.setdefault("marker", ".")
                    scatter_kwargs.setdefault("linestyle", "")
                    self.scatter_axes[i][j].plot(self.coverage_data[j], self.coverage_data[i], **scatter_kwargs)

                if j == i:
                    if isinstance(self.bandwidth, (list, tuple, np.ndarray)):
                        _bandwidth = self.bandwidth[i]
                    else:
                        _bandwidth = self.bandwidth

                    kde = kale.KDE(self.coverage_data[i], bandwidth=_bandwidth, diagonal=True)
                    points, density = kde.density(np.linspace(self.lower_bound[i], self.upper_bound[i], 500),
                                                  probability=True)
                    self.scatter_axes[i][j].plot(points, density)
                    self.scatter_axes[i][j].set_ylim([-0.1, np.max([1.1, np.max(density)])])
                    _j_margin = (self.upper_bound[j] - self.lower_bound[j]) * 0.1
                    self.scatter_axes[i][j].set_aspect((self.upper_bound[j]
                                                        - self.lower_bound[j]
                                                        + 2 * _j_margin)
                                                       / (np.max([1.1, np.max(density)]) - 0.1))
                    plt.grid(True)

        plt.show()

        return self.scatter_fig, self.scatter_axes
