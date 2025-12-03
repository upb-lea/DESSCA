import h5py
import matplotlib.pyplot as plt
import numpy as np
import os

if __name__ == "__main__":
    agent_no = "1"
    folder_name = "Dessca/Dessca_" + agent_no
    #folder_name = "Uniform/Uniform_" + agent_no

    #file_name = "training"
    file_name = "validation"

    nb = 0
    most_recent = 0
    start_zoom = 0
    stop_zoom = 0

    if most_recent:
        files = ["./" + folder_name + "/" + file for file in os.listdir("./" + folder_name) if (file.startswith(file_name))]
        files.sort(key=os.path.getmtime, reverse=True)
        path = files[0]
        nb = int(path[path.index("training_") + len("training_"): path.index(".hdf5")])
    else:
        path = folder_name + "/" + file_name + "_" + str(nb) + ".hdf5"

    fig, axarr = plt.subplots(2, 2, figsize=(30, 15))
    fig.suptitle(folder_name + "_episode_" + str(nb), fontsize=16)

    with h5py.File(path, "r") as f:
        lim = np.copy(f['limits'])

        obs = np.transpose(np.copy(f['observations']))
        rew = np.copy(f['rewards'])
        history = np.copy(f['history'])

    for i in range(len(lim)):
        obs[i] = obs[i] * lim[i]

    tau = 1e-4

    plt.subplot(4, 3, 1)
    plt.title("i_d")
    plt.plot(obs[0], color="blue")
    plt.plot(obs[7], color="red")
    plt.grid()

    plt.subplot(4, 3, 4)
    plt.title("i_q")
    plt.plot(obs[1], color="blue")
    plt.plot(obs[8], color="red")
    plt.grid()

    plt.subplot(4, 3, 3)
    plt.title("\omega")
    plt.plot(obs[2], color="blue")
    plt.grid()

    plt.subplot(4, 3, 7)
    plt.title("reward history")
    plt.ylim([0.05, 0.15])
    plt.plot(history, color="blue")
    plt.grid()

    epsilon = np.arctan2(obs[4], obs[3])
    plt.subplot(4, 3, 6)
    plt.title("\epsilon")
    plt.plot(epsilon)
    plt.grid()

    plt.subplot(4, 3, (2, 5))
    plt.title("i_d-i_q")
    rect = plt.Rectangle((-280, -280), 560, 560, color="red", alpha=0.5)
    plt.gcf().gca().add_artist(rect)
    circle = plt.Circle((0, 0), 270, color='orange', fill=True, alpha=0.5)
    plt.gcf().gca().add_artist(circle)
    circle = plt.Circle((0, 0), 230, color='white', fill=True)
    plt.gcf().gca().add_artist(circle)
    plt.plot(obs[0], obs[1], color="blue")
    plt.scatter(obs[7], obs[8], color="red", zorder=10)
    plt.scatter(obs[0][0], obs[1][0], color="red", marker="x", zorder=10)
    plt.xlim([-280, 280])
    plt.ylim([-280, 280])
    plt.grid()
    # plot voltage limits ellipsis
    p = 3
    r_s = 17.932e-3
    l_d = 0.37e-3
    l_q = 1.2e-3
    psi_p = 65.65e-3
    i_d = np.linspace(-280, 280, 1000)
    u, c = np.unique(obs[2], return_counts=True)
    dup = u[c > 5]
    w_el = dup * p
    for _w_el in w_el:
        V_s = lim[7] * 2 / np.sqrt(3)
        i_q_plus = np.sqrt(V_s ** 2 / (_w_el ** 2 * l_q ** 2) - (l_d ** 2) / (l_q ** 2) * (i_d + psi_p / l_d) ** 2)
        i_q_minus = - np.sqrt(V_s ** 2 / (_w_el ** 2 * l_q ** 2) - (l_d ** 2) / (l_q ** 2) * (i_d + psi_p / l_d) ** 2)
        plt.plot(i_d, i_q_plus, color="blue")
        plt.plot(i_d, i_q_minus, color="blue", label=r"available voltage")

    plt.subplot(4, 3, 8)
    plt.title("u_d")
    plt.plot(obs[5])
    plt.grid()

    plt.subplot(4, 3, 9)
    plt.title("u_q")
    plt.plot(obs[6])
    plt.grid()

    plt.subplot(4, 3, 10)
    plt.title("r")
    plt.plot(rew)
    print(f"Return: {np.sum(rew)}")

    plt.subplot(4, 3, 11)
    plt.title("cos-sin(eps)")
    plt.plot(obs[3], color="blue")
    plt.plot(obs[4], color="red")

    if most_recent:
        plotName = 'Plots/most_recent.pdf'
    else:
        plotName = 'Plots/' + file_name + "_" + str(nb) + '.pdf'

    plt.savefig(plotName, bbox_inches='tight')

    plt.close()