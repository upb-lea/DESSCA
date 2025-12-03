import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import h5py

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

def print_confidence_interval(data, n=None, alpha=0.05):
    whole_mean = np.mean(data)
    whole_std = np.std(data)
    z_l, z_u = sp.stats.norm.interval(1 - alpha, 0, 1)
    if n is None:
        n = data.size
    print(f"Sample mean = {whole_mean}")
    print(f"Lower bound = {whole_mean + z_l * whole_std / np.sqrt(n)}")
    print(f"Upper bound = {whole_mean + z_u * whole_std / np.sqrt(n)}")

experiment_name = "Uniform"
files = ["./" + experiment_name + "/" + experiment_name + "_" + str(nb) +"/validation_0.hdf5" for nb in range(50)]

returns_per_agents = []
for _file in files:
    with h5py.File(_file, "r") as f:
        rewards = np.copy(f["rewards"])

    return_this_agent = np.sum(rewards)
    returns_per_agents.append(return_this_agent)

returns_per_agents_uniform = np.array(returns_per_agents)

print("Uniform CI")
print_confidence_interval(returns_per_agents_uniform / 19050, n =50)

experiment_name = "Dessca"
files = ["./" + experiment_name + "/" + experiment_name + "_" + str(nb) +"/validation_0.hdf5" for nb in range(50)]

returns_per_agents = []
for _file in files:
    with h5py.File(_file, "r") as f:
        rewards = np.copy(f["rewards"])

    return_this_agent = np.sum(rewards)
    returns_per_agents.append(return_this_agent)

returns_per_agents_dessca = np.array(returns_per_agents)

print("DESSCA CI")
print_confidence_interval(returns_per_agents_dessca / 19050, n =50)

max_return = 19050

plt.figure(figsize=(3, 3))
box_uniform = plt.boxplot(returns_per_agents_uniform / max_return, positions=[0], showmeans=True)
box_dessca = plt.boxplot(returns_per_agents_dessca / max_return, positions=[0.5], showmeans=True)
plt.grid()
plt.xticks([0, 0.5], [r"$\mathrm{ES}$", r"$\mathrm{DESSCA}$"])
plt.ylabel(r"$g / g_\mathrm{max}$")
plt.tick_params(axis='both', direction="in", left=True, right=True, bottom=True, top=True)

plotName = "CurrentControl_Boxplots" + '.pdf'
plt.savefig(plotName, bbox_inches='tight')
plt.close()


print(f"ES MEDIAN: {np.median(returns_per_agents_uniform) / max_return}")
print(f"DESSCA MEDIAN: {np.median(returns_per_agents_dessca) / max_return}")
print(f"ES INTERQUARTILE RANGE: {sp.stats.iqr(returns_per_agents_uniform) / max_return}")
print(f"DESSCA INTERQUARTILE RANGE: {sp.stats.iqr(returns_per_agents_dessca) / max_return}")
print(f"relative improvement: {np.median(returns_per_agents_dessca) / np.median(returns_per_agents_uniform) - 1}")
