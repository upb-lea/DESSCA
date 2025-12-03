import json
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os
import sys
sys.path.append("../..")
from dessca import dessca_model

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

def print_confidence_interval(data, alpha=0.05):
    whole_mean = np.mean(data)
    whole_std = np.std(data)
    z_l, z_u = sp.stats.norm.interval(1 - alpha, 0, 1)
    n = data.size
    print(f"Sample mean = {whole_mean}")
    print(f"Lower bound = {whole_mean + z_l * whole_std / np.sqrt(n)}")
    print(f"Upper bound = {whole_mean + z_u * whole_std / np.sqrt(n)}")

experiment_name = "Uniform"
logfiles = list(filter(lambda x: "Validation" in x, os.listdir("./" + experiment_name)))
files = ["./" + experiment_name + "/" + file for file in logfiles]

returns_per_agents = []
for _file in files:
    with open(_file, 'r') as json_file:
        data = json.load(json_file)

        states = np.copy(data["state_history"])
        rewards = np.copy(data["reward_history"])

    returns_this_agent = []
    for _episode in rewards:
        return_this_episode = np.sum(_episode)
        returns_this_agent.append(return_this_episode)

    returns_per_agents.append(returns_this_agent)

dessca_model1 = dessca_model(box_constraints=[[  - 2.4,    2.4],
                                             [     -7,      7],
                                             [ -np.pi, +np.pi],
                                             [    -10,     10]],
                            state_names=[r"$x$", r"$v$", r"$\epsilon$", r"$\omega$"])

shortest_len = None
for _runs in returns_per_agents:
    if shortest_len is None:
        shortest_len = len(_runs)
    if len(_runs) < shortest_len:
        shortest_len = len(_runs)

for idx, G in enumerate(returns_per_agents):
    returns_per_agents[idx] = G[:shortest_len] # crop by minimum number of executed episodes

returns_per_agents = np.array(returns_per_agents)
mean_GG_uniform = np.mean(returns_per_agents, axis=1)
sigma_GG_uniform = np.std(returns_per_agents, axis=0)

print("Uniform CI")
print_confidence_interval(returns_per_agents/200)


experiment_name = "Dessca"
logfiles = list(filter(lambda x: "Validation" in x, os.listdir("./" + experiment_name)))
files = ["./" + experiment_name + "/" + file for file in logfiles]


returns_all_agents = []
for _file in files:
    with open(_file, 'r') as json_file:
        data = json.load(json_file)

        states = np.copy(data["state_history"])
        rewards = np.copy(data["reward_history"])

    returns_this_agent = []
    for _episode in rewards:
        return_this_episode = np.sum(_episode)
        returns_this_agent.append(return_this_episode)

    returns_all_agents.append(returns_this_agent)

dessca_model2 = dessca_model(box_constraints=[[  - 2.4,    2.4],
                                             [     -7,      7],
                                             [ -np.pi, +np.pi],
                                             [    -10,     10]],
                            state_names=[r"$x$", r"$v$", r"$\epsilon$", r"$\omega$"])

shortest_len = None
for _runs in returns_all_agents:
    if shortest_len is None:
        shortest_len = len(_runs)
    if len(_runs) < shortest_len:
        shortest_len = len(_runs)

for idx, G in enumerate(returns_all_agents):
    returns_all_agents[idx] = G[:shortest_len]

returns_per_agents = np.array(returns_all_agents)
mean_GG_dessca = np.mean(returns_per_agents, axis=1)

print("Dessca CI")
print_confidence_interval(returns_per_agents / 200)

max_return = 200

plt.figure(figsize=(3, 3))
box_uniform = plt.boxplot(mean_GG_uniform / max_return, positions=[0])
box_dessca = plt.boxplot(mean_GG_dessca / max_return, positions=[0.5])
plt.grid()
plt.xticks([0, 0.5], [r"$\mathrm{ES}$", r"$\mathrm{DESSCA}$"])
plt.ylabel(r"$g / g_\mathrm{max}$")
plt.tick_params(axis='both', direction="in", left=True, right=True, bottom=True, top=True)

plotName = "CP_Boxplots" + '.pdf'
plt.savefig(plotName, bbox_inches='tight')
plt.close()

print(f"ES MEDIAN: {np.median(mean_GG_uniform) / max_return}")
print(f"DESSCA MEDIAN: {np.median(mean_GG_dessca) / max_return}")
print(f"ES INTERQUARTILE RANGE: {sp.stats.iqr(mean_GG_uniform) / max_return}")
print(f"DESSCA INTERQUARTILE RANGE: {sp.stats.iqr(mean_GG_dessca) / max_return}")
print(f"relative improvement: {np.median(mean_GG_dessca) / np.median(mean_GG_uniform) - 1}")



