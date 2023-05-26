from scipy.stats import binom
import time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LogNorm
import os
import math
import pickle

n_nodes = 7
n = n_nodes*n_nodes-n_nodes
pm=1.0/n
num_run = 1000000
#l=0.5
l_list = [0.05, 0.1, 0.2, 0.5, 1.0]
nc_start = 32
nc_end = n

expected_improvement_RL = []
expected_improvement_RL_optimal = []
expected_improvement_mutation = []
expected_improvement_difference = []
expected_improvement_difference_optimal = []
expected_improvement_difference_RL_self = []
heat_map_RL = np.zeros((len(l_list), n+1))
heat_map_RL_optimal = np.zeros((len(l_list), n+1))
heat_map_mutation = np.zeros((len(l_list), n+1))
for i, l in enumerate(l_list):
    for nc_1 in range(nc_start, nc_end):#range(n+1):
            nd_1 = n-nc_1
            p_c = nc_1/n
            e_z_c = math.sqrt(p_c/(1-p_c))
            e_z_w = 1.0/e_z_c
            z_c = math.log(e_z_c)
            nd_samples = binom.rvs(n,1-p_c, size=num_run)
            improvement_list = []
            for nd_sample in nd_samples:
                p_c_update = e_z_c/(e_z_c+e_z_w*math.exp(-2*(nd_1-nd_sample)*l*(1-p_c)))
                p_w_update = e_z_w/(e_z_w+e_z_c*math.exp(-2*(nd_1-nd_sample)*l*p_c))
                nc_update = nd_sample*(1-p_w_update) + (n-nd_sample)*p_c_update
                improvement = nc_update-nc_1
                improvement_list.append(improvement)
            heat_map_RL[i][nc_1] = np.mean(improvement_list)
            # RL with all correct entries to have ~100% probability to be true
            p_c = 1 - nd_1/(nd_1+1.0)
            e_z_c = math.sqrt(p_c/(1-p_c))
            e_z_w = 1.0/e_z_c
            nd_samples = binom.rvs(nd_1+1,1-p_c, size=num_run)
            improvement_list_optimal = []
            for nd_sample in nd_samples:
                p_c_update = e_z_c/(e_z_c+e_z_w*math.exp(-2*(nd_1-nd_sample)*l*(1-p_c)))
                p_w_update = e_z_w/(e_z_w+e_z_c*math.exp(-2*(nd_1-nd_sample)*l*p_c))
                nc_update = nd_sample*(1-p_w_update) + (nd_1+1-nd_sample)*p_c_update + n-nd_1-1
                improvement = nc_update-nc_1
                improvement_list_optimal.append(improvement)
            heat_map_RL_optimal[i][nc_1] = np.mean(improvement_list_optimal)
            heat_map_mutation[i][nc_1] = np.mean(np.maximum(0,n-nc_1-binom.rvs(nc_1, pm, size=num_run)-binom.rvs(n-nc_1, 1-pm, size=num_run)))

heatmap_difference_RL = heat_map_RL - heat_map_mutation
heatmap_difference_RL_optimal = heat_map_RL_optimal - heat_map_mutation
heatmap_difference_RL_self = heat_map_RL_optimal - heat_map_RL

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','nas101_expected_improvement_mutation_pm_{}_heatmap_icml.pkl'.format(pm))
with open(result_file_name, 'wb') as result_file:
    pickle.dump(heat_map_mutation, result_file)
result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','nas101_expected_improvement_RL_uniform_heatmap_icml.pkl')
with open(result_file_name, 'wb') as result_file:
    pickle.dump(heat_map_RL, result_file)
result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','nas101_expected_improvement_RL_optimal_heatmap_icml.pkl')
with open(result_file_name, 'wb') as result_file:
    pickle.dump(heat_map_RL_optimal, result_file)

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 30}

#matplotlib.rc('font', **font)
plt.rc('font', size=22)  # controls default text sizes
plt.rc('axes', titlesize=14)     # fontsize of the axes title
plt.rc('axes', labelsize=20)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=20)    # fontsize of the tick labels
plt.rc('ytick', labelsize=20)
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

fig_width = 16
fig_height = 4

fig = plt.figure(figsize=(fig_width, fig_height))
#ax = sns.heatmap(heatmap_graph_crossover, annot=True, fmt='.2f', linewidth=0.5)
#ax = sns.heatmap(heatmap_graph_crossover, fmt='.2f', linewidth=0.5)
ax = sns.heatmap(heat_map_mutation[:len(l_list),nc_start:nc_end], annot=True, fmt='.2f', linewidth=0.5, xticklabels=list(reversed(range(n+1-nc_end,n+1-nc_start))), yticklabels=l_list)
ax.invert_yaxis()
ax.invert_xaxis()
plt.title("Expected Improvement of mutation (n={})".format(n_nodes))
plt.ylabel(r'value of $\alpha \cdot \eta$')
plt.xlabel("No. of incorrect entries in parent 1")
plt.xticks(fontsize=8, rotation=90)
plt.yticks(fontsize=8, rotation=0)
plt.show()
plot_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Plots','nas101_expected_improvement_mutation_pm_{}_heatmap_icml.pdf'.format(pm))
fig.savefig(plot_file_name, bbox_inches='tight')

#plt.imshow(heatmap_difference, cmap='hot', interpolation='nearest')
#fig = plt.figure(figsize=(16, 15))
fig = plt.figure(figsize=(fig_width, fig_height))
#ax = sns.heatmap(heatmap_graph_crossover, annot=True, fmt='.2f', linewidth=0.5)
#ax = sns.heatmap(heatmap_graph_crossover, fmt='.2f', linewidth=0.5)
ax = sns.heatmap(heatmap_difference_RL[:len(l_list),nc_start:nc_end], annot=True, fmt='.2f', linewidth=0.5, xticklabels=list(reversed(range(n+1-nc_end,n+1-nc_start))), yticklabels=l_list)
ax.invert_yaxis()
ax.invert_xaxis()
plt.title("Difference between RL_unbiased and mutation in Expected Improvement (n={})".format(n_nodes))
plt.ylabel(r'value of $\alpha \cdot \eta$')
plt.xlabel("No. of incorrect entries in parent 1")
plt.xticks(fontsize=8, rotation=90)
plt.yticks(fontsize=8, rotation=0)
plt.show()
plot_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Plots','nas101_expected_improvement_RL_uniform_vs_mutation_pm_{}_heatmap_icml.pdf'.format(pm))
fig.savefig(plot_file_name, bbox_inches='tight')

#fig = plt.figure(figsize=(16, 15))
fig = plt.figure(figsize=(fig_width, fig_height))
#ax = sns.heatmap(heatmap_graph_crossover, annot=True, fmt='.2f', linewidth=0.5)
#ax = sns.heatmap(heatmap_graph_crossover, fmt='.2f', linewidth=0.5)
ax = sns.heatmap(heatmap_difference_RL_optimal[:len(l_list),nc_start:nc_end], annot=True, fmt='.2f', linewidth=0.5, xticklabels=list(reversed(range(n+1-nc_end,n+1-nc_start))), yticklabels=l_list)
ax.invert_yaxis()
ax.invert_xaxis()
plt.title("Difference between RL_oracle and mutation in Expected Improvement (n={})".format(n_nodes))
#plt.ylabel("learning rate for RL")
plt.ylabel(r'value of $\alpha \cdot \eta$')
plt.xlabel("No. of incorrect entries in parent 1")
plt.xticks(fontsize=8, rotation=90)
plt.yticks(fontsize=8, rotation=0)
plt.show()
plot_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Plots','nas101_expected_improvement_RL_optimal_vs_mutation_pm_{}_heatmap_icml.pdf'.format(pm))
fig.savefig(plot_file_name, bbox_inches='tight')

#fig = plt.figure(figsize=(16, 15))
fig = plt.figure(figsize=(fig_width, fig_height))
#ax = sns.heatmap(heatmap_graph_crossover, annot=True, fmt='.2f', linewidth=0.5)
#ax = sns.heatmap(heatmap_graph_crossover, fmt='.2f', linewidth=0.5)
ax = sns.heatmap(heatmap_difference_RL_self[:len(l_list),nc_start:nc_end], annot=True, fmt='.2f', linewidth=0.5, xticklabels=list(reversed(range(n+1-nc_end,n+1-nc_start))), yticklabels=l_list)
ax.invert_yaxis()
ax.invert_xaxis()
plt.title("Difference between RL_oracle and RL_unbiased in Expected Improvement (n={})".format(n_nodes))
#plt.ylabel("learning rate for RL")
plt.ylabel(r'value of $\alpha \cdot \eta$')
plt.xlabel("Expected No. of incorrect entries of RL controller")
plt.xticks(fontsize=8, rotation=90)
plt.yticks(fontsize=8, rotation=0)
plt.show()
plot_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Plots','nas101_expected_improvement_RL_optimal_vs_RL_uniform_heatmap_icml.pdf')
fig.savefig(plot_file_name, bbox_inches='tight')

#fig = plt.figure(figsize=(16, 15))
fig = plt.figure(figsize=(fig_width, fig_height))
#ax = sns.heatmap(heatmap_graph_crossover, annot=True, fmt='.2f', linewidth=0.5)
#ax = sns.heatmap(heatmap_graph_crossover, fmt='.2f', linewidth=0.5)
ax = sns.heatmap(heat_map_RL[:len(l_list),nc_start:nc_end], annot=True, fmt='.2f', linewidth=0.5, xticklabels=list(reversed(range(n+1-nc_end,n+1-nc_start))), yticklabels=l_list)
ax.invert_yaxis()
ax.invert_xaxis()
plt.title("Expected Improvement of RL_unbiased (n={})".format(n_nodes))
#plt.ylabel("learning rate for RL")
plt.ylabel(r'value of $\alpha \cdot \eta$')
plt.xlabel("Expected No. of incorrect entries of RL controller")
plt.xticks(fontsize=8, rotation=90)
plt.yticks(fontsize=8, rotation=0)
plt.show()
plot_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Plots','nas101_expected_improvement_RL_uniform_heatmap_icml.pdf')
fig.savefig(plot_file_name, bbox_inches='tight')

#fig = plt.figure(figsize=(16, 15))
fig = plt.figure(figsize=(fig_width, fig_height))
#ax = sns.heatmap(heatmap_graph_crossover, annot=True, fmt='.2f', linewidth=0.5)
#ax = sns.heatmap(heatmap_graph_crossover, fmt='.2f', linewidth=0.5)
ax = sns.heatmap(heat_map_RL_optimal[:len(l_list),nc_start:nc_end], annot=True, fmt='.2f', linewidth=0.5, xticklabels=list(reversed(range(n+1-nc_end,n+1-nc_start))), yticklabels=l_list)
ax.invert_yaxis()
ax.invert_xaxis()
plt.title("Expected Improvement of RL_oracle (n={})".format(n_nodes))
#plt.ylabel("learning rate for RL")
plt.ylabel(r'value of $\alpha \cdot \eta$')
plt.xlabel("Expected No. of incorrect entries of RL controller")
plt.xticks(fontsize=8, rotation=90)
plt.yticks(fontsize=8, rotation=0)
plt.show()
plot_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Plots','nas101_expected_improvement_RL_optimal_heatmap_icml.pdf')
fig.savefig(plot_file_name, bbox_inches='tight')
