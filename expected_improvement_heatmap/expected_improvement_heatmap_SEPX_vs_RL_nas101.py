from scipy.stats import binom
import time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LogNorm
import os
import math

n_nodes = 7
n = n_nodes*n_nodes-n_nodes
alpha = 0.9
beta = 0.0

n1_0 = 9
n0_0 = n-n1_0
n1_2 = 9
n0_2 = n-n1_2
n1_1 = 9
n0_1 = n-n1_1
pm=1.0/n
pc=0.5
num_run = 1000000

nd_02_random = n1_0*n0_2/n + n0_0*n1_2/n

print("nd_02_random: {}".format(nd_02_random))

l=0.1
nc_start = 32
nc_end = n

expected_improvement_RL = []
expected_improvement_RL_optimal = []

heatmap_graph_crossover = np.zeros((n+1,n+1))
heatmap_difference_SEP_RL = np.zeros((n+1,n+1))
heatmap_difference_SEP_RL_optimal = np.zeros((n+1,n+1))
for nd_12 in range(1,11):#range(n+1):
    for nc_1 in range(nc_start, nc_end):#range(n+1):
        # RL with each entry having the same probability to be correct
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
        expected_improvement_RL.append(np.mean(improvement_list))
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
        expected_improvement_RL_optimal.append(np.mean(improvement_list_optimal))
        n_s = max(nc_1-nd_12,0)
        expected_improvement_crossover = np.mean(np.maximum(0,(n-nc_1)*nd_12/(n-n_s)-binom.rvs(nd_12, pc, size=num_run)))
        heatmap_difference_SEP_RL[nd_12][nc_1] = expected_improvement_crossover - expected_improvement_RL[-1]
        heatmap_difference_SEP_RL_optimal[nd_12][nc_1] = expected_improvement_crossover - expected_improvement_RL_optimal[-1]

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
#plt.imshow(heatmap_difference, cmap='hot', interpolation='nearest')
fig = plt.figure(figsize=(16, 10))
#ax = sns.heatmap(heatmap_graph_crossover, annot=True, fmt='.2f', linewidth=0.5)
#ax = sns.heatmap(heatmap_graph_crossover, fmt='.2f', linewidth=0.5)
ax = sns.heatmap(heatmap_difference_SEP_RL[:11,nc_start:nc_end], annot=True, fmt='.2f', linewidth=0.5, xticklabels=list(reversed(range(n+1-nc_end,n+1-nc_start))))
ax.invert_yaxis()
ax.invert_xaxis()
plt.title("Difference between SEP crossover and RL_unbiased in Expected Improvement, n={}, $n^1_{}$={}, $n^1_1$={}, $n^1_2$={}".format(n_nodes, '{\mathrm{opt}}', n1_0, n1_1, n1_2), fontweight="bold")
plt.ylabel("$d_e^*$ between parent 1 and 2")
plt.xlabel("$d_e^*$ between parent 1 and global optimum")
plt.xticks(rotation=90)
plt.yticks(rotation=0)

plot_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Plots','nas101_expected_improvement_SEP_vs_RL_uniform_n_{}_n10_{}_n12_{}_heatmap_icml.pdf'.format(n, n1_0, n1_2))
fig.savefig(plot_file_name, bbox_inches='tight')
plt.show()

#plt.imshow(heatmap_difference, cmap='hot', interpolation='nearest')
fig = plt.figure(figsize=(16, 10))
#ax = sns.heatmap(heatmap_graph_crossover, annot=True, fmt='.2f', linewidth=0.5)
#ax = sns.heatmap(heatmap_graph_crossover, fmt='.2f', linewidth=0.5)
ax = sns.heatmap(heatmap_difference_SEP_RL_optimal[:11,nc_start:nc_end], annot=True, fmt='.2f', linewidth=0.5, xticklabels=list(reversed(range(n+1-nc_end,n+1-nc_start))))
ax.invert_yaxis()
ax.invert_xaxis()
plt.title("Difference between SEP crossover and RL_oracle in Expected Improvement, n={}, $n^1_{}$={}, $n^1_1$={}, $n^1_2$={}".format(n_nodes, '{\mathrm{opt}}', n1_0, n1_1, n1_2), fontweight="bold")
plt.ylabel("$d_e^*$ between parent 1 and 2")
plt.xlabel("$d_e^*$ between parent 1 and global optimum")
plt.xticks(rotation=90)
plt.yticks(rotation=0)

plot_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Plots','nas101_expected_improvement_SEP_vs_RL_optimal_n_{}_n10_{}_n12_{}_heatmap_icml.pdf'.format(n, n1_0, n1_2))
fig.savefig(plot_file_name, bbox_inches='tight')
plt.show()
