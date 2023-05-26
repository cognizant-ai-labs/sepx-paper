from scipy.stats import binom
import time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LogNorm
import os
import pickle


population_size = 100
tournament_size = 10
repeat_num = 50
max_time_budget = int(10000)
variant_name = "edit_path_center_crossover_GEDfitness_NoIsomorphism_stats_theory"
benchmark_name = "nas-bench-101"
result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','{}_pop{}_tournament{}_run{}_evaluation{}.pkl'.format(
                                                variant_name, population_size, tournament_size, repeat_num, max_time_budget))
with open(result_file_name, 'rb') as result_file:
    #random_data, evolution_mutation_data, evolution_crossover_raw_data, evolution_crossover_match_data, evolution_crossover_refined_data = pickle.load(result_file)
    _, _, _, _, evolution_crossover_refined_data = pickle.load(result_file)

n_nodes = 7
n = n_nodes*n_nodes-n_nodes

heatmap_graph_crossover_empirical = np.zeros((n+1,n+1))
heatmap_num_edges_empirical = np.zeros((n+1,n+1))
total_count = 0
for _, _, _, _, _, _, theory_list in evolution_crossover_refined_data:
    for time_spent, crossover_ged, ged_to_best_1, ged_to_best_2, ged_parents_edge_only, ged_to_best_1_edge_only, ged_to_best_2_edge_only, num_edges_1, num_edges_2 in theory_list[:1000]:
        heatmap_graph_crossover_empirical[int(ged_parents_edge_only[0])][n-int(ged_to_best_2_edge_only[0])] += 1
        heatmap_num_edges_empirical[num_edges_1][num_edges_2] += 1
        total_count += 1

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 30}

#matplotlib.rc('font', **font)
plt.rc('font', size=20)  # controls default text sizes
plt.rc('axes', titlesize=20)     # fontsize of the axes title
plt.rc('axes', labelsize=20)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=20)    # fontsize of the tick labels
plt.rc('ytick', labelsize=20)
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
#plt.imshow(heatmap_difference, cmap='hot', interpolation='nearest')

fig = plt.figure(figsize=(16, 15))
#ax = sns.heatmap(heatmap_graph_crossover, annot=True, fmt='.2f', linewidth=0.5)
#ax = sns.heatmap(heatmap_graph_crossover_empirical, fmt='.2f', linewidth=0.5)
ax = sns.heatmap(heatmap_graph_crossover_empirical[:11,32:]/total_count, fmt='.3f', annot=True, linewidth=0.5, xticklabels=list(reversed(range(11))))
ax.invert_yaxis()
ax.invert_xaxis()
plt.title("Relative Frequency of Events Happened (NAS-bench-101)", fontweight="bold")
plt.ylabel("$d_e^*$ between parent 1 and 2")
plt.xlabel("$d_e^*$ between parent 1 and global optimum")
plt.xticks(rotation=90)
plt.yticks(rotation=0)

plot_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Plots','nas101_num_occurances_GED_parent_nc_parent2_n_{}_heatmap_icml.pdf'.format(n))
fig.savefig(plot_file_name, bbox_inches='tight')
plt.show()
'''
plt.rc('font', size=20)  # controls default text sizes
plt.rc('axes', titlesize=20)     # fontsize of the axes title
plt.rc('axes', labelsize=20)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=20)    # fontsize of the tick labels
plt.rc('ytick', labelsize=20)
'''
fig = plt.figure(figsize=(16, 15))
#ax = sns.heatmap(heatmap_graph_crossover, annot=True, fmt='.2f', linewidth=0.5)
#ax = sns.heatmap(heatmap_num_edges_empirical, fmt='.2f', linewidth=0.5)
ax = sns.heatmap(heatmap_num_edges_empirical[:13,:13]/total_count, fmt='.3f', annot=True, linewidth=0.5)
ax.invert_yaxis()
plt.title("Relative Frequency of Events Happened (NAS-bench-101)", fontweight="bold")
plt.ylabel("No. of edges in parent 1")
plt.xlabel("No. of edges in parent 2")
plt.xticks(rotation=90)
plt.yticks(rotation=0)

plot_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Plots','nas101_num_occurances_num_edges_n_{}_heatmap_icml.pdf'.format(n))
fig.savefig(plot_file_name, bbox_inches='tight')
plt.show()
'''
heatmap_raw_crossover = np.zeros((n+1,n+1))
for n1_0 in range(n+1):
    for n1_2 in range(n+1):
        n0_0 = n-n1_0
        n0_2 = n-n1_2
        nd_02_random = n1_0*n0_2/n + n0_0*n1_2/n
        heatmap_raw_crossover[n1_0][n1_2] = nd_02_random

fig = plt.figure(figsize=(16, 15))
#ax = sns.heatmap(heatmap_graph_crossover, annot=True, fmt='.2f', linewidth=0.5)
ax = sns.heatmap(heatmap_raw_crossover, fmt='.2f', linewidth=0.5)
ax.invert_yaxis()
plt.title("Expected GED_to_optimal (parent 2) using raw crossover".format(n1_0, n1_2))
plt.xlabel("number of 1s in parent 2")
plt.ylabel("number of 1s in global optimal")
plt.xticks(fontsize=8, rotation=90)
plt.yticks(fontsize=8, rotation=0)

plot_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Plots','Expected_GED_to_optimal_raw_crossover.pdf'.format(n1_0, n1_2))
fig.savefig(plot_file_name, bbox_inches='tight')
plt.show()
'''
