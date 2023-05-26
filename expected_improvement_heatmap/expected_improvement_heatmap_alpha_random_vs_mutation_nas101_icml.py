from scipy.stats import binom
import time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LogNorm
import os

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
nd_12_random = n1_1*n0_2/n + n0_1*n1_2/n

print("nd_02_random: {}".format(nd_02_random))

heatmap_graph_crossover = np.zeros((n+1,n+1))
for nd_12 in range(1,11):#range(n+1):
    for nc_1 in range(32,n+1):#range(n+1):
        #nd_02_graph_crossover = n-nd_12+(2*nd_12/n-1)*nc_1
        #nd_02_graph_crossover = n-nc_1+((2*nc_1-(alpha+1)*n)/(n-n*alpha))*nd_12
        #expected_improvement_crossover = np.mean(np.maximum(0,n-nc_1-(n-nc_1)*(n*(1-alpha)-nd_12)/(n*(1-alpha))-binom.rvs(nd_12, pc, size=num_run)))
        #n_s = max(nc_1-nd_12,0)
        #print("nc_1: {}, nd_12: {}, alpha: {}".format(nc_1, nd_12, alpha))
        #expected_improvement_crossover = np.mean(np.maximum(0,n-nc_1-(n-nc_1)*(n*(1-alpha)-nd_12)/(n*(1-alpha))-binom.rvs(nd_12, pc, size=num_run)))
        n0w_1=(n-nc_1-(n0_0-n0_1))/2.0
        n1w_1=(n-nc_1-(n1_0-n1_1))/2.0
        expected_improvement_random = np.mean(np.maximum(0,n-nc_1-n1w_1*n1_2/n-n0w_1*n0_2/n-binom.rvs(round(nd_12_random), pc, size=num_run)))
        #expected_improvement_crossover = np.mean(np.maximum(0,(n-nc_1)*nd_12/(n-n_s)-binom.rvs(nd_12, pc, size=num_run)))
        #expected_improvement_crossover = np.mean(np.maximum(0,n-nc_1-(n-nc_1)*(1-beta)*((n*(1-alpha)-(n-nc_1)*beta)-(nd_12-(n-nc_1)*beta))/(n*(1-alpha)-(n-nc_1)*beta)-binom.rvs(nd_12, pc, size=num_run)))
        #nd_02_mutation = n-nc_1+(2*nc_1-n)*pm
        expected_improvement_mutation = np.mean(np.maximum(0,n-nc_1-binom.rvs(nc_1, pm, size=num_run)-binom.rvs(n-nc_1, 1-pm, size=num_run)))
        #heatmap_graph_crossover[nd_12][nc_1] = nd_02_graph_crossover - nd_02_random
        #heatmap_graph_crossover[nd_12][nc_1] = (nd_02_graph_crossover+(n-nc_1))/2.0 - nd_02_mutation
        heatmap_graph_crossover[nd_12][nc_1] = expected_improvement_random - expected_improvement_mutation

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
ax = sns.heatmap(heatmap_graph_crossover[:11,32:], annot=True, fmt='.2f', linewidth=0.5, xticklabels=list(reversed(range(11))))
ax.invert_yaxis()
ax.invert_xaxis()
plt.title("Difference between standard crossover and mutation in Expected Improvement, n={}, $n^1_{}$={}, $n^1_1$={}, $n^1_2$={}".format(n_nodes, '{\mathrm{opt}}', n1_0, n1_1, n1_2), fontweight="bold")
plt.ylabel("$d_e^*$ between parent 1 and 2")
plt.xlabel("$d_e^*$ between parent 1 and global optimum")
plt.xticks(rotation=90)
plt.yticks(rotation=0)

plot_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Plots','nas101_random_vs_mutation_pm_{}_alpha_new_difference_GED_2-to-optimal_n_{}_n10_{}_n12_{}_heatmap_icml.pdf'.format(pm, n, n1_0, n1_2))
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
