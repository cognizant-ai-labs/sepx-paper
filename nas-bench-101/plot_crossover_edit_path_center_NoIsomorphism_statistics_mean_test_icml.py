# Standard imports
import copy
import numpy as np
import matplotlib.pyplot as plt
import random
from ged_util_nas101 import nas101_to_digraph, ged_nas101, ged_nas101_optimize, crossover_edit_path_center, digraph_to_nas101, node_match
from multiprocessing import Pool
import pickle
import os
import networkx as nx

population_size = 100
tournament_size = 10
repeat_num = 50
max_time_budget = int(10000)
exp_variant = "edit_path_center_crossover_NoIsomorphism_stats_theory_mean_test"
#exp_variant = "edit_path_center_crossover_GEDfitness_NoIsomorphism_stats"
random_data = []
evolution_mutation_data = []
evolution_crossover_raw_data = []
evolution_crossover_match_data = []
evolution_crossover_refined_data = []
for run_id in range(repeat_num):
    result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','{}_pop{}_tournament{}_run1_evaluation{}_run_id_{}.pkl'.format(
                                                    exp_variant, population_size, tournament_size, max_time_budget, run_id))
    with open(result_file_name, 'rb') as result_file:
        random_data_tmp, evolution_mutation_data_tmp, evolution_crossover_raw_data_tmp, evolution_crossover_match_data_tmp, evolution_crossover_refined_data_tmp = pickle.load(result_file)
    random_data.append(random_data_tmp[0])
    evolution_mutation_data.append(evolution_mutation_data_tmp[0])
    evolution_crossover_raw_data.append(evolution_crossover_raw_data_tmp[0])
    evolution_crossover_match_data.append(evolution_crossover_match_data_tmp[0])
    evolution_crossover_refined_data.append(evolution_crossover_refined_data_tmp[0])

# loading rl data
run_num = 50
n_iters = 10000
lr = 0.5
y_star_test = 0.056824247042338016
rl_data = []
for run in range(run_num):
    result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results_rl','nas101_rl_steps_{}_lr_{}_run_{}.pkl'.format(
                                                    n_iters, lr, run))
    with open(result_file_name, 'rb') as result_file:
        rl_data.append(pickle.load(result_file))
    #print(rl_data[-1]['logits_record'][-1])
    print(rl_data[-1]['runtime'][-1])
mean_rl = []
median_rl = []
std_rl = []
for i in range(n_iters):
    test_acc_tmp_list = []
    for run in range(run_num):
        test_acc_tmp_list.append(1-y_star_test-rl_data[run]['regret_test'][i])
    mean_rl.append(np.mean(test_acc_tmp_list))
    median_rl.append(np.median(test_acc_tmp_list))
    std_rl.append(np.std(test_acc_tmp_list))

def calc_optimal_ratio(raw_data, data_index, optimal_target):
    run_num = len(raw_data)
    optimal_ratio_list = []
    for time_step in range(len(raw_data[0][data_index])):
        count_num = 0
        for run in range(run_num):
            if raw_data[run][data_index][time_step] == optimal_target:
                count_num += 1
        optimal_ratio_list.append(float(count_num)/float(run_num))
    return optimal_ratio_list

plt.rc('font', size=14)  # controls default text sizes
plt.rc('axes', titlesize=14)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
'''
data_index = 1
DATASET_NAME = 'NAS-bench-101'
optimal_target = 0.9518229365348816
#optimal_target = 91.60666665039064
#optimal_target = 73.49333323567708
#optimal_target = 46.766666727701825
f = plt.figure()
optimal_ratio_list = calc_optimal_ratio(random_data, data_index, optimal_target)
plt.plot(optimal_ratio_list, label='random search')
optimal_ratio_list = calc_optimal_ratio(evolution_mutation_data, data_index, optimal_target)
plt.plot(optimal_ratio_list, label='RE mutation only')
optimal_ratio_list = calc_optimal_ratio(evolution_crossover_raw_data, data_index, optimal_target)
plt.plot(optimal_ratio_list, label='RE + standard crossover')
optimal_ratio_list = calc_optimal_ratio(evolution_crossover_refined_data, data_index, optimal_target)
plt.plot(optimal_ratio_list, label='RE + SEP crossover')
plt.ylabel("ratio to reach global optimum")
plt.xlabel("timestep")
plt.title("percentage of runs that reaches optimal solution ({})".format(DATASET_NAME), fontweight="bold", fontsize=12)
plt.legend(fontsize=12)
plt.grid()
plt.show()

plot_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Plots','optimal_ratio_center_crossover_pop{}_tournament{}_run{}_evaluation{}.pdf'.format(
                                                population_size, tournament_size, repeat_num, max_time_budget))
f.savefig(plot_file_name, bbox_inches='tight')
'''
'''
f = plt.figure()
window_size=100
for _, _, _, ged_to_best, _, status_list in evolution_crossover_refined_data:
    valid_ratio_list = []
    crossover_count = 0
    print(ged_to_best[1000:])
    for status in status_list:
        if type(status)==tuple:
            valid_ratio_list.append(status[2]/status[0])
            crossover_count+=1
    moving_average = np.convolve(valid_ratio_list, np.ones(window_size)/window_size, mode='valid')
    print("crossover_count: {}".format(crossover_count))
    plt.plot(valid_ratio_list, 'o', label='is_crossover', alpha=0.5)
    plt.ylabel("ratio of valid graphs")
    plt.xlabel("times")
    plt.title("ratio of valid graphs during refined crossover")
    plt.show()
'''
'''
f = plt.figure()
window_size=100
for times, _, _, _, _, status_list in evolution_crossover_match_data:
    operation_list = []
    time_list = times
    time_previous = 0
    crossover_count = 0
    for status in status_list:
        if type(status)==tuple:
            operation_list.append(1)
            crossover_count+=1
        else:
            if status == time_previous+1:
                operation_list[-1]=0
            #operation_list.append(0)
            time_previous = status
    moving_average = np.convolve(operation_list, np.ones(window_size)/window_size, mode='valid')
    print("crossover_count: {}".format(crossover_count))
    plt.plot(moving_average, 'o', label='is_crossover', alpha=0.5)
    plt.ylabel("ratio of valid crossover (moving average)")
    plt.xlabel("times")
    plt.title("ratio of valid crossover during evolution (match_crossover)")
    plt.show()
'''

'''
f = plt.figure()

plt.subplot(2, 2, 1)
for times, best_valid, best_test, _ in random_data:
    plt.plot(times, best_valid, label='valid', color='red', alpha=0.5)
    plt.plot(times, best_test, label='test', color='blue', alpha=0.5)

plt.ylabel('accuracy')
plt.xlabel('time spent (seconds)')
plt.ylim(0.92, 0.96)
plt.grid()
plt.title('Random search trajectories (red=validation, blue=test)')


plt.subplot(2, 2, 2)
for times, best_valid, best_test, _ in evolution_mutation_data:
    plt.plot(times, best_valid, label='valid', color='red', alpha=0.5)
    plt.plot(times, best_test, label='test', color='blue', alpha=0.5)

plt.ylabel('accuracy')
plt.xlabel('time spent (seconds)')
plt.ylim(0.92, 0.96)
plt.grid()
plt.title('Evolution search trajectories (red=validation, blue=test)')

plt.subplot(2, 2, 3)
for times, best_valid, best_test, _ in evolution_crossover_raw_data:
    plt.plot(times, best_valid, label='valid', color='red', alpha=0.5)
    plt.plot(times, best_test, label='test', color='blue', alpha=0.5)

plt.ylabel('accuracy')
plt.xlabel('time spent (seconds)')
plt.ylim(0.92, 0.96)
plt.grid()
plt.title('Evolution search raw crossover trajectories (red=validation, blue=test)')


plt.subplot(2, 2, 4)
for times, best_valid, best_test, _, parents_ged, _ in evolution_crossover_refined_data:
    plt.plot(times, best_valid, label='valid', color='red', alpha=0.5)
    plt.plot(times, best_test, label='test', color='blue', alpha=0.5)

plt.ylabel('accuracy')
plt.xlabel('time spent (seconds)')
plt.ylim(0.92, 0.96)
plt.grid()
plt.title('Evolution search refined crossover trajectories (red=validation, blue=test)')
plt.show()

plot_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Plots','convergence_center_crossover_pop{}_tournament{}_run{}_evaluation{}.pdf'.format(
                                                population_size, tournament_size, repeat_num, max_time_budget))
f.savefig(plot_file_name, bbox_inches='tight')

window_size = 100
moving_average_data_match = []
f = plt.figure()
for times, best_valid, best_test, _, parents_ged, _ in evolution_crossover_refined_data:
    for i in range(len(parents_ged)):
        if i>100:
            if type(parents_ged[i])==int:
                parents_ged[i] = parents_ged[i-1]
    #print(parents_ged[9000:])
    moving_average = np.convolve(parents_ged, np.ones(window_size)/window_size, mode='valid')
    plt.plot(times[window_size-1:], moving_average, label='parents_ged_match', alpha=0.5)
    moving_average_data_match.append((times[window_size-1:], moving_average))
plt.ylabel('ged_parents (match_crossover)')
plt.xlabel('total number of candidates evaluated')
plt.grid()
plt.title('graph edit distance between parents')
plt.show()
plot_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Plots','ged_parents_match_crossover_pop{}_tournament{}_run{}_evaluation{}.pdf'.format(
                                                population_size, tournament_size, repeat_num, max_time_budget))
f.savefig(plot_file_name, bbox_inches='tight')

moving_average_data_refined = []
f = plt.figure()
for times, best_valid, best_test, _, parents_ged, _ in evolution_crossover_refined_data:
    for i in range(len(parents_ged)):
        if i>100:
            if type(parents_ged[i])==int:
                parents_ged[i] = parents_ged[i-1]
    #print(parents_ged[9000:])
    moving_average = np.convolve(parents_ged, np.ones(window_size)/window_size, mode='valid')
    plt.plot(times[window_size-1:], moving_average, label='parents_ged_refined', alpha=0.5)
    moving_average_data_refined.append((times[window_size-1:], moving_average))
plt.ylabel('ged_parents (refined_crossover)')
plt.xlabel('total number of candidates evaluated')
plt.grid()
plt.title('graph edit distance between parents')
plt.show()
plot_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Plots','ged_parents_center_crossover_pop{}_tournament{}_run{}_evaluation{}.pdf'.format(
                                                population_size, tournament_size, repeat_num, max_time_budget))
f.savefig(plot_file_name, bbox_inches='tight')
'''
# Compare the mean test accuracy along with error bars.
def plot_data(data, color, label, marker, gran=1, max_budget=5000000):
    """Computes the mean and IQR fixed time steps."""
    xs = range(0, max_budget+1, gran)
    mean = [0.0]
    per25 = [0.0]
    per75 = [0.0]
    std = [0.0]

    repeats = len(data)
    pointers = [1 for _ in range(repeats)]

    cur = gran
    while cur < max_budget+1:
        all_vals = []
        for repeat in range(repeats):
            while (pointers[repeat] < len(data[repeat][0]) and data[repeat][0][pointers[repeat]] < cur):
                pointers[repeat] += 1
            prev_time = data[repeat][0][pointers[repeat]-1]
            prev_test = data[repeat][2][pointers[repeat]-1]
            next_time = data[repeat][0][pointers[repeat]]
            next_test = data[repeat][2][pointers[repeat]]
            assert prev_time < cur and next_time >= cur

            # Linearly interpolate the test between the two surrounding points
            cur_val = ((cur - prev_time) / (next_time - prev_time)) * (next_test - prev_test) + prev_test

            all_vals.append(cur_val)

        all_vals = sorted(all_vals)
        std.append(np.std(all_vals))
        mean.append(sum(all_vals) / float(len(all_vals)))
        per25.append(all_vals[int(0.25 * repeats)])
        per75.append(all_vals[int(0.75 * repeats)])

        cur += gran

    plt.plot(xs, mean, color=color, label=label, marker=marker, markevery=0.2, markersize=10, linewidth=2)
    plt.fill_between(xs, np.array(mean)+np.array(std), np.array(mean)-np.array(std), alpha=0.1, linewidth=0, facecolor=color)

def plot_data_val(data, color, label, gran=1, max_budget=5000000):
    """Computes the mean and IQR fixed time steps."""
    xs = range(0, max_budget+1, gran)
    mean = [0.0]
    per25 = [0.0]
    per75 = [0.0]

    repeats = len(data)
    pointers = [1 for _ in range(repeats)]

    cur = gran
    while cur < max_budget+1:
        all_vals = []
        for repeat in range(repeats):
            while (pointers[repeat] < len(data[repeat][0]) and data[repeat][0][pointers[repeat]] < cur):
                pointers[repeat] += 1
            prev_time = data[repeat][0][pointers[repeat]-1]
            prev_test = data[repeat][1][pointers[repeat]-1]
            next_time = data[repeat][0][pointers[repeat]]
            next_test = data[repeat][1][pointers[repeat]]
            assert prev_time < cur and next_time >= cur

            # Linearly interpolate the test between the two surrounding points
            cur_val = ((cur - prev_time) / (next_time - prev_time)) * (next_test - prev_test) + prev_test

            all_vals.append(cur_val)

        all_vals = sorted(all_vals)
        mean.append(sum(all_vals) / float(len(all_vals)))
        per25.append(all_vals[int(0.25 * repeats)])
        per75.append(all_vals[int(0.75 * repeats)])

        cur += gran

    plt.plot(xs, mean, color=color, label=label, linewidth=2)
    plt.fill_between(xs, per25, per75, alpha=0.1, linewidth=0, facecolor=color)

def plot_data_ged_to_best(data, color, label, gran=1, max_budget=5000000):
    """Computes the mean and IQR fixed time steps."""
    xs = range(0, max_budget+1, gran)
    mean = [0.0]
    per25 = [0.0]
    per75 = [0.0]
    std = [0.0]

    repeats = len(data)
    pointers = [1 for _ in range(repeats)]

    cur = gran
    while cur < max_budget+1:
        all_vals = []
        for repeat in range(repeats):
            while (pointers[repeat] < len(data[repeat][0]) and data[repeat][0][pointers[repeat]] < cur):
                pointers[repeat] += 1
            prev_time = data[repeat][0][pointers[repeat]-1]
            prev_test = data[repeat][3][pointers[repeat]-1]
            next_time = data[repeat][0][pointers[repeat]]
            next_test = data[repeat][3][pointers[repeat]]
            assert prev_time < cur and next_time >= cur

            # Linearly interpolate the test between the two surrounding points
            cur_val = ((cur - prev_time) / (next_time - prev_time)) * (next_test - prev_test) + prev_test

            all_vals.append(cur_val)

        all_vals = sorted(all_vals)
        std.append(np.std(all_vals))
        mean.append(sum(all_vals) / float(len(all_vals)))
        per25.append(all_vals[int(0.25 * repeats)])
        per75.append(all_vals[int(0.75 * repeats)])

        cur += gran

    plt.plot(xs, mean, color=color, label=label, linewidth=2)
    plt.fill_between(xs, np.array(mean)+np.array(std), np.array(mean)-np.array(std), alpha=0.1, linewidth=0, facecolor=color)

def plot_data_parents_ged(data, color, label, gran=1, max_budget=5000000):
    """Computes the mean and IQR fixed time steps."""
    xs = range(0, max_budget+1, gran)
    mean = [0.0]
    per25 = [0.0]
    per75 = [0.0]

    repeats = len(data)
    pointers = [1 for _ in range(repeats)]

    cur = gran
    while cur < max_budget+1:
        all_vals = []
        for repeat in range(repeats):
            while (pointers[repeat] < len(data[repeat][0]) and data[repeat][0][pointers[repeat]] < cur):
                pointers[repeat] += 1
            prev_time = data[repeat][0][pointers[repeat]-1]
            prev_test = data[repeat][4][pointers[repeat]-1]
            next_time = data[repeat][0][pointers[repeat]]
            next_test = data[repeat][4][pointers[repeat]]
            assert prev_time < cur and next_time >= cur

            # Linearly interpolate the test between the two surrounding points
            cur_val = ((cur - prev_time) / (next_time - prev_time)) * (next_test - prev_test) + prev_test

            all_vals.append(cur_val)

        all_vals = sorted(all_vals)
        mean.append(sum(all_vals) / float(len(all_vals)))
        per25.append(all_vals[int(0.25 * repeats)])
        per75.append(all_vals[int(0.75 * repeats)])

        cur += gran

    plt.plot(xs, mean, color=color, label=label, linewidth=2)
    plt.fill_between(xs, per25, per75, alpha=0.1, linewidth=0, facecolor=color)

def plot_data_parents_ged_moving_average(data, color, label, gran=1, max_budget=5000000):
    """Computes the mean and IQR fixed time steps."""
    xs = range(99, max_budget+1, gran)
    mean = [0.0]
    per25 = [0.0]
    per75 = [0.0]

    repeats = len(data)
    pointers = [1 for _ in range(repeats)]

    cur = 100
    while cur < max_budget+1:
        all_vals = []
        for repeat in range(repeats):
            while (pointers[repeat] < len(data[repeat][0]) and data[repeat][0][pointers[repeat]] < cur):
                pointers[repeat] += 1
            prev_time = data[repeat][0][pointers[repeat]-1]
            prev_test = data[repeat][1][pointers[repeat]-1]
            next_time = data[repeat][0][pointers[repeat]]
            next_test = data[repeat][1][pointers[repeat]]
            assert prev_time < cur and next_time >= cur

            # Linearly interpolate the test between the two surrounding points
            cur_val = ((cur - prev_time) / (next_time - prev_time)) * (next_test - prev_test) + prev_test

            all_vals.append(cur_val)

        all_vals = sorted(all_vals)
        mean.append(sum(all_vals) / float(len(all_vals)))
        per25.append(all_vals[int(0.25 * repeats)])
        per75.append(all_vals[int(0.75 * repeats)])

        cur += gran

    plt.plot(xs, mean, color=color, label=label, linewidth=2)
    plt.fill_between(xs, per25, per75, alpha=0.1, linewidth=0, facecolor=color)

f = plt.figure()
plot_data(random_data, 'c', 'random search', 'o', max_budget=max_time_budget)
plot_data(evolution_mutation_data, 'blue', 'RE mutation only', 'v', max_budget=max_time_budget)
plot_data(evolution_crossover_raw_data, 'g', 'RE + standard crossover', '^', max_budget=max_time_budget)
xs = range(1, max_time_budget+1)
plt.plot(xs, mean_rl, color='#ff7f0e', label='RL', marker='s', markevery=0.2, markersize=10, linewidth=2)
#plt.plot(xs, median_rl, color='yellow', label='RL', linewidth=2)
plt.fill_between(xs, np.array(mean_rl)+np.array(std_rl), np.array(mean_rl)-np.array(std_rl), alpha=0.1, linewidth=0, facecolor='#ff7f0e')
plot_data(evolution_crossover_refined_data, 'red', 'RE + SEP crossover', '*', max_budget=max_time_budget)
plt.legend(loc='lower right', fontsize=12)
#plt.ylim(0.92, 0.95)
plt.title('Convergence Plot on NAS-bench-101 (50 runs)', fontweight="bold")
plt.xlabel('total number of architectures evaluated')
plt.ylabel('test accuracy')
plt.grid()
plt.show()
plot_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Plots','test_accuracy_rl_mean_test_lr_{}_pop{}_tournament{}_run{}_evaluation{}_icml.pdf'.format(
                                                lr, population_size, tournament_size, repeat_num, max_time_budget))
f.savefig(plot_file_name, bbox_inches='tight')

f = plt.figure()
plot_data_val(random_data, 'red', 'random', max_budget=max_time_budget)
plot_data_val(evolution_mutation_data, 'blue', 'evolution_mutation', max_budget=max_time_budget)
plot_data_val(evolution_crossover_raw_data, 'g', 'evolution_crossover_raw', max_budget=max_time_budget)
plot_data_val(evolution_crossover_match_data, 'y', 'evolution_crossover_match', max_budget=max_time_budget)
plot_data_val(evolution_crossover_refined_data, 'c', 'evolution_crossover_refined', max_budget=max_time_budget)
plt.legend(loc='lower right')
#plt.ylim(0.92, 0.97)
plt.xlabel('total number of candidates evaluated')
plt.ylabel('validation accuracy')
plt.grid()
plt.show()
plot_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Plots','validation_accuracy_center_crossover_pop{}_tournament{}_run{}_evaluation{}.pdf'.format(
                                                population_size, tournament_size, repeat_num, max_time_budget))
f.savefig(plot_file_name, bbox_inches='tight')

f = plt.figure()
plot_data_ged_to_best(random_data, 'c', 'random search', max_budget=max_time_budget)
plot_data_ged_to_best(evolution_mutation_data, 'blue', 'RE mutation only', max_budget=max_time_budget)
plot_data_ged_to_best(evolution_crossover_raw_data, 'g', 'RE + standard crossover', max_budget=max_time_budget)
plot_data_ged_to_best(evolution_crossover_refined_data, 'red', 'RE + SEP crossover', max_budget=max_time_budget)
plt.legend(loc='upper right', fontsize=12)
plt.title('Convergence Plot on NAS-bench-101 (50 runs)', fontweight="bold")
plt.xlabel('total number of architectures evaluated')
plt.ylabel('GED to global optimum')
plt.grid()
plt.show()
plot_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Plots','ged_to_best_center_crossover_pop{}_tournament{}_run{}_evaluation{}.pdf'.format(
                                                population_size, tournament_size, repeat_num, max_time_budget))
f.savefig(plot_file_name, bbox_inches='tight')

f = plt.figure()
plot_data_parents_ged(evolution_crossover_match_data, 'y', 'evolution_crossover_match', max_budget=max_time_budget)
plot_data_parents_ged(evolution_crossover_refined_data, 'c', 'evolution_crossover_refined', max_budget=max_time_budget)
plt.legend(loc='lower right')
plt.xlabel('total number of candidates evaluated')
plt.ylabel('parents ged')
plt.grid()
plt.show()
plot_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Plots','ged_parents_eachpoint_center_crossover_pop{}_tournament{}_run{}_evaluation{}.pdf'.format(
                                                population_size, tournament_size, repeat_num, max_time_budget))
f.savefig(plot_file_name, bbox_inches='tight')

f = plt.figure()
plot_data_parents_ged_moving_average(moving_average_data_match, 'y', 'evolution_crossover_match', max_budget=max_time_budget)
plot_data_parents_ged_moving_average(moving_average_data_refined, 'c', 'evolution_crossover_refined', max_budget=max_time_budget)
plt.legend(loc='lower right')
plt.xlabel('total number of candidates evaluated')
plt.ylabel('parents ged (moving average)')
plt.grid()
plt.show()
plot_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Plots','ged_parents_movingaverage_center_crossover_pop{}_tournament{}_run{}_evaluation{}.pdf'.format(
                                                population_size, tournament_size, repeat_num, max_time_budget))
f.savefig(plot_file_name, bbox_inches='tight')
