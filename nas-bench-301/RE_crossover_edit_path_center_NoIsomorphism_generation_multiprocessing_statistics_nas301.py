# Standard imports
import copy
import numpy as np
import matplotlib.pyplot as plt
import random
from ged_util_nas301 import nas301_to_digraph, ged_nas301, ged_nas301_optimize, crossover_edit_path_center_nas301, digraph_to_nas301, edge_match, valid_nas301, digraph_to_nas301_general
from multiprocessing import Pool
import pickle
import os
import networkx as nx
from collections import namedtuple
from ConfigSpace.read_and_write import json as cs_json
import nasbench301 as nb

NODE_NUM=4
# Default dirs for models
# Note: Uses 0.9 as the default models, switch to 1.0 to use 1.0 models
version = '1.0'

current_dir = os.path.dirname(os.path.abspath(__file__))
models_0_9_dir = os.path.join(current_dir, 'nb_models_0.9')
model_paths_0_9 = {
    model_name : os.path.join(models_0_9_dir, '{}_v0.9'.format(model_name))
    for model_name in ['xgb', 'gnn_gin', 'lgb_runtime']
}
models_1_0_dir = os.path.join(current_dir, 'nb_models_1.0')
model_paths_1_0 = {
    model_name : os.path.join(models_1_0_dir, '{}_v1.0'.format(model_name))
    for model_name in ['xgb', 'gnn_gin', 'lgb_runtime']
}
model_paths = model_paths_0_9 if version == '0.9' else model_paths_1_0

# If the models are not available at the paths, automatically download
# the models
# Note: If you would like to provide your own model locations, comment this out
if not all(os.path.exists(model) for model in model_paths.values()):
    nb.download_models(version=version, delete_zip=True,
                    download_dir=current_dir)

# Load the performance surrogate model
#NOTE: Loading the ensemble will set the seed to the same as used during training (logged in the model_configs.json)
#NOTE: Defaults to using the default model download path
print("==> Loading performance surrogate model...")
ensemble_dir_performance = model_paths['xgb']
print(ensemble_dir_performance)
performance_model = nb.load_ensemble(ensemble_dir_performance)

OPS = [
        'max_pool_3x3',
        'avg_pool_3x3',
        'skip_connect',
        'sep_conv_3x3',
        'sep_conv_5x5',
        'dil_conv_3x3',
        'dil_conv_5x5'
        ]

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

def data_query_nas301(raw_spec, node_num=4):
    spec = copy.deepcopy(raw_spec) 
    for i in range(len(spec)):
        spec[i] = (OPS[spec[i][0]], spec[i][1])
    genotype_config = Genotype(
                normal=spec[:node_num*2],
                normal_concat=[2, 3, 4, 5],
                reduce=spec[node_num*2:],
                reduce_concat=[2, 3, 4, 5]
                )
    data = {}
    data['validation_accuracy'] = performance_model.predict(config=genotype_config, representation="genotype", with_noise=True)
    data['test_accuracy'] = performance_model.predict(config=genotype_config, representation="genotype", with_noise=False)
    return data

def random_spec(cell_num=2, node_num=4, ops_num=7):
    operation_list = []
    for _ in range(cell_num):
        for i in range(node_num):
            income_index = np.random.choice(i+2, 2, replace=False)
            for index in income_index:
                operation_list.append((np.random.choice(ops_num), index))
    return operation_list

def ged_to_best_calc(new_spec):
    return 6.0

def ged_calc(spec_1, spec_2):
    dg_1 = nas301_to_digraph(spec_1)
    dg_2 = nas301_to_digraph(spec_2)
    ged, _ = ged_nas301(dg_1, dg_2)
    return ged

def ged_calc_optimize(spec_1, spec_2):
    dg_1 = nas301_to_digraph(spec_1)
    dg_2 = nas301_to_digraph(spec_2)
    ged = ged_nas301_optimize(dg_1, dg_2)
    return ged

def array_rank_transform(arr):
    sorted_list = sorted(arr)
    rank = 1
    # seed initial rank as 1 because that's first item in input list
    sorted_rank_list = [1]
    for i in range(1, len(sorted_list)):
        if sorted_list[i] != sorted_list[i-1]:
            rank += 1
        sorted_rank_list.append(rank)
    rank_list = []
    for item in arr:
        for index, element in enumerate(sorted_list):
            if element == item:
                rank_list.append(sorted_rank_list[index])
                # we want to break out of inner for loop
                break
    return rank_list


def mutate_spec(old_spec, mutation_rate=1.0, ops_num=7):
    rand_edge_index = random.randint(0,len(old_spec)-1)
    node_index = rand_edge_index//2 + 2
    if rand_edge_index%2==0:
        pair_edge_index = rand_edge_index+1
    else:
        pair_edge_index = rand_edge_index-1
    same = True
    while same:
        new_op_index = np.random.choice(ops_num)
        new_node_index = np.random.choice(node_index)
        if (new_op_index!=old_spec[rand_edge_index][0] and new_node_index==old_spec[rand_edge_index][1]) or (new_node_index!=old_spec[rand_edge_index][1] and new_node_index!=old_spec[pair_edge_index][1]):
            same = False
    new_spec = copy.deepcopy(old_spec)
    new_spec[rand_edge_index] = (new_op_index, new_node_index)
    return new_spec

def refined_parents_selection(population, sample_size):
    samples = random_combination(population, sample_size)
    best_candidate = sorted(samples, key=lambda i:i[0])[-1]
    population_size = len(population)
    distance_list = np.zeros(population_size)
    perf_diff_list = np.zeros(population_size)
    #ged_pair_list = []
    for i in range(population_size):
        #ged_pair_list.append((population[i][1], best_candidate[1]))
        distance_list[i] = ged_calc_optimize(population[i][1], best_candidate[1])
        perf_diff_list[i] = abs(population[i][0]-best_candidate[0])
    #with Pool() as pool:
    #    distance_list = np.array(pool.starmap(ged_calc_optimize, ged_pair_list))
    #print(distance_list)
    aggregate_list = distance_list - array_rank_transform(perf_diff_list)

    best_aggregate = 0
    best_index = 0
    best_fitness = 0
    for i in range(population_size):
        if aggregate_list[i] > best_aggregate and perf_diff_list[i]!=0:
            best_aggregate = aggregate_list[i]
            best_fitness = population[i][0]
            best_index = i
        elif aggregate_list[i] == best_aggregate and perf_diff_list[i]!=0:
            if population[i][0] > best_fitness:
                best_aggregate = aggregate_list[i]
                best_fitness = population[i][0]
                best_index = i

    return best_candidate, population[best_index]

def crossover_spec_raw(old_spec_1, old_spec_2):
    new_spec = copy.deepcopy(old_spec_1)
    for i in range(len(old_spec_1)):
        if random.random() <= 0.5:
            new_spec[i] = old_spec_2[i]
    return new_spec

def crossover_spec_match(old_spec_1, old_spec_2, trial_max=50):
    dg_1 = nas301_to_digraph(old_spec_1)
    dg_2 = nas301_to_digraph(old_spec_2)
    trial_num = 0
    #offspring_num = 0
    duplicate_num = 0
    graph_valid_num = 0
    while trial_num < trial_max:
        trial_num += 1
        dg_new, ged, edit_path = crossover_edit_path_center_nas301(dg_1, dg_2)
        #print("parents ged: {}".format(ged))
        #ged1, _ = ged_nas101(dg_new, dg_1)
        #print("ged to parent1: {}".format(ged_1))
        #ged2, _ = ged_nas101(dg_new, dg_2)
        #print("ged to parent2: {}".format(ged_2))
        if nx.is_isomorphic(dg_new, dg_1, edge_match=edge_match) or nx.is_isomorphic(dg_new, dg_2, edge_match=edge_match):
            duplicate_num += 1
            continue
        result_spec = digraph_to_nas301(dg_new)
        if type(result_spec) == list:
            graph_valid_num += 1
            return result_spec, ged, (trial_num, duplicate_num, graph_valid_num)
    return None, None, (trial_num, duplicate_num, graph_valid_num)

def crossover_spec_center(old_spec_1, old_spec_2, trial_max=50):
    """Computes a valid child spec via matched crossover between two old_spec."""
    if len(old_spec_1.original_matrix) < len(old_spec_2.original_matrix):
        old_spec_1, old_spec_2 = old_spec_2, old_spec_1
    NUM_VERTICES = len(old_spec_1.original_matrix)
    new_matrix = copy.deepcopy(old_spec_1.original_matrix)
    new_ops = copy.deepcopy(old_spec_1.original_ops)

    dg_1 = nas101_to_digraph(old_spec_1.original_matrix, old_spec_1.original_ops)
    dg_2 = nas101_to_digraph(old_spec_2.original_matrix, old_spec_2.original_ops)
    ged, edit_path = ged_nas101(dg_1, dg_2)
    #print("ged: {}, edit_path: {}".format(ged, edit_path))
    mapping_dict = {}
    for i in range(NUM_VERTICES):
        mapping_dict[edit_path[0][i][0]] = edit_path[0][i][1]

    trial_num = 0
    while trial_num < trial_max:
        trial_num += 1
        sample_indices = random.sample(range(int(ged)), int(np.ceil(ged/2.0)))
        crossover_num = 0

        for src in range(0, NUM_VERTICES - 1):
            for dst in range(src + 1, NUM_VERTICES):
                #if type(mapping_dict[src])!=int or type(mapping_dict[dst])!=int:
                #    print("mapping_dict[src]: {}, mapping_dict[dst]: {}".format(mapping_dict[src], mapping_dict[dst]))
                if (type(mapping_dict[src])==int and type(mapping_dict[dst])==int) and new_matrix[src, dst] != old_spec_2.original_matrix[mapping_dict[src], mapping_dict[dst]]:
                    if crossover_num in sample_indices:
                        new_matrix[src, dst] = old_spec_2.original_matrix[mapping_dict[src], mapping_dict[dst]]
                    crossover_num += 1

        for ind in range(1,NUM_VERTICES - 1):
            if type(mapping_dict[ind])==int and new_ops[ind] != old_spec_2.original_ops[mapping_dict[ind]]:
                if crossover_num in sample_indices:
                    new_ops[ind] = old_spec_2.original_ops[mapping_dict[ind]]
                crossover_num += 1

        #print("sample_indices: {}, crossover_num: {}".format(sample_indices, crossover_num))

        dg_new = nas101_to_digraph(new_matrix, new_ops)
        ged1, _ = ged_nas101(dg_new, dg_1)
        #print("ged to parent1: {}".format(ged))
        ged2, _ = ged_nas101(dg_new, dg_2)
        #print("ged to parent2: {}".format(ged))
        #print("ged type: {}".format(type(ged)))
        if ged1<0.5 or ged2<0.5:
            continue

        new_spec = api.ModelSpec(new_matrix, new_ops)
        if nasbench.is_valid(new_spec):
            #print("parent1: {} , {}".format(old_spec_1.original_matrix, old_spec_1.original_ops))
            #print("parent2: {} , {}".format(old_spec_2.original_matrix, old_spec_2.original_ops))
            #print("offspring: {} , {}".format(new_spec.original_matrix, new_spec.original_ops))
            #print(mapping_dict)
            #print(type(mapping_dict[0])==int)
            #print("ged: {}, node mapping: {}".format(ged, edit_path[0]))
            return new_spec, ged
    return None, None

def crossover_spec_edit_path_center(old_spec_1, old_spec_2, trial_max=50):
    dg_1_normal = nas301_to_digraph(old_spec_1[:NODE_NUM*2])
    dg_2_normal = nas301_to_digraph(old_spec_2[:NODE_NUM*2])
    dg_1_reduce = nas301_to_digraph(old_spec_1[NODE_NUM*2:])
    dg_2_reduce = nas301_to_digraph(old_spec_2[NODE_NUM*2:])
    trial_num = 0
    #offspring_num = 0
    duplicate_num = 0
    graph_valid_num = 0
    while trial_num < trial_max:
        trial_num += 1
        dg_new_normal, ged_normal, edit_path_normal = crossover_edit_path_center_nas301(dg_1_normal, dg_2_normal)
        dg_new_reduce, ged_reduce, edit_path_reduce = crossover_edit_path_center_nas301(dg_1_reduce, dg_2_reduce)
        #print("parents ged: {}".format(ged))
        #ged1, _ = ged_nas101(dg_new, dg_1)
        #print("ged to parent1: {}".format(ged_1))
        #ged2, _ = ged_nas101(dg_new, dg_2)
        #print("ged to parent2: {}".format(ged_2))
        if (nx.is_isomorphic(dg_new_normal, dg_1_normal, edge_match=edge_match) or nx.is_isomorphic(dg_new_normal, dg_2_normal, edge_match=edge_match))and(nx.is_isomorphic(dg_new_reduce, dg_1_reduce, edge_match=edge_match) or nx.is_isomorphic(dg_new_reduce, dg_2_reduce, edge_match=edge_match)):
            duplicate_num += 1
            continue
        result_spec_normal = digraph_to_nas301(dg_new_normal)
        result_spec_reduce = digraph_to_nas301(dg_new_reduce)
        if type(result_spec_normal) == list and type(result_spec_reduce) == list and valid_nas301(result_spec_normal) and valid_nas301(result_spec_reduce):
            graph_valid_num += 1
            return result_spec_normal+result_spec_reduce, ged_normal+ged_reduce, (trial_num, duplicate_num, graph_valid_num)
    return None, None, (trial_num, duplicate_num, graph_valid_num)

def crossover_spec_edit_path_center_general(old_spec_1, old_spec_2, trial_max=50):
    dg_1_normal = nas301_to_digraph(old_spec_1[:NODE_NUM*2])
    dg_2_normal = nas301_to_digraph(old_spec_2[:NODE_NUM*2])
    dg_1_reduce = nas301_to_digraph(old_spec_1[NODE_NUM*2:])
    dg_2_reduce = nas301_to_digraph(old_spec_2[NODE_NUM*2:])
    trial_num = 0
    #offspring_num = 0
    duplicate_num = 0
    graph_valid_num = 0
    while trial_num < trial_max:
        trial_num += 1
        dg_new_normal, ged_normal, edit_path_normal = crossover_edit_path_center_nas301(dg_1_normal, dg_2_normal)
        dg_new_reduce, ged_reduce, edit_path_reduce = crossover_edit_path_center_nas301(dg_1_reduce, dg_2_reduce)
        #print("parents ged: {}".format(ged))
        #ged1, _ = ged_nas101(dg_new, dg_1)
        #print("ged to parent1: {}".format(ged_1))
        #ged2, _ = ged_nas101(dg_new, dg_2)
        #print("ged to parent2: {}".format(ged_2))
        if (nx.is_isomorphic(dg_new_normal, dg_1_normal, edge_match=edge_match) or nx.is_isomorphic(dg_new_normal, dg_2_normal, edge_match=edge_match))and(nx.is_isomorphic(dg_new_reduce, dg_1_reduce, edge_match=edge_match) or nx.is_isomorphic(dg_new_reduce, dg_2_reduce, edge_match=edge_match)):
            duplicate_num += 1
            continue
        result_spec_normal = digraph_to_nas301_general(dg_new_normal)
        result_spec_reduce = digraph_to_nas301_general(dg_new_reduce)
        if type(result_spec_normal) == list and type(result_spec_reduce) == list and valid_nas301(result_spec_normal) and valid_nas301(result_spec_reduce):
            graph_valid_num += 1
            return result_spec_normal+result_spec_reduce, ged_normal+ged_reduce, (trial_num, duplicate_num, graph_valid_num)
    return None, None, (trial_num, duplicate_num, graph_valid_num)

def random_combination(iterable, sample_size):
    """Random selection from itertools.combinations(iterable, r)."""
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.sample(range(n), sample_size))
    return tuple(pool[i] for i in indices)

def run_random_search(max_time_budget=5e6):
    times, best_valids, best_tests, ged_to_best = [0.0], [0.0], [0.0], [10]
    time_spent = 0
    while True:
        spec = random_spec()
        data = data_query_nas301(spec)

        # It's important to select models only based on validation accuracy, test
        # accuracy is used only for comparing different search trajectories.
        if data['validation_accuracy'] > best_valids[-1]:
            best_valids.append(data['validation_accuracy'])
            best_tests.append(data['test_accuracy'])
            ged_to_best.append(ged_to_best_calc(spec))
        else:
            best_valids.append(best_valids[-1])
            best_tests.append(best_tests[-1])
            ged_to_best.append(ged_to_best[-1])

        time_spent += 1
        times.append(time_spent)
        print("\rrandom timestep {}".format(time_spent), end=" ", flush=True)
        if time_spent > max_time_budget:
            # Break the first time we exceed the budget.
            break

    return times, best_valids, best_tests, ged_to_best

def run_evolution_search(max_time_budget=5e6,
                         population_size=50,
                         tournament_size=10,
                         mutation_rate=1.0):
    """Run a single roll-out of regularized evolution to a fixed time budget."""
    times, best_valids, best_tests, ged_to_best = [0.0], [0.0], [0.0], [10]
    population = []   # (validation, spec) tuples

    # For the first population_size individuals, seed the population with randomly
    # generated cells.
    time_spent = 0
    for _ in range(population_size):
        spec = random_spec()
        data = data_query_nas301(spec)
        time_spent += 1
        times.append(time_spent)
        population.append((data['validation_accuracy'], spec))

        if data['validation_accuracy'] > best_valids[-1]:
            best_valids.append(data['validation_accuracy'])
            best_tests.append(data['test_accuracy'])
            ged_to_best.append(ged_to_best_calc(spec))
        else:
            best_valids.append(best_valids[-1])
            best_tests.append(best_tests[-1])
            ged_to_best.append(ged_to_best[-1])

        if time_spent > max_time_budget:
            break

    # After the population is seeded, proceed with evolving the population.
    while True:
        sample = random_combination(population, tournament_size)
        best_spec = sorted(sample, key=lambda i:i[0])[-1][1]
        new_spec = mutate_spec(best_spec[:NODE_NUM*2], mutation_rate) + mutate_spec(best_spec[NODE_NUM*2:], mutation_rate)

        data = data_query_nas301(new_spec)
        time_spent += 1
        times.append(time_spent)

        # In regularized evolution, we kill the oldest individual in the population.
        population.append((data['validation_accuracy'], new_spec))
        population.pop(0)

        if data['validation_accuracy'] > best_valids[-1]:
            best_valids.append(data['validation_accuracy'])
            best_tests.append(data['test_accuracy'])
            ged_to_best.append(ged_to_best_calc(new_spec))
        else:
            best_valids.append(best_valids[-1])
            best_tests.append(best_tests[-1])
            ged_to_best.append(ged_to_best[-1])
        print("\rmutation timestep {}".format(time_spent), end=" ", flush=True)

        if time_spent > max_time_budget:
            break

    return times, best_valids, best_tests, ged_to_best

def run_evolution_search_raw_crossover(max_time_budget=5e6,
                         population_size=50,
                         tournament_size=10,
                         mutation_rate=1.0):
    """Run a single roll-out of regularized evolution to a fixed time budget."""
    times, best_valids, best_tests, ged_to_best = [0.0], [0.0], [0.0], [10]
    population = []   # (validation, spec) tuples

    # For the first population_size individuals, seed the population with randomly
    # generated cells.
    time_spent = 0
    for _ in range(population_size):
        spec = random_spec()
        data = data_query_nas301(spec)
        time_spent += 1
        times.append(time_spent)
        population.append((data['validation_accuracy'], spec))

        if data['validation_accuracy'] > best_valids[-1]:
            best_valids.append(data['validation_accuracy'])
            best_tests.append(data['test_accuracy'])
            ged_to_best.append(ged_to_best_calc(spec))
        else:
            best_valids.append(best_valids[-1])
            best_tests.append(best_tests[-1])
            ged_to_best.append(ged_to_best[-1])

        if time_spent > max_time_budget:
            break

    # After the population is seeded, proceed with evolving the population.
    while True:

        # mutation
        sample = random_combination(population, tournament_size)
        best_spec = sorted(sample, key=lambda i:i[0])[-1][1]
        new_spec = mutate_spec(best_spec[:NODE_NUM*2], mutation_rate) + mutate_spec(best_spec[NODE_NUM*2:], mutation_rate)

        data = data_query_nas301(new_spec)
        time_spent += 1
        times.append(time_spent)

        # In regularized evolution, we kill the oldest individual in the population.
        population.append((data['validation_accuracy'], new_spec))
        population.pop(0)

        if data['validation_accuracy'] > best_valids[-1]:
            best_valids.append(data['validation_accuracy'])
            best_tests.append(data['test_accuracy'])
            ged_to_best.append(ged_to_best_calc(new_spec))
        else:
            best_valids.append(best_valids[-1])
            best_tests.append(best_tests[-1])
            ged_to_best.append(ged_to_best[-1])

        if time_spent > max_time_budget:
            break

        #crossover
        while True:
            sample = random_combination(population, tournament_size)
            best_spec_1 = sorted(sample, key=lambda i:i[0])[-1][1]
            sample = random_combination(population, tournament_size)
            best_spec_2 = sorted(sample, key=lambda i:i[0])[-1][1]
            if not np.array_equal(best_spec_1, best_spec_2):
                break

        trial_num_max = 50
        trial_num = 0
        valid = False
        while trial_num < trial_num_max:
            new_spec = crossover_spec_raw(best_spec_1, best_spec_2)
            if valid_nas301(new_spec[:NODE_NUM*2]) and valid_nas301(new_spec[NODE_NUM*2:]):
                valid = True
                break
            trial_num += 1

        if not valid:
            continue

        data = data_query_nas301(new_spec)
        time_spent += 1
        times.append(time_spent)
        print("\rraw_crossover timestep {}".format(time_spent), end=" ", flush=True)

        # In regularized evolution, we kill the oldest individual in the population.
        population.append((data['validation_accuracy'], new_spec))
        population.pop(0)

        if data['validation_accuracy'] > best_valids[-1]:
            best_valids.append(data['validation_accuracy'])
            best_tests.append(data['test_accuracy'])
            ged_to_best.append(ged_to_best_calc(new_spec))
        else:
            best_valids.append(best_valids[-1])
            best_tests.append(best_tests[-1])
            ged_to_best.append(ged_to_best[-1])

        if time_spent > max_time_budget:
            break


    return times, best_valids, best_tests, ged_to_best

def run_evolution_search_matched_crossover(max_time_budget=5e6,
                         population_size=50,
                         tournament_size=10,
                         mutation_rate=1.0):
    """Run a single roll-out of regularized evolution to a fixed time budget."""
    times, best_valids, best_tests, ged_to_best, parents_ged = [0.0], [0.0], [0.0], [10], [10]
    population = []   # (validation, spec) tuples
    status_list = [] # int for mutation, tuple for crossover

    # For the first population_size individuals, seed the population with randomly
    # generated cells.
    time_spent = 0
    for _ in range(population_size):
        spec = random_spec()
        data = data_query_nas301(spec)
        time_spent += 1
        times.append(time_spent)
        population.append((data['validation_accuracy'], spec))

        if data['validation_accuracy'] > best_valids[-1]:
            best_valids.append(data['validation_accuracy'])
            best_tests.append(data['test_accuracy'])
            ged_to_best.append(ged_to_best_calc(spec))
        else:
            best_valids.append(best_valids[-1])
            best_tests.append(best_tests[-1])
            ged_to_best.append(ged_to_best[-1])
        parents_ged.append(10)

        if time_spent > max_time_budget:
            break

    # After the population is seeded, proceed with evolving the population.
    while True:
        # mutation
        sample = random_combination(population, tournament_size)
        best_spec = sorted(sample, key=lambda i:i[0])[-1][1]
        new_spec = mutate_spec(best_spec[:NODE_NUM*2], mutation_rate) + mutate_spec(best_spec[NODE_NUM*2:], mutation_rate)

        data = data_query_nas301(new_spec)
        time_spent += 1
        status_list.append(time_spent)
        times.append(time_spent)

        # In regularized evolution, we kill the oldest individual in the population.
        population.append((data['validation_accuracy'], new_spec))
        population.pop(0)

        if data['validation_accuracy'] > best_valids[-1]:
            best_valids.append(data['validation_accuracy'])
            best_tests.append(data['test_accuracy'])
            ged_to_best.append(ged_to_best_calc(new_spec))
        else:
            best_valids.append(best_valids[-1])
            best_tests.append(best_tests[-1])
            ged_to_best.append(ged_to_best[-1])
        parents_ged.append(10)

        if time_spent > max_time_budget:
            break

        #crossover
        while True:
            sample = random_combination(population, tournament_size)
            best_spec_1 = sorted(sample, key=lambda i:i[0])[-1][1]
            sample = random_combination(population, tournament_size)
            best_spec_2 = sorted(sample, key=lambda i:i[0])[-1][1]
            if not np.array_equal(best_spec_1, best_spec_2):
                break

        #new_spec, crossover_ged = crossover_spec_center(best_spec_1, best_spec_2)
        new_spec, crossover_ged, num_stats = crossover_spec_edit_path_center(best_spec_1, best_spec_2)
        status_list.append(num_stats)
        if type(crossover_ged)!=np.float64:
            continue

        #print(new_spec.original_matrix)
        #print(new_spec.original_ops)
        #print(nasbench.is_valid(new_spec))
        data = data_query_nas301(new_spec)
        time_spent += 1
        times.append(time_spent)
        print("\rcrossover timestep {}".format(time_spent), end=" ", flush=True)

        # In regularized evolution, we kill the oldest individual in the population.
        population.append((data['validation_accuracy'], new_spec))
        population.pop(0)

        if data['validation_accuracy'] > best_valids[-1]:
            best_valids.append(data['validation_accuracy'])
            best_tests.append(data['test_accuracy'])
            ged_to_best.append(ged_to_best_calc(new_spec))
        else:
            best_valids.append(best_valids[-1])
            best_tests.append(best_tests[-1])
            ged_to_best.append(ged_to_best[-1])
        parents_ged.append(crossover_ged)

        if time_spent > max_time_budget:
            break

    return times, best_valids, best_tests, ged_to_best, parents_ged, status_list


def run_evolution_search_refined_crossover(max_time_budget=5e6,
                         population_size=50,
                         tournament_size=10,
                         mutation_rate=1.0,
                         best_survival=False):
    """Run a single roll-out of regularized evolution to a fixed time budget."""
    times, best_valids, best_tests, ged_to_best, parents_ged = [0.0], [0.0], [0.0], [10], [10]
    population = []   # (validation, spec) tuples
    status_list = [] # int for mutation, tuple for crossover

    # For the first population_size individuals, seed the population with randomly
    # generated cells.
    time_spent = 0
    for _ in range(population_size):
        spec = random_spec()
        data = data_query_nas301(spec)
        time_spent += 1
        times.append(time_spent)
        population.append((data['validation_accuracy'], spec))

        if data['validation_accuracy'] > best_valids[-1]:
            best_valids.append(data['validation_accuracy'])
            best_tests.append(data['test_accuracy'])
            ged_to_best.append(ged_to_best_calc(spec))
        else:
            best_valids.append(best_valids[-1])
            best_tests.append(best_tests[-1])
            ged_to_best.append(ged_to_best[-1])
        parents_ged.append(10)

        if time_spent > max_time_budget:
            break

    # After the population is seeded, proceed with evolving the population.
    while True:
        # mutation
        sample = random_combination(population, tournament_size)
        best_spec = sorted(sample, key=lambda i:i[0])[-1][1]
        new_spec = mutate_spec(best_spec[:NODE_NUM*2], mutation_rate) + mutate_spec(best_spec[NODE_NUM*2:], mutation_rate)

        data = data_query_nas301(new_spec)
        time_spent += 1
        status_list.append(time_spent)
        times.append(time_spent)

        # In regularized evolution, we kill the oldest individual in the population.
        population.append((data['validation_accuracy'], new_spec))
        population.pop(0)

        if data['validation_accuracy'] > best_valids[-1]:
            best_valids.append(data['validation_accuracy'])
            best_tests.append(data['test_accuracy'])
            ged_to_best.append(ged_to_best_calc(new_spec))
        else:
            best_valids.append(best_valids[-1])
            best_tests.append(best_tests[-1])
            ged_to_best.append(ged_to_best[-1])
        parents_ged.append(10)

        if time_spent > max_time_budget:
            break

        #crossover
        while True:
            sample = random_combination(population, tournament_size)
            best_spec_1 = sorted(sample, key=lambda i:i[0])[-1][1]
            sample = random_combination(population, tournament_size)
            best_spec_2 = sorted(sample, key=lambda i:i[0])[-1][1]
            if not np.array_equal(best_spec_1, best_spec_2):
                break

        #new_spec, crossover_ged = crossover_spec_center(best_spec_1, best_spec_2)
        new_spec, crossover_ged, num_stats = crossover_spec_edit_path_center_general(best_spec_1, best_spec_2)
        status_list.append(num_stats)
        if type(crossover_ged)!=np.float64:
            continue

        #print(new_spec.original_matrix)
        #print(new_spec.original_ops)
        #print(nasbench.is_valid(new_spec))
        data = data_query_nas301(new_spec)
        time_spent += 1
        times.append(time_spent)
        print("\rcrossover timestep {}".format(time_spent), end=" ", flush=True)

        # In regularized evolution, we kill the oldest individual in the population.
        population.append((data['validation_accuracy'], new_spec))
        population.pop(0)

        if data['validation_accuracy'] > best_valids[-1]:
            best_valids.append(data['validation_accuracy'])
            best_tests.append(data['test_accuracy'])
            ged_to_best.append(ged_to_best_calc(new_spec))
        else:
            best_valids.append(best_valids[-1])
            best_tests.append(best_tests[-1])
            ged_to_best.append(ged_to_best[-1])
        parents_ged.append(crossover_ged)

        if time_spent > max_time_budget:
            break

    return times, best_valids, best_tests, ged_to_best, parents_ged, status_list

# Run random search and evolution search 10 times each. This should take a few
# minutes to run. Note that each run would have taken days of compute to
# actually train and evaluate if the dataset were not precomputed.
random_data = []
evolution_mutation_data = []
evolution_crossover_raw_data = []
evolution_crossover_match_data = []
evolution_crossover_refined_data = []
population_size = 100
tournament_size = 10
repeat_num = 30
max_time_budget = int(2000)
for repeat in range(repeat_num):
    print('Running repeat %d' % (repeat + 1))
    times, best_valid, best_test, ged_to_best = run_random_search(max_time_budget=max_time_budget)
    random_data.append((times, best_valid, best_test, ged_to_best))

    times, best_valid, best_test, ged_to_best = run_evolution_search(max_time_budget=max_time_budget, population_size=population_size, tournament_size=tournament_size)
    evolution_mutation_data.append((times, best_valid, best_test, ged_to_best))

    print("crossover_raw start")
    times, best_valid, best_test, ged_to_best = run_evolution_search_raw_crossover(max_time_budget=max_time_budget, population_size=population_size, tournament_size=tournament_size)
    evolution_crossover_raw_data.append((times, best_valid, best_test, ged_to_best))
    
    print("crossover_match start")
    times, best_valid, best_test, ged_to_best, parents_ged, status_list = run_evolution_search_matched_crossover(max_time_budget=max_time_budget, population_size=population_size, tournament_size=tournament_size)
    evolution_crossover_match_data.append((times, best_valid, best_test, ged_to_best, parents_ged, status_list))
    
    print("crossover_refined start")
    times, best_valid, best_test, ged_to_best, parents_ged, status_list = run_evolution_search_refined_crossover(max_time_budget=max_time_budget, population_size=population_size, tournament_size=tournament_size)
    evolution_crossover_refined_data.append((times, best_valid, best_test, ged_to_best, parents_ged, status_list))

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','nas301_edit_path_center_crossover_NoIsomorphism_stats_pop{}_tournament{}_run{}_evaluation{}.pkl'.format(
                                                population_size, tournament_size, repeat_num, max_time_budget))
with open(result_file_name, 'wb') as result_file:
    pickle.dump((random_data, evolution_mutation_data, evolution_crossover_raw_data, evolution_crossover_match_data, evolution_crossover_refined_data), result_file)
    #pickle.dump((random_data, evolution_mutation_data, evolution_crossover_raw_data, evolution_crossover_refined_data), result_file)

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
moving_average_data = []
f = plt.figure()
for times, best_valid, best_test, _, parents_ged, _ in evolution_crossover_refined_data:
    moving_average = np.convolve(parents_ged, np.ones(window_size)/window_size, mode='valid')
    plt.plot(times[window_size-1:], moving_average, label='parents_ged', alpha=0.5)
    moving_average_data.append((times[window_size-1:], moving_average))
plt.ylabel('ged_parents')
plt.xlabel('total number of candidates evaluated')
plt.grid()
plt.title('graph edit distance between parents')
plt.show()
plot_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Plots','ged_parents_center_crossover_pop{}_tournament{}_run{}_evaluation{}.pdf'.format(
                                                population_size, tournament_size, repeat_num, max_time_budget))
f.savefig(plot_file_name, bbox_inches='tight')

# Compare the mean test accuracy along with error bars.
def plot_data(data, color, label, gran=1, max_budget=5000000):
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
            prev_test = data[repeat][2][pointers[repeat]-1]
            next_time = data[repeat][0][pointers[repeat]]
            next_test = data[repeat][2][pointers[repeat]]
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
        mean.append(sum(all_vals) / float(len(all_vals)))
        per25.append(all_vals[int(0.25 * repeats)])
        per75.append(all_vals[int(0.75 * repeats)])

        cur += gran

    plt.plot(xs, mean, color=color, label=label, linewidth=2)
    plt.fill_between(xs, per25, per75, alpha=0.1, linewidth=0, facecolor=color)

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
plot_data(random_data, 'red', 'random', max_budget=max_time_budget)
plot_data(evolution_mutation_data, 'blue', 'evolution_mutation', max_budget=max_time_budget)
plot_data(evolution_crossover_raw_data, 'g', 'evolution_crossover_raw', max_budget=max_time_budget)
plot_data(evolution_crossover_match_data, 'y', 'evolution_crossover_center_general', max_budget=max_time_budget)
plot_data(evolution_crossover_refined_data, 'c', 'evolution_crossover_center', max_budget=max_time_budget)
plt.legend(loc='lower right')
plt.xlabel('total number of candidates evaluated')
plt.ylabel('test accuracy')
plt.grid()
plt.show()
plot_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Plots','test_accuracy_center_crossover_pop{}_tournament{}_run{}_evaluation{}.pdf'.format(
                                                population_size, tournament_size, repeat_num, max_time_budget))
f.savefig(plot_file_name, bbox_inches='tight')

f = plt.figure()
plot_data_val(random_data, 'red', 'random', max_budget=max_time_budget)
plot_data_val(evolution_mutation_data, 'blue', 'evolution_mutation', max_budget=max_time_budget)
plot_data_val(evolution_crossover_raw_data, 'g', 'evolution_crossover_raw', max_budget=max_time_budget)
plot_data_val(evolution_crossover_match_data, 'y', 'evolution_crossover_center_general', max_budget=max_time_budget)
plot_data_val(evolution_crossover_refined_data, 'c', 'evolution_crossover_center', max_budget=max_time_budget)
plt.legend(loc='lower right')
plt.xlabel('total number of candidates evaluated')
plt.ylabel('validation accuracy')
plt.grid()
plt.show()
plot_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Plots','validation_accuracy_center_crossover_pop{}_tournament{}_run{}_evaluation{}.pdf'.format(
                                                population_size, tournament_size, repeat_num, max_time_budget))
f.savefig(plot_file_name, bbox_inches='tight')

f = plt.figure()
plot_data_ged_to_best(random_data, 'red', 'random', max_budget=max_time_budget)
plot_data_ged_to_best(evolution_mutation_data, 'blue', 'evolution_mutation', max_budget=max_time_budget)
plot_data_ged_to_best(evolution_crossover_raw_data, 'g', 'evolution_crossover_raw', max_budget=max_time_budget)
plot_data_ged_to_best(evolution_crossover_match_data, 'y', 'evolution_crossover_match', max_budget=max_time_budget)
plot_data_ged_to_best(evolution_crossover_refined_data, 'c', 'evolution_crossover_refined', max_budget=max_time_budget)
plt.legend(loc='lower right')
plt.xlabel('total number of candidates evaluated')
plt.ylabel('ged_to_best')
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
plot_data_parents_ged_moving_average(moving_average_data, 'y', 'evolution_crossover_match', max_budget=max_time_budget)
plot_data_parents_ged_moving_average(evolution_crossover_refined_data, 'c', 'evolution_crossover_refined', max_budget=max_time_budget)
plt.legend(loc='lower right')
plt.xlabel('total number of candidates evaluated')
plt.ylabel('parents ged (moving average)')
plt.grid()
plt.show()
plot_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Plots','ged_parents_movingaverage_center_crossover_pop{}_tournament{}_run{}_evaluation{}.pdf'.format(
                                                population_size, tournament_size, repeat_num, max_time_budget))
f.savefig(plot_file_name, bbox_inches='tight')
