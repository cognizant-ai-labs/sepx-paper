# Standard imports
import copy
import numpy as np
import matplotlib.pyplot as plt
import random
from ged_util_nasnlp_theory import nasnlp_to_digraph, ged_nasnlp, ged_nasnlp_optimize, crossover_edit_path_center, digraph_to_nasnlp, node_match, nasnlp_valid, sample_random_architecture, mutate_recipe, arch_to_compact, ged_nasnlp_edge_only
#from multiprocessing import mp
import multiprocessing
import pickle
import os
import networkx as nx
import time

from conversions_util import convert_recipe_to_compact, convert_compact_to_recipe
from encodings_nlp_util import encode_nlp

NUM_PRESERVE_CPU = 3
'''
target_recipe = {
                'i':{'op':'linear', 'input':['x', 'h_prev_0']},
                'i_act':{'op':'activation_tanh', 'input':['i']},

                'j':{'op':'linear', 'input':['x', 'h_prev_0']},
                'j_act':{'op':'activation_sigm', 'input':['j']},

                'f':{'op':'linear', 'input':['x', 'h_prev_0']},
                'f_act':{'op':'activation_sigm', 'input':['f']},

                'o':{'op':'linear', 'input':['x', 'h_prev_0']},
                'o_act':{'op':'activation_tanh', 'input':['o']},

                'h_new_1_part1':{'op':'elementwise_prod', 'input':['f_act', 'h_prev_1']},
                'h_new_1_part2':{'op':'elementwise_prod', 'input':['i_act', 'j_act']},

                'h_new_1':{'op':'elementwise_sum', 'input':['h_new_1_part1', 'h_new_1_part2']},

                'h_new_1_act':{'op':'activation_tanh', 'input':['h_new_1']},
                'h_new_0':{'op':'elementwise_prod', 'input':['h_new_1_act', 'o_act']}
                }
'''
target_recipe = {
            'r':{'op':'linear', 'input':['x', 'h_prev_0']},
            'r_act':{'op':'activation_sigm', 'input':['r']},

            'z':{'op':'linear', 'input':['x', 'h_prev_0']},
            'z_act':{'op':'activation_sigm', 'input':['z']},

            'rh':{'op':'elementwise_prod', 'input':['r_act', 'h_prev_0']},
            'h_tilde':{'op':'linear', 'input':['x', 'rh']},
            'h_tilde_act':{'op':'activation_tanh', 'input':['h_tilde']},

            'h_new_0':{'op':'blend', 'input':['z_act', 'h_prev_0', 'h_tilde_act']}
            }

dg_target = nasnlp_to_digraph(target_recipe)

def query(recipe, train_partial=False):
    data = {}
    data['validation_accuracy'] = 94
    data['test_accuracy'] = 94
    return data


def theory_stats(spec_1, spec_2):
    dg_1 = nasnlp_to_digraph(spec_1)
    dg_2 = nasnlp_to_digraph(spec_2)
    ged_parents_edge_only = ged_nasnlp_edge_only(dg_1, dg_2)
    ged_to_best_1_edge_only = ged_nasnlp_edge_only(dg_1, dg_target)
    ged_to_best_2_edge_only = ged_nasnlp_edge_only(dg_2, dg_target)
    num_edges_1 = dg_1.number_of_edges()
    num_edges_2 = dg_2.number_of_edges()
    return ged_parents_edge_only, ged_to_best_1_edge_only, ged_to_best_2_edge_only, num_edges_1, num_edges_2

def ged_to_best_calc(new_spec):
    dg_new = nasnlp_to_digraph(new_spec)
    if len(dg_new.nodes) < len(dg_target.nodes):
        dg_1, dg_2 = dg_target, dg_new
    else:
        dg_1, dg_2 = dg_new, dg_target
    ged, edit_path = ged_nasnlp(dg_1, dg_2)
    return ged

def ged_calc(spec_1, spec_2):
    dg_1 = nasnlp_to_digraph(spec_1)
    dg_2 = nasnlp_to_digraph(spec_2)
    ged, _ = ged_nasnlp(dg_1, dg_2)
    return ged

def ged_calc_optimize(spec_1, spec_2):
    dg_1 = nasnlp_to_digraph(spec_1)
    dg_2 = nasnlp_to_digraph(spec_2)
    ged = ged_nasnlp_optimize(dg_1, dg_2)
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

def crossover_spec_raw(recipe_1, recipe_2):
    compact_1 = convert_recipe_to_compact(recipe_1)
    arch_1 = encode_nlp(compact_1, 12, None, lc_feature=False)
    compact_2 = convert_recipe_to_compact(recipe_2)
    arch_2 = encode_nlp(compact_2, 12, None, lc_feature=False)
    if len(arch_1) == len(arch_2):
        for i in range(len(arch_1)):
            if random.random() <= 0.5:
                arch_1[i] = arch_2[i]
    #print(arch_1)
    compact_new = arch_to_compact(arch_1)
    #print(compact_new)
    recipe_new = convert_compact_to_recipe(compact_new)
    return recipe_new


def crossover_spec_match(old_spec_1, old_spec_2, trial_max=50):
    """Computes a valid child spec via matched crossover between two old_spec."""
    if len(old_spec_1.original_matrix) < len(old_spec_2.original_matrix):
        old_spec_1, old_spec_2 = old_spec_2, old_spec_1
    NUM_VERTICES = len(old_spec_1.original_matrix)
    new_matrix = copy.deepcopy(old_spec_1.original_matrix)
    new_ops = copy.deepcopy(old_spec_1.original_ops)

    dg_1 = nas101_to_digraph(old_spec_1.original_matrix, old_spec_1.original_ops)
    dg_2 = nas101_to_digraph(old_spec_2.original_matrix, old_spec_2.original_ops)
    ged, edit_path = ged_nas101(dg_1, dg_2)
    mapping_dict = {}
    for i in range(NUM_VERTICES):
        mapping_dict[edit_path[0][i][0]] = edit_path[0][i][1]

    trial_num = 0
    duplicate_num = 0
    while trial_num < trial_max:
        trial_num += 1
        for src in range(0, NUM_VERTICES - 1):
            for dst in range(src + 1, NUM_VERTICES):
                if random.random() <= 0.5 and (type(mapping_dict[src])==int and type(mapping_dict[dst])==int):
                    new_matrix[src, dst] = old_spec_2.original_matrix[mapping_dict[src], mapping_dict[dst]]

        for ind in range(1,NUM_VERTICES - 1):
            if random.random() <= 0.5 and type(mapping_dict[ind])==int:
                new_ops[ind] = old_spec_2.original_ops[mapping_dict[ind]]
        dg_new = nas101_to_digraph(new_matrix, new_ops)
        #ged1, _ = ged_nas101(dg_new, dg_1)
        #print("ged to parent1: {}".format(ged))
        #ged2, _ = ged_nas101(dg_new, dg_2)
        #print("ged to parent2: {}".format(ged))
        #print("ged type: {}".format(type(ged)))
        if nx.is_isomorphic(dg_new, dg_1, node_match=node_match) or nx.is_isomorphic(dg_new, dg_2, node_match=node_match):
            duplicate_num += 1
            continue
        new_spec = api.ModelSpec(new_matrix, new_ops)
        if nasbench.is_valid(new_spec):
            #print("parent1: {} , {}".format(old_spec_1.original_matrix, old_spec_1.original_ops))
            #print("parent2: {} , {}".format(old_spec_2.original_matrix, old_spec_2.original_ops))
            #print("offspring: {} , {}".format(new_spec.original_matrix, new_spec.original_ops))
            #print(mapping_dict)
            #print(type(mapping_dict[0])==int)
            #print("ged: {}, node mapping: {}".format(ged, edit_path[0]))
            return new_spec, ged, (trial_num, duplicate_num)
    return None, None, (trial_num, duplicate_num)

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
    dg_1 = nasnlp_to_digraph(old_spec_1)
    dg_2 = nasnlp_to_digraph(old_spec_2)
    if len(dg_1.nodes) < len(dg_2.nodes):
        dg_1, dg_2 = dg_2, dg_1
    ged, edit_path = ged_nasnlp(dg_1, dg_2)
    trial_num = 0
    #offspring_num = 0
    duplicate_num = 0
    graph_valid_num = 0
    while trial_num < trial_max:
        trial_num += 1
        dg_new = crossover_edit_path_center(dg_1, dg_2, ged, edit_path)
        #print("parents ged: {}".format(ged))
        #ged1, _ = ged_nas101(dg_new, dg_1)
        #print("ged to parent1: {}".format(ged_1))
        #ged2, _ = ged_nas101(dg_new, dg_2)
        #print("ged to parent2: {}".format(ged_2))
        if nx.is_isomorphic(dg_new, dg_1, node_match=node_match) or nx.is_isomorphic(dg_new, dg_2, node_match=node_match):
            duplicate_num += 1
            continue
        result_spec = digraph_to_nasnlp(dg_new)
        if type(result_spec) == dict:
            graph_valid_num += 1
            if nasnlp_valid(dg_new):
                #print("parent1: {} , {}".format(old_spec_1.original_matrix, old_spec_1.original_ops))
                #print("parent2: {} , {}".format(old_spec_2.original_matrix, old_spec_2.original_ops))
                #print("offspring: {} , {}".format(new_spec.original_matrix, new_spec.original_ops))
                #print(mapping_dict)
                #print(type(mapping_dict[0])==int)
                #print("ged: {}, node mapping: {}".format(ged, edit_path[0]))
                return result_spec, ged, (trial_num, duplicate_num, graph_valid_num)
    return None, None, (trial_num, duplicate_num, graph_valid_num)

def random_combination(iterable, sample_size):
    """Random selection from itertools.combinations(iterable, r)."""
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.sample(range(n), sample_size))
    return tuple(pool[i] for i in indices)


def random_search_single(time_spent, max_time_budget, times, best_valids, best_tests, ged_to_best):
    while time_spent.value < max_time_budget:
        spec = sample_random_architecture()
        #print(spec)
        data = query(spec)
        time_start = time.time()
        ged = ged_to_best_calc(spec)
        #print("spec: {}, ged_to_best: {}".format(spec, ged))

        # It's important to select models only based on validation accuracy, test
        # accuracy is used only for comparing different search trajectories.
        if ged < ged_to_best[-1]:
            best_valids.append(data['validation_accuracy'])
            best_tests.append(data['test_accuracy'])
            ged_to_best.append(ged)
        else:
            best_valids.append(best_valids[-1])
            best_tests.append(best_tests[-1])
            ged_to_best.append(ged_to_best[-1])
        time_spent.value += 1
        times.append(time_spent.value)
        #print("trial {}, ged_to_best: {}, computation time: {}".format(time_spent.value, ged, time.time()-time_start))

def run_random_search_mp(max_time_budget=5e6):
    """Run a single roll-out of random search to a fixed time budget."""
    #manager = mp.Manager()
    manager = multiprocessing.Manager()
    times, best_valids, best_tests, ged_to_best = manager.list([0.0]), manager.list([0.0]), manager.list([0.0]), manager.list([30])
    time_spent = manager.Value('i', 0)
    process_list =[]
    print("multiprocessing.cpu_count(): {}".format(multiprocessing.cpu_count()))
    for cpu_id in range(multiprocessing.cpu_count()-NUM_PRESERVE_CPU):
        process_list.append(multiprocessing.Process(target=random_search_single, args=(time_spent, max_time_budget, times, best_valids, best_tests, ged_to_best)))
    for process in process_list:
        process.start()
    for process in process_list:
        process.join()

    return list(times), list(best_valids), list(best_tests), list(ged_to_best)

def initialize_population_single(population_size, population, best_valids, best_tests, ged_to_best):
    while len(population) < population_size:
        spec = sample_random_architecture()
        data = query(spec)
        ged = ged_to_best_calc(spec)
        population.append((-ged, spec))

        if -population[-1][0] < ged_to_best[-1]:
            best_valids.append(data['validation_accuracy'])
            best_tests.append(data['test_accuracy'])
            ged_to_best.append(ged)
        else:
            best_valids.append(best_valids[-1])
            best_tests.append(best_tests[-1])
            ged_to_best.append(ged_to_best[-1])
        #print("len(population): {}".format(len(population)))

def run_evolution_search_single(population, tournament_size, mutation_rate, time_spent, max_time_budget,
                                times, best_valids, best_tests, ged_to_best):
    while time_spent.value < max_time_budget:
        sample = random_combination(population, tournament_size)
        best_spec = sorted(sample, key=lambda i:i[0])[-1][1]
        new_spec = mutate_recipe(best_spec, mutation_rate)

        data = query(new_spec)
        time_spent.value += 1
        times.append(time_spent.value)

        # In regularized evolution, we kill the oldest individual in the population.
        ged = ged_to_best_calc(new_spec)
        population.append((-ged, new_spec))
        population.pop(0)

        if -population[-1][0] < ged_to_best[-1]:
            best_valids.append(data['validation_accuracy'])
            best_tests.append(data['test_accuracy'])
            ged_to_best.append(ged)
        else:
            best_valids.append(best_valids[-1])
            best_tests.append(best_tests[-1])
            ged_to_best.append(ged_to_best[-1])

def run_evolution_search_mp(max_time_budget=5e6,
                         population_size=50,
                         tournament_size=10,
                         mutation_rate=1.0):
    manager = multiprocessing.Manager()
    times, best_valids, best_tests, ged_to_best = manager.list([0.0]), manager.list([0.0]), manager.list([0.0]), manager.list([30])
    population = manager.list([])
    time_spent = manager.Value('i', 0)
    process_list =[]
    print("multiprocessing.cpu_count(): {}".format(multiprocessing.cpu_count()))
    for cpu_id in range(multiprocessing.cpu_count()-NUM_PRESERVE_CPU):
        process_list.append(multiprocessing.Process(target=initialize_population_single, args=(population_size, population, best_valids, best_tests, ged_to_best)))
    for process in process_list:
        process.start()
    for process in process_list:
        process.join()
    for _ in range(population_size):
        time_spent.value += 1
        times.append(time_spent.value)
    while len(population) > population_size:
        population.pop(0)

    process_list =[]
    print("multiprocessing.cpu_count(): {}".format(multiprocessing.cpu_count()))
    for cpu_id in range(multiprocessing.cpu_count()-NUM_PRESERVE_CPU):
        process_list.append(multiprocessing.Process(target=run_evolution_search_single, args=(population, tournament_size, mutation_rate, time_spent, max_time_budget,
                                                                                            times, best_valids, best_tests, ged_to_best)))
    for process in process_list:
        process.start()
    for process in process_list:
        process.join()

    return list(times), list(best_valids), list(best_tests), list(ged_to_best)

def run_evolution_search_raw_crossover_single(population, tournament_size, mutation_rate, time_spent, max_time_budget,
                                                times, best_valids, best_tests, ged_to_best):
    while time_spent.value < max_time_budget:
        # mutation
        sample = random_combination(population, tournament_size)
        best_spec = sorted(sample, key=lambda i:i[0])[-1][1]
        new_spec = mutate_recipe(best_spec, mutation_rate)

        data = query(new_spec)
        time_spent.value += 1
        times.append(time_spent.value)
        #print("time_spent.value: {}".format(time_spent.value))

        # In regularized evolution, we kill the oldest individual in the population.
        ged = ged_to_best_calc(new_spec)
        population.append((-ged, new_spec))
        population.pop(0)

        if -population[-1][0] < ged_to_best[-1]:
            best_valids.append(data['validation_accuracy'])
            best_tests.append(data['test_accuracy'])
            ged_to_best.append(ged)
        else:
            best_valids.append(best_valids[-1])
            best_tests.append(best_tests[-1])
            ged_to_best.append(ged_to_best[-1])

        if time_spent.value >= max_time_budget:
            break

        #crossover
        while True:
            sample = random_combination(population, tournament_size)
            best_spec_1 = sorted(sample, key=lambda i:i[0])[-1][1]
            sample = random_combination(population, tournament_size)
            best_spec_2 = sorted(sample, key=lambda i:i[0])[-1][1]
            if best_spec_1 != best_spec_2:
                break

        trial_num_max = 50
        trial_num = 0
        valid = False
        while trial_num < trial_num_max:
            trial_num += 1
            new_spec = crossover_spec_raw(best_spec_1, best_spec_2)
            if new_spec == best_spec_1 or new_spec == best_spec_2:
                continue
            if nasnlp_valid(nasnlp_to_digraph(new_spec)):
                valid = True
                break

        if not valid:
            continue

        data = query(new_spec)
        time_spent.value += 1
        times.append(time_spent.value)
        #print("time_spent.value: {}".format(time_spent.value))

        # In regularized evolution, we kill the oldest individual in the population.
        ged = ged_to_best_calc(new_spec)
        population.append((-ged, new_spec))
        population.pop(0)

        if -population[-1][0] < ged_to_best[-1]:
            best_valids.append(data['validation_accuracy'])
            best_tests.append(data['test_accuracy'])
            ged_to_best.append(ged)
        else:
            best_valids.append(best_valids[-1])
            best_tests.append(best_tests[-1])
            ged_to_best.append(ged_to_best[-1])

def run_evolution_search_raw_crossover_mp(max_time_budget=5e6,
                         population_size=50,
                         tournament_size=10,
                         mutation_rate=1.0):
    manager = multiprocessing.Manager()
    times, best_valids, best_tests, ged_to_best = manager.list([0.0]), manager.list([0.0]), manager.list([0.0]), manager.list([30])
    population = manager.list([])
    time_spent = manager.Value('i', 0)
    process_list =[]
    print("multiprocessing.cpu_count(): {}".format(multiprocessing.cpu_count()))
    for cpu_id in range(multiprocessing.cpu_count()-NUM_PRESERVE_CPU):
        process_list.append(multiprocessing.Process(target=initialize_population_single, args=(population_size, population, best_valids, best_tests, ged_to_best)))
    for process in process_list:
        process.start()
    for process in process_list:
        process.join()
    for _ in range(population_size):
        time_spent.value += 1
        times.append(time_spent.value)
    while len(population) > population_size:
        population.pop(0)

    process_list =[]
    print("multiprocessing.cpu_count(): {}".format(multiprocessing.cpu_count()))
    for cpu_id in range(multiprocessing.cpu_count()-NUM_PRESERVE_CPU):
        process_list.append(multiprocessing.Process(target=run_evolution_search_raw_crossover_single, args=(population, tournament_size, mutation_rate, time_spent, max_time_budget,
                                                                                            times, best_valids, best_tests, ged_to_best)))
    for process in process_list:
        process.start()
    for process in process_list:
        process.join()

    return list(times), list(best_valids), list(best_tests), list(ged_to_best)

def run_evolution_search_matched_crossover(max_time_budget=5e6,
                         population_size=50,
                         tournament_size=10,
                         mutation_rate=1.0):
    """Run a single roll-out of regularized evolution to a fixed time budget."""
    nasbench.reset_budget_counters()
    times, best_valids, best_tests, ged_to_best, parents_ged = [0.0], [0.0], [0.0], [10], [10]
    population = []   # (validation, spec) tuples
    status_list = [] # int for mutation, tuple for crossover

    # For the first population_size individuals, seed the population with randomly
    # generated cells.
    time_spent = 0
    for _ in range(population_size):
        spec = random_spec()
        data = nasbench.query(spec)
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
        new_spec = mutate_spec(best_spec, mutation_rate)

        data = nasbench.query(new_spec)
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
            if (not np.array_equal(best_spec_1.original_matrix, best_spec_2.original_matrix) or \
                not np.array_equal(best_spec_1.original_ops, best_spec_2.original_ops)):
                break

        new_spec, crossover_ged, num_stats = crossover_spec_match(best_spec_1, best_spec_2)
        status_list.append(num_stats)
        if type(crossover_ged)!=np.float64:
            continue

        data = nasbench.query(new_spec)
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
        parents_ged.append(crossover_ged)

        if time_spent > max_time_budget:
            break

    return times, best_valids, best_tests, ged_to_best, parents_ged, status_list

def run_evolution_search_refined_crossover_single(population, tournament_size, mutation_rate, time_spent, max_time_budget,
                                                times, best_valids, best_tests, ged_to_best, parents_ged, status_list, event_list, theory_list):
    while time_spent.value < max_time_budget:
        # mutation
        sample = random_combination(population, tournament_size)
        best_spec = sorted(sample, key=lambda i:i[0])[-1][1]
        new_spec = mutate_recipe(best_spec, mutation_rate)

        data = query(new_spec)
        time_spent.value += 1
        status_list.append(time_spent.value)
        event_list.append(0)
        times.append(time_spent.value)

        # In regularized evolution, we kill the oldest individual in the population.
        ged = ged_to_best_calc(new_spec)
        population.append((-ged, new_spec))
        population.pop(0)

        if -population[-1][0] < ged_to_best[-1]:
            best_valids.append(data['validation_accuracy'])
            best_tests.append(data['test_accuracy'])
            ged_to_best.append(ged)
        else:
            best_valids.append(best_valids[-1])
            best_tests.append(best_tests[-1])
            ged_to_best.append(ged_to_best[-1])
        parents_ged.append(parents_ged[-1])

        if time_spent.value >= max_time_budget:
            break

        #crossover
        while True:
            sample = random_combination(population, tournament_size)
            ged_to_best_1, best_spec_1= sorted(sample, key=lambda i:i[0])[-1]
            ged_to_best_1 = -ged_to_best_1
            sample = random_combination(population, tournament_size)
            ged_to_best_2, best_spec_2 = sorted(sample, key=lambda i:i[0])[-1]
            ged_to_best_2 = -ged_to_best_2
            if best_spec_1 != best_spec_2:
                break

        #new_spec, crossover_ged = crossover_spec_center(best_spec_1, best_spec_2)
        new_spec, crossover_ged, num_stats = crossover_spec_edit_path_center(best_spec_1, best_spec_2)
        status_list.append(num_stats)
        ged_parents_edge_only, ged_to_best_1_edge_only, ged_to_best_2_edge_only, num_edges_1, num_edges_2 = theory_stats(best_spec_1, best_spec_2)
        theory_list.append((time_spent.value, crossover_ged, ged_to_best_1, ged_to_best_2, ged_parents_edge_only, ged_to_best_1_edge_only, ged_to_best_2_edge_only, num_edges_1, num_edges_2))
        if type(crossover_ged)!=np.float64:
            event_list.append(1)
            continue
        else:
            event_list.append(2)

        #print("new spec after crossover: {}".format(new_spec))
        #print(new_spec.original_matrix)
        #print(new_spec.original_ops)
        #print(nasbench.is_valid(new_spec))
        data = query(new_spec)
        time_spent.value += 1
        times.append(time_spent.value)
        print("\rcrossover timestep {}".format(time_spent.value), end=" ", flush=True)

        # In regularized evolution, we kill the oldest individual in the population.
        ged = ged_to_best_calc(new_spec)
        population.append((-ged, new_spec))
        population.pop(0)

        if -population[-1][0] < ged_to_best[-1]:
            best_valids.append(data['validation_accuracy'])
            best_tests.append(data['test_accuracy'])
            ged_to_best.append(ged)
        else:
            best_valids.append(best_valids[-1])
            best_tests.append(best_tests[-1])
            ged_to_best.append(ged_to_best[-1])
        parents_ged.append(crossover_ged)

def run_evolution_search_refined_crossover_mp(max_time_budget=5e6,
                         population_size=50,
                         tournament_size=10,
                         mutation_rate=1.0,
                         best_survival=False):
    manager = multiprocessing.Manager()
    times, best_valids, best_tests, ged_to_best, parents_ged = manager.list([0.0]), manager.list([0.0]), manager.list([0.0]), manager.list([30]), manager.list([10])
    population = manager.list([])
    status_list = manager.list([]) # int for mutation, tuple for crossover
    event_list = manager.list([]) # 0 for mutation, 1 for invalid crossover, 2 for valid crossover
    theory_list = manager.list([]) # store stats for theory verification
    time_spent = manager.Value('i', 0)
    process_list =[]
    print("multiprocessing.cpu_count(): {}".format(multiprocessing.cpu_count()))
    for cpu_id in range(multiprocessing.cpu_count()-NUM_PRESERVE_CPU):
        process_list.append(multiprocessing.Process(target=initialize_population_single, args=(population_size, population, best_valids, best_tests, ged_to_best)))
    for process in process_list:
        process.start()
    for process in process_list:
        process.join()
    for _ in range(population_size):
        time_spent.value += 1
        times.append(time_spent.value)
        parents_ged.append(10)
    while len(population) > population_size:
        population.pop(0)

    process_list =[]
    print("multiprocessing.cpu_count(): {}".format(multiprocessing.cpu_count()))
    for cpu_id in range(multiprocessing.cpu_count()-NUM_PRESERVE_CPU):
        process_list.append(multiprocessing.Process(target=run_evolution_search_refined_crossover_single, args=(population, tournament_size, mutation_rate, time_spent, max_time_budget,
                                                                                            times, best_valids, best_tests, ged_to_best, parents_ged, status_list, event_list, theory_list)))
    for process in process_list:
        process.start()
    for process in process_list:
        process.join()

    return list(times), list(best_valids), list(best_tests), list(ged_to_best), list(parents_ged), list(status_list), list(event_list), list(theory_list)

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
max_time_budget = int(1000)
benchmark_name = "nasnlp_GEDfitness_eventcheck_GRU_mp_theory"
for repeat in range(repeat_num):
    print('Running repeat %d' % (repeat + 1))
    times, best_valid, best_test, ged_to_best = run_random_search_mp(max_time_budget=max_time_budget)
    random_data.append((times, best_valid, best_test, ged_to_best))

    times, best_valid, best_test, ged_to_best = run_evolution_search_mp(max_time_budget=max_time_budget, population_size=population_size, tournament_size=tournament_size)
    evolution_mutation_data.append((times, best_valid, best_test, ged_to_best))
    print("crossover_raw start")
    times, best_valid, best_test, ged_to_best = run_evolution_search_raw_crossover_mp(max_time_budget=max_time_budget, population_size=population_size, tournament_size=tournament_size)
    evolution_crossover_raw_data.append((times, best_valid, best_test, ged_to_best))
    '''
    print("crossover_match start")
    times, best_valid, best_test, ged_to_best, parents_ged, status_list = run_evolution_search_matched_crossover(max_time_budget=max_time_budget, population_size=population_size, tournament_size=tournament_size)
    evolution_crossover_match_data.append((times, best_valid, best_test, ged_to_best, parents_ged, status_list))
    '''
    print("crossover_refined start")
    times, best_valid, best_test, ged_to_best, parents_ged, status_list, event_list, theory_list = run_evolution_search_refined_crossover_mp(max_time_budget=max_time_budget, population_size=population_size, tournament_size=tournament_size)
    evolution_crossover_refined_data.append((times, best_valid, best_test, ged_to_best, parents_ged, status_list, event_list, theory_list))

result_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Results','{}_edit_path_center_crossover_NoIsomorphism_stats_pop{}_tournament{}_run{}_evaluation{}.pkl'.format(
                                                benchmark_name, population_size, tournament_size, repeat_num, max_time_budget))
with open(result_file_name, 'wb') as result_file:
    #pickle.dump((random_data, evolution_mutation_data, evolution_crossover_raw_data, evolution_crossover_match_data, evolution_crossover_refined_data), result_file)
    pickle.dump((random_data, evolution_mutation_data, evolution_crossover_raw_data, evolution_crossover_refined_data), result_file)

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
#plot_data(evolution_crossover_match_data, 'y', 'evolution_crossover_match', max_budget=max_time_budget)
plot_data(evolution_crossover_refined_data, 'c', 'evolution_crossover_refined', max_budget=max_time_budget)
plt.legend(loc='lower right')
plt.ylim(0.92, 0.96)
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
#plot_data_val(evolution_crossover_match_data, 'y', 'evolution_crossover_match', max_budget=max_time_budget)
plot_data_val(evolution_crossover_refined_data, 'c', 'evolution_crossover_refined', max_budget=max_time_budget)
plt.legend(loc='lower right')
plt.ylim(0.92, 0.97)
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
#plot_data_ged_to_best(evolution_crossover_match_data, 'y', 'evolution_crossover_match', max_budget=max_time_budget)
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
#plot_data_parents_ged(evolution_crossover_match_data, 'y', 'evolution_crossover_match', max_budget=max_time_budget)
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
#plot_data_parents_ged_moving_average(moving_average_data, 'y', 'evolution_crossover_match', max_budget=max_time_budget)
plot_data_parents_ged_moving_average(evolution_crossover_refined_data, 'c', 'evolution_crossover_refined', max_budget=max_time_budget)
plt.legend(loc='lower right')
plt.xlabel('total number of candidates evaluated')
plt.ylabel('parents ged (moving average)')
plt.grid()
plt.show()
plot_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Plots','ged_parents_movingaverage_center_crossover_pop{}_tournament{}_run{}_evaluation{}.pdf'.format(
                                                population_size, tournament_size, repeat_num, max_time_budget))
f.savefig(plot_file_name, bbox_inches='tight')
