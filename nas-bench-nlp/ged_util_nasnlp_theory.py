import networkx as nx
from networkx.algorithms.similarity import graph_edit_distance, optimize_graph_edit_distance
import random
import numpy as np
from conversions_util import convert_recipe_to_compact, make_compact_mutable, convert_compact_to_recipe
import copy

HIDDEN_TUPLE_SIZE = 2
INTERMEDIATE_VERTICES = 7
MAIN_OPERATIONS = ['linear', 'blend', 'elementwise_prod', 'elementwise_sum']
MAIN_WEIGHTS = [3., 1., 1., 1.]
MAIN_PROBABILITIES = np.array(MAIN_WEIGHTS) / np.sum(MAIN_WEIGHTS)
LINEAR_CONNECTIONS = [2, 3]
LINEAR_CONNECTION_WEIGHTS = [4, 1]
LINEAR_CONNECTION_PROBABILITIES = np.array(LINEAR_CONNECTION_WEIGHTS) / np.sum(LINEAR_CONNECTION_WEIGHTS)
ACTIVATIONS = ['activation_tanh', 'activation_sigm', 'activation_leaky_relu']
ACTIVATION_WEIGHTS = [1., 1., 1.]
ACTIVATION_PROBABILITIES = np.array(ACTIVATION_WEIGHTS) / np.sum(ACTIVATION_WEIGHTS)
INPUT_NODES = ['x', 'h_prev_0', 'h_prev_1', 'h_prev_2']

def nasnlp_to_digraph(recepie):
    dg = nx.DiGraph()
    for key in recepie.keys():
        if not dg.has_node(key):
            dg.add_node(key, op=recepie[key]['op'])

    for key in recepie.keys():
        for input_node in recepie[key]['input']:
            if not dg.has_node(input_node):
                dg.add_node(input_node, op=input_node)
            dg.add_edge(input_node, key)
    return dg

def digraph_to_nasnlp(dg):
    try:
        sorted_nodes = list(nx.topological_sort(dg))
        recepie = {}
        for i in range(len(sorted_nodes)):
            if dg.nodes[sorted_nodes[i]]['op'] not in INPUT_NODES:
                recepie[sorted_nodes[i]] = {'op': dg.nodes[sorted_nodes[i]]['op'], 'input': []}
        for edge in dg.edges:
            recepie[edge[1]]['input'].append(edge[0])
        return recepie
    except nx.exception.NetworkXUnfeasible:
        #print("error")
        return None

def node_match(node1, node2):
    return node1['op'] == node2['op']

def node_subst_cost(node1, node2):
    if node1['op'] == node2['op']:
        return 0
    elif (node1['op'] == 'x' or node2['op'] == 'x') or ("h_prev" in node1['op'] or "h_prev" in node2['op']) or ("h_new" in node1['op'] or "h_new" in node2['op']):
        return 100
    else:
        return 1

def ged_nasnlp(dg1, dg2):
    #return graph_edit_distance(dg1, dg2, node_match=node_match)
    return graph_edit_distance(dg1, dg2, node_subst_cost=node_subst_cost, timeout=300)

def ged_nasnlp_optimize(dg1, dg2):
    for ged in optimize_graph_edit_distance(dg1, dg2, node_subst_cost=node_subst_cost):
        return ged


def ged_nasnlp_edge_only(dg1, dg2):
    return graph_edit_distance(dg1, dg2, timeout=300)

def crossover_edit_path_center(dg1, dg2, ged, edit_path):
    np.random.seed(random.randint(1, 333333))
    sample_indices = random.sample(range(int(ged)), int(np.ceil(ged/2.0)))
    crossover_num = 0
    mapping_dict_1 = {}
    mapping_dict_2 = {}
    dg = dg1.copy()
    edit_record = []
    for i in range(len(edit_path[0])):
        if edit_path[0][i][0]:
            mapping_dict_1[edit_path[0][i][0]] = edit_path[0][i][1]
        if edit_path[0][i][1]:
            mapping_dict_2[edit_path[0][i][1]] = edit_path[0][i][0]
    for node_edit in edit_path[0]:
        if not node_edit[0]:
            if crossover_num in sample_indices:
                mapping_dict_2[node_edit[1]] = "add_{}".format(node_edit[1])
                dg.add_node("add_{}".format(node_edit[1]), op=dg2.nodes[node_edit[1]]['op'])
                edit_record.append((crossover_num, "add_node {}".format(node_edit)))
            crossover_num += 1
    for edge_edit in edit_path[1]:
        if type(edge_edit[0])!=tuple:
            if crossover_num in sample_indices and mapping_dict_2[edge_edit[1][0]] and mapping_dict_2[edge_edit[1][1]]:
                dg.add_edge(mapping_dict_2[edge_edit[1][0]], mapping_dict_2[edge_edit[1][1]])
                edit_record.append((crossover_num, "add_edge {}".format(edge_edit)))
            crossover_num += 1
        elif type(edge_edit[1])!=tuple:
            if crossover_num in sample_indices:
                dg.remove_edge(edge_edit[0][0], edge_edit[0][1])
                edit_record.append((crossover_num, "remove_edge {}".format(edge_edit)))
            crossover_num += 1
    for node in dg1.nodes:
        if mapping_dict_1[node] and dg1.nodes[node]['op'] != dg2.nodes[mapping_dict_1[node]]['op']:
            if crossover_num in sample_indices:
                dg.nodes[node]['op'] = dg2.nodes[mapping_dict_1[node]]['op']
                edit_record.append((crossover_num, "op_change {}".format(node)))
            crossover_num += 1
    for node_edit in edit_path[0]:
        if not node_edit[1]:
            if crossover_num in sample_indices:
                dg.remove_node(node_edit[0])
                edit_record.append((crossover_num, "remove_node {}".format(node_edit)))
            crossover_num += 1
    #print("parents_ged: {}, ged to parent1: {}, ged to parent2: {}".format(ged, ged_nasnlp(dg, dg1)[0], ged_nasnlp(dg, dg2)[0]))
    #print("parents_edit_path: {}, edit_path to parent1: {}, edit_path to parent2: {}".format(edit_path, ged_nasnlp(dg, dg1)[1], ged_nasnlp(dg, dg2)[1]))
    #print("edit_record: {}".format(edit_record))
    #print("sample_indices: {}".format(sample_indices))
    #print("offspring: {}, parent1: {}, parent2: {}".format(dg, dg1, dg2))
    return dg

def nasnlp_valid(dg):
    prev_hidden_nodes = [f'h_prev_{i}' for i in range(HIDDEN_TUPLE_SIZE)]
    if not dg.has_node('x'):
        return False
    hidden_valid = False
    for i in range(HIDDEN_TUPLE_SIZE):
        if dg.has_node(f'h_prev_{i}'):
            if not dg.has_node(f'h_new_{i}'):
                return False
            if nx.has_path(dg,f'h_prev_{i}',f'h_new_{i}') and nx.has_path(dg,'x',f'h_new_{i}'):
                hidden_valid = True
        elif dg.has_node(f'h_new_{i}'):
            return False
        '''
        for prev_hidden_node in prev_hidden_nodes:
            if dg.has_edge(prev_hidden_node, f'h_new_{i}') or dg.has_edge(f'h_new_{i}', prev_hidden_node):
                return False
        '''
    if not hidden_valid:
        return False
    for node in dg.nodes():
        if dg.in_degree(node) > 2:
            return False
    return True

def generate_redundant_graph(recipe, base_nodes):
    """
    This code is from NAS-Bench-NLP https://arxiv.org/abs/2006.07116
    """
    i = 0
    activation_nodes = []
    while i < HIDDEN_TUPLE_SIZE + INTERMEDIATE_VERTICES:
        op = np.random.choice(MAIN_OPERATIONS, 1, p=MAIN_PROBABILITIES)[0]
        if op == 'linear':
            num_connections = np.random.choice(LINEAR_CONNECTIONS, 1,
                                               p=LINEAR_CONNECTION_PROBABILITIES)[0]
            connection_candidates = base_nodes + activation_nodes
            if num_connections > len(connection_candidates):
                num_connections = len(connection_candidates)

            connections = np.random.choice(connection_candidates, num_connections, replace=False)
            recipe[f'node_{i}'] = {'op':op, 'input':connections}
            i += 1

            # after linear force add activation node tied to the new node, if possible (nodes budget)
            op = np.random.choice(ACTIVATIONS, 1, p=ACTIVATION_PROBABILITIES)[0]
            recipe[f'node_{i}'] = {'op':op, 'input':[f'node_{i - 1}']}
            activation_nodes.append(f'node_{i}')
            i += 1

        elif op in ['blend', 'elementwise_prod', 'elementwise_sum']:
            # inputs must exclude x
            if op == 'blend':
                num_connections = 3
            else:
                num_connections = 2
            connection_candidates = list(set(base_nodes) - set('x')) + list(recipe.keys())
            if num_connections <= len(connection_candidates):
                connections = np.random.choice(connection_candidates, num_connections, replace=False)
                recipe[f'node_{i}'] = {'op':op, 'input':connections}
                i += 1

def create_hidden_nodes(recipe):
    """
    This code is from NAS-Bench-NLP https://arxiv.org/abs/2006.07116
    """
    new_hiddens_map = {}
    for k in np.random.choice(list(recipe.keys()), HIDDEN_TUPLE_SIZE, replace=False):
        new_hiddens_map[k] = f'h_new_{len(new_hiddens_map)}'

    for k in new_hiddens_map:
        recipe[new_hiddens_map[k]] = recipe[k]
        del recipe[k]

    for k in recipe:
        recipe[k]['input'] = [new_hiddens_map.get(x, x) for x in recipe[k]['input']]

def remove_redundant_nodes(recipe):
    """
    This code is from NAS-Bench-NLP https://arxiv.org/abs/2006.07116
    """
    q = [f'h_new_{i}' for i in range(HIDDEN_TUPLE_SIZE)]
    visited = set(q)
    while len(q) > 0:
        if q[0] in recipe:
            for node in recipe[q[0]]['input']:
                if node not in visited:
                    q.append(node)
                    visited.add(node)
        q = q[1:]

    for k in list(recipe.keys()):
        if k not in visited:
            del recipe[k]

    return visited

def sample_random_architecture():
    np.random.seed(random.randint(1, 333333))
    while True:
        prev_hidden_nodes = [f'h_prev_{i}' for i in range(HIDDEN_TUPLE_SIZE)]
        base_nodes = ['x'] + prev_hidden_nodes

        recipe = {}
        generate_redundant_graph(recipe, base_nodes)
        #print("redundant recipe: {}".format(recipe))
        create_hidden_nodes(recipe)
        #print("hidden recipe: {}".format(recipe))
        visited_nodes = remove_redundant_nodes(recipe)
        valid_recipe = True

        # check that all input nodes are in the graph
        for node in base_nodes:
            if node not in visited_nodes:
                valid_recipe = False
                break

        # constraint: prev hidden nodes are not connected directly to new hidden nodes
        for i in range(HIDDEN_TUPLE_SIZE):
            if len(set(recipe[f'h_new_{i}']['input']) & set(prev_hidden_nodes)) > 0:
                valid_recipe = False
                break

        if len(recipe.keys()) > HIDDEN_TUPLE_SIZE + INTERMEDIATE_VERTICES:
            valid_recipe = False

        if valid_recipe:
            return recipe

def mutate_recipe(parent, mutation_rate=1):
    """
    This will mutate the cell in one of two ways:
    change an edge; change an op.
    Todo: mutate by adding/removing nodes.
    Todo: mutate the list of hidden nodes.
    Todo: edges between initial hidden nodes are not mutated.
    """
    np.random.seed(random.randint(1, 333333))
    parent_compact = convert_recipe_to_compact(parent)
    parent_compact = make_compact_mutable(parent_compact)
    compact = copy.deepcopy(parent_compact)

    edges, ops, hiddens = compact
    max_node_idx = max([max(edge) for edge in edges])

    for _ in range(int(mutation_rate)):
        mutation_type = np.random.choice(2) + 1

        if mutation_type == 0:
            # change a hidden node. Note: currently not being used
            hiddens.pop(np.random.choice(len(hiddens)))
            choices = [i for i in range(4, max_node_idx) if i not in hiddens]
            hiddens.append(np.random.choice(choices))
            hiddens.sort()

        elif mutation_type == 1:
            # change an edge
            # Currently cannot change an edge to/from an h_prev node
            edge_choices = [i for i in range(len(edges)) if edges[i][0] >= 4]
            if len(edge_choices) > 0:
                i = np.random.choice(edge_choices)
                node_choices = [j for j in range(4, edges[i][1])]
                if len(node_choices) > 0:
                    edges[i][0] = np.random.choice(node_choices)

        else:
            # change an op. Note: the first 4 nodes don't have ops
            idx_choices = [i for i in range(len(ops)) if ops[i] not in [0, 6, 7]]
            if len(idx_choices) > 0:
                idx = np.random.choice(idx_choices)
                num_inputs = len([edge for edge in edges if edge[1] == idx])

                # each operation can have 1, 2, [2,3], or 3 inputs only
                groups = [[0], [1, 2, 3], [4, 5]]
                group = groups[num_inputs]
                choices = [i for i in group if i != ops[idx]]
                ops[idx] = np.random.choice(choices)

    compact = (edges, ops, hiddens)
    new_recipe = convert_compact_to_recipe(compact)
    return new_recipe

def arch_to_compact(arch, max_nodes=12):
    edges = []
    hiddens = list(arch[len(arch)-3:])
    idx_zero_end = len(arch) - 4
    count = 0
    while arch[idx_zero_end] != 0:
        count += 1
        idx_zero_end -= 1
    if count >= 2:
        ops = list(arch[(max_nodes+1)*(max_nodes+1):len(arch) - 3])
    else:
        idx_zero_start = idx_zero_end
        while arch[idx_zero_start] == 0:
            idx_zero_start -= 1
        ops = list(arch[(max_nodes+1)*(max_nodes+1):idx_zero_start+1])
        ops.append(arch[len(arch) - 4])
    adj_matrix = np.zeros((max_nodes+1, max_nodes+1))
    for i in range(len(adj_matrix)):
        for j in range(len(adj_matrix[0])):
            adj_matrix[i][j] = arch[i*(max_nodes+1)+j]
            if adj_matrix[i][j] == 1:
                if j == max_nodes:
                    col = len(ops)-1
                else:
                    col = j
                if i == max_nodes:
                    row = len(ops)-1
                else:
                    row = i
                if col >= len(ops)-1 or row >= len(ops)-1:
                    continue
                edges.append((row, col))
    return (tuple(edges), tuple(ops), tuple(hiddens))

