import networkx as nx
from networkx.algorithms.similarity import graph_edit_distance, optimize_graph_edit_distance
import random
import numpy as np

def nas101_to_digraph(adjacency, operations):
    dg = nx.DiGraph()
    for i in range(len(operations)):
        dg.add_node(i, op=operations[i])
    for i in range(len(adjacency)-1):
        for j in range(i+1, len(adjacency)):
            if adjacency[i][j] == 1:
                dg.add_edge(i,j)
    return dg

def digraph_to_nas101(dg):
    try:
        sorted_nodes = list(nx.topological_sort(dg))
        nodes_dict = {}
        operations = []
        for i in range(len(sorted_nodes)):
            nodes_dict[sorted_nodes[i]] = i
            operations.append(dg.nodes[sorted_nodes[i]]['op'])
        adjacency = np.zeros((len(sorted_nodes), len(sorted_nodes))).astype(int)
        for edge in dg.edges:
            adjacency[nodes_dict[edge[0]]][nodes_dict[edge[1]]] = 1
        return (adjacency, operations)
    except nx.exception.NetworkXUnfeasible:
        #print("error")
        return None

def node_match(node1, node2):
    return node1['op'] == node2['op']

def node_subst_cost(node1, node2):
    if node1['op'] == node2['op']:
        return 0
    elif (node1['op'] == 'input' or node2['op'] == 'input') or (node1['op'] == 'output' or node2['op'] == 'output'):
        return 100
    else:
        return 1

def ged_nas101(dg1, dg2):
    #return graph_edit_distance(dg1, dg2, node_match=node_match)
    return graph_edit_distance(dg1, dg2, node_subst_cost=node_subst_cost)

def ged_nas101_edge_only(dg1, dg2):
    #return graph_edit_distance(dg1, dg2, node_match=node_match)
    return graph_edit_distance(dg1, dg2)

def ged_nas101_optimize(dg1, dg2):
    for ged in optimize_graph_edit_distance(dg1, dg2, node_subst_cost=node_subst_cost):
        return ged

def crossover_edit_path_center(dg1, dg2, CR=0.5):
    ged, edit_path = ged_nas101(dg1, dg2)
    sample_indices = random.sample(range(int(ged)), int(np.ceil(ged*CR)))
    crossover_num = 0
    mapping_dict_1 = {}
    mapping_dict_2 = {}
    dg = dg1.copy()
    for i in range(len(dg1.nodes)):
        mapping_dict_1[edit_path[0][i][0]] = edit_path[0][i][1]
        mapping_dict_2[edit_path[0][i][1]] = edit_path[0][i][0]
    for node_edit in edit_path[0]:
        if type(node_edit[0])!=int:
            if crossover_num in sample_indices:
                mapping_dict_2[node_edit[1]] = len(dg.nodes)
                dg.add_node(len(dg.nodes), op=dg2.nodes[node_edit[1]]['op'])
            crossover_num += 1
    for edge_edit in edit_path[1]:
        if type(edge_edit[0])!=tuple:
            if crossover_num in sample_indices:
                dg.add_edge(mapping_dict_2[edge_edit[1][0]], mapping_dict_2[edge_edit[1][1]])
            crossover_num += 1
        elif type(edge_edit[1])!=tuple:
            if crossover_num in sample_indices:
                dg.remove_edge(edge_edit[0][0], edge_edit[0][1])
            crossover_num += 1
    for ind in range(len(dg1.nodes)):
        if type(mapping_dict_1[ind])==int and dg1.nodes[ind]['op'] != dg2.nodes[mapping_dict_1[ind]]['op']:
            if crossover_num in sample_indices:
                dg.nodes[ind]['op'] = dg2.nodes[mapping_dict_1[ind]]['op']
            crossover_num += 1
    for node_edit in edit_path[0]:
        if type(node_edit[1])!=int:
            if crossover_num in sample_indices:
                dg.remove_node(node_edit[0])
            crossover_num += 1
    return dg, ged, edit_path



