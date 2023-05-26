import networkx as nx
from networkx.algorithms.similarity import graph_edit_distance, optimize_graph_edit_distance
import random
import numpy as np

def valid_nas301(spec, inter_node_num=4):
    for i in range(inter_node_num):
        if spec[i*2][1] == spec[i*2+1][1] or spec[i*2][1]>=i+2 or spec[i*2+1][1]>=i+2:
            return False
    return True

def nas301_to_digraph(operations, node_num=7, inter_node_num=4):
    dg = nx.DiGraph()
    for i in range(node_num):
        dg.add_node(i)

    for i in range(inter_node_num):
        for j in range(2):
            dg.add_edge(operations[i*2+j][1], i+2, op=operations[i*2+j][0])
        dg.add_edge(i+2, node_num-1, op=2)

    return dg

def argsort(seq):
    return sorted(range(len(seq)), key=seq.__getitem__)

def digraph_to_nas301(dg):
    try:
        sorted_nodes = list(nx.topological_sort(dg))
        operations = []
        for i in range(2, len(sorted_nodes)-1):
            edge_list = dg.in_edges(i, data=True)
            if len(edge_list) != 2:
                return None
            for edge in edge_list:
                operations.append((edge[2]['op'], edge[0]))
        operations_sorted = []
        node_max_index_list = []
        for i in range(4):
            node_max_index_list.append(max(operations[i*2][1],operations[i*2+1][1]))
        arg_list = argsort(node_max_index_list)
        for arg in arg_list:
            operations_sorted.append(operations[arg*2])
            operations_sorted.append(operations[arg*2+1])
        return operations_sorted
    except nx.exception.NetworkXUnfeasible:
        #print("error")
        return None

def digraph_to_nas301_general(dg):
    try:
        sorted_nodes = list(nx.topological_sort(dg))
        operations = []
        for i in range(2, len(sorted_nodes)-1):
            edge_list = dg.in_edges(i, data=True)
            if len(edge_list) != 2:
                return None
            for edge in edge_list:
                operations.append((edge[2]['op'], edge[0]))
        return operations
    except nx.exception.NetworkXUnfeasible:
        #print("error")
        return None

def edge_match(edge1, edge2):
    return edge1['op'] == edge2['op']

def edge_subst_cost(edge1, edge2):
    if edge1['op'] == edge2['op']:
        return 0
    else:
        return 1

def ged_nas301(dg1, dg2):
    #return graph_edit_distance(dg1, dg2, node_match=node_match)
    return graph_edit_distance(dg1, dg2, edge_subst_cost=edge_subst_cost)

def ged_nas301_optimize(dg1, dg2):
    for ged in optimize_graph_edit_distance(dg1, dg2, edge_subst_cost=edge_subst_cost):
        return ged

def crossover_edit_path_center_nas301(dg1, dg2):
    ged, edit_path = ged_nas301(dg1, dg2)
    sample_indices = random.sample(range(int(ged)), int(np.ceil(ged/2.0)))
    crossover_num = 0
    mapping_dict_1 = {}
    mapping_dict_2 = {}
    dg = dg1.copy()
    for i in range(len(dg1.nodes)):
        mapping_dict_1[edit_path[0][i][0]] = edit_path[0][i][1]
        mapping_dict_2[edit_path[0][i][1]] = edit_path[0][i][0]
    for edge_edit in edit_path[1]:
        if type(edge_edit[0])!=tuple:
            if crossover_num in sample_indices:
                dg.add_edge(mapping_dict_2[edge_edit[1][0]], mapping_dict_2[edge_edit[1][1]], op=dg2[edge_edit[1][0]][edge_edit[1][1]]['op'])
            crossover_num += 1
        elif type(edge_edit[1])!=tuple:
            if crossover_num in sample_indices:
                dg.remove_edge(edge_edit[0][0], edge_edit[0][1])
            crossover_num += 1
        elif dg1[edge_edit[0][0]][edge_edit[0][1]]['op']!=dg2[edge_edit[1][0]][edge_edit[1][1]]['op']:
            if crossover_num in sample_indices:
                dg[edge_edit[0][0]][edge_edit[0][1]]['op']=dg2[edge_edit[1][0]][edge_edit[1][1]]['op']
            crossover_num += 1
    return dg, ged, edit_path
