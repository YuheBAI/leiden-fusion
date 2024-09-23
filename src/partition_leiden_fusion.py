import networkx as nx
import argparse
import pickle
from ogb.nodeproppred import NodePropPredDataset, PygNodePropPredDataset
import pandas as pd
import copy
import igraph as ig
import leidenalg as la
import numpy as np

from collections import defaultdict, Counter
from partition_utils import *
import os
import datetime
import time


def findCommunityNeighborAndEdgeConnection(data, community_assignment):
    neighbors = defaultdict(set)
    edge_cuts =  defaultdict(lambda: defaultdict(int))
    edge_index = data['edge_index']
    for i in range(edge_index.shape[1]):
        u, v = edge_index[:, i]
        community_u = community_assignment[u]
        community_v = community_assignment[v]
        if community_u != community_v:
            neighbors[community_u].add(community_v)
            neighbors[community_v].add(community_u)
            edge_cuts[community_u][community_v]+=1
            edge_cuts[community_v][community_u]+=1
    return neighbors, edge_cuts 


def updateNeighborAndEdge(neighbors,edge_cuts,community_u, community_v):
    neighbors[community_u]=neighbors[community_u].union(neighbors[community_v])
    neighbors[community_u].discard(community_v)
    neighbors[community_u].discard(community_u)
    
    del neighbors[community_v]
    #update edge_cuts
    for neighbor_u in neighbors[community_u]:
        edge_cuts[community_u][neighbor_u]+=edge_cuts[community_v][neighbor_u]
        edge_cuts[neighbor_u][community_u] = edge_cuts[community_u][neighbor_u]
    edge_cuts[community_u][community_v] = 0
    edge_cuts[community_v][community_u] = 0
    del edge_cuts[community_u][community_v]
    del edge_cuts[community_v]

    for key in neighbors:
        if community_v in neighbors[key] and key != community_u:
            neighbors[key].discard(community_v)
            neighbors[key].add(community_u)
            edge_cuts[neighbor_u][community_v] =0
            del edge_cuts[neighbor_u][community_v]

def getAdapterNeighbors(neighbors, community_sizes, u, limite_size):
    adapter_neighbors = []
    max_size =  limite_size - community_sizes[u]
    for community in neighbors[u]:
        if community_sizes[community] < max_size:
            adapter_neighbors.append(community)
    return adapter_neighbors


def getLargestEdgeCutsNeighbors(edge_cuts, neighbors,  community_sizes, u, limite_size):
    adapter_neighbors = getAdapterNeighbors(neighbors, community_sizes, u, limite_size)
    max_edge_cut = -1
    select_community = -1
    smallest_size = -1
    if len(adapter_neighbors)>0:
        for community in adapter_neighbors:
            if edge_cuts[u][community] > max_edge_cut:
                max_edge_cut = edge_cuts[u][community]
                select_community = community
    else:
        for community in neighbors[u]:
            if smallest_size == -1 or community_sizes[community] < smallest_size:
                smallest_size = community_sizes[community]
                select_community = community

    return community_sizes[select_community], select_community

def community_selection(community_sizes,neighbors, min_size):
    for community in neighbors:
        if len(neighbors[community])  ==1 and community_sizes[community]< min_size:
            return  community
    return min(community_sizes, key=community_sizes.get)

def fusionner_edge_cut(node_partitions, G, limite_size_coef =1.05,num_parts=16):
    print("Merge communities...")
    community_assignment = {node: community for node, community in enumerate(node_partitions)}
    community_sizes = Counter(community_assignment.values())
    size_original = len(community_sizes)

    limite_size  = limite_size_coef*len(community_assignment)/num_parts

    min_size = (2-limite_size_coef)*sum(community_sizes.values())/num_parts
    neighbors,edge_cuts = findCommunityNeighborAndEdgeConnection(G, community_assignment) # this takes long time


    dictFusion = defaultdict(set)
    count = 0
    community_diminue=[]

    while len(neighbors) > num_parts:
        count += 1
        community_u = community_selection(community_sizes,neighbors, min_size)
        size_neighbor, community_v = getLargestEdgeCutsNeighbors(edge_cuts, neighbors, community_sizes, community_u, limite_size)

        if community_u in dictFusion:
            updateNeighborAndEdge(neighbors,edge_cuts, community_u, community_v)
            dictFusion[community_u].add(community_v)
            if community_v in dictFusion:

                dictFusion[community_u]=dictFusion[community_u].union(dictFusion[community_v])

                del dictFusion[community_v]
            community_sizes[community_u] += community_sizes[community_v]
            del community_sizes[community_v]
            community_diminue.append(community_v)
        else:
            updateNeighborAndEdge(neighbors,edge_cuts, community_v, community_u)
            community_sizes[community_v] += community_sizes[community_u]
            del community_sizes[community_u]
            community_diminue.append(community_u)
            if community_v in dictFusion:
                dictFusion[community_v].add(community_u)
            else:
                dictFusion[community_v]={community_u}

    newNodePartition = copy.deepcopy(node_partitions)
    listKey = list(dictFusion.keys())
    sortedList = sorted(listKey)
    tabNonConcat =[]
    for i in range(size_original):
        if i not in community_diminue and i not in dictFusion:
            tabNonConcat.append(i)
    dictNew = {element: index for index, element in enumerate(sortedList)}

    for index,node in enumerate(node_partitions):
        isin = False
        if node not in dictFusion:
            for key in dictFusion:
                if node in dictFusion[key]:
                    newNodePartition[index] = dictNew[key]
                    isin = True
                    break
            if not isin:
                newNodePartition[index] = len(dictFusion)+tabNonConcat.index(node)
        if node in dictFusion:
            newNodePartition[index] = dictNew[node]

    return newNodePartition

def run_leiden_auto(G_ig, max_comm_size=None):
    print("Running leiden to get auto number of partitions...")
    print(datetime.datetime.now())
    start_time = time.time()
    if max_comm_size == None :
        partition = la.find_partition(G_ig, la.ModularityVertexPartition)
    else:
        partition = la.find_partition(G_ig, la.ModularityVertexPartition, max_comm_size=max_comm_size)
    partition_data = partition.membership

    print(datetime.datetime.now())
    end_time = time.time()
    training_time = end_time - start_time
    
    filename = './node_partitions/partitioning_times.txt'
    with open(filename, 'a') as file: 
        file.write(f"Partitioning time for leiden auto: {training_time}\n")

    file_path = 'node_partitions/node_partitions_leiden_auto.txt'
    with open(file_path, 'w') as file:
        for part in partition_data:
            file.write(f'{part}\n')

def convert_to_igraph(graph):
    edges = graph['edge_index'].T.tolist()
    g = ig.Graph(edges=edges, directed=False)
    return g

def partition_leiden(name, num_parts, write=True):
    print(datetime.datetime.now())
    dataset = NodePropPredDataset(name=name)
    graph, label = dataset[0]

    directory = './node_partitions'
    if not os.path.exists(directory):
        os.makedirs(directory)

    try:
        with open('node_partitions/node_partitions_leiden_auto.txt', 'r') as file:
            node_partitions = [int(line.strip()) for line in file]
    except FileNotFoundError:
        print("Create igraph...")
        G_ig = convert_to_igraph(graph)
        run_leiden_auto(G_ig)
        with open('node_partitions/node_partitions_leiden_auto.txt', 'r') as file:
            node_partitions = [int(line.strip()) for line in file]
    
    #fusion
    start_time = time.time()
    newNodePartition = fusionner_edge_cut(node_partitions, graph, num_parts=num_parts)
    print(datetime.datetime.now())
    end_time = time.time()
    training_time = end_time - start_time

    filename = './node_partitions/partitioning_times.txt'
    with open(filename, 'a') as file:
        file.write(f"Partition_leiden_{num_parts}_parts:\n")
        file.write(f"Partitioning Time: {training_time}\n")

    print_stats(newNodePartition)

    with open(f'./node_partitions/node_partitions_leiden_{num_parts}_partition.txt', 'w') as f:
        for part in newNodePartition:
            f.write(f'{part}\n')
    print("Partition file saved.")
    return newNodePartition

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='ogbn-arxiv', help="name of dataset to partition")
    parser.add_argument("-n", "--num_parts", default=2, type=int, help="number of partitions")

    args = parser.parse_args()
    partition_leiden(args.dataset, args.num_parts)
