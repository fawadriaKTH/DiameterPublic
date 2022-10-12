# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import matplotlib.pyplot as plt
import networkx as nx
import math
from timeit import default_timer as timer
from random import sample
import json
#from copy import deepcopy


# The path to the input graph, where the graph is given as a txt file (2-column)

#path = 'testGraph.txt'



G = nx.read_weighted_edgelist(path)
G.remove_edges_from(nx.selfloop_edges(G))
cc = list(nx.connected_components(G))
largest_cc = max(nx.connected_components(G), key=len)
H = G.subgraph(largest_cc)
degrees = {node:val for (node, val) in H.degree()}
highest_d = max(degrees, key=degrees.get)



def k_centers(H, k):
  # Greedy k-centers heuristic (acting on nodes)
  centers = []
  #cities = list(H.nodes())
  centers.append(sample(H.nodes(),1)[0])
  k = k-1
  city_d = nx.single_source_shortest_path_length(H, source=centers[0])
  clusters = {k:centers[0] for k in H.nodes()} # nodes assigned to closest center
  while k!= 0:
    new_center = max(city_d, key = lambda i: city_d[i])
    centers.append(new_center)
    city_d2 = nx.single_source_shortest_path_length(H, source=new_center)
    clusters2 = {k:new_center for k in H.nodes() if city_d2[k]<city_d[k]}
    clusters.update(clusters2)
    city_d = {k:min(city_d[k],city_d2[k]) for k in city_d}
    k = k-1
  return centers, clusters

def findBiggestCluster(clusters):
    # Return largest cluster (as a set of nodes) after running k_centers heuristic.
    # Assumes clusters is a dict with key=nodes, values = centers.
    
    # Invert the dictionary (non-bijective!)
    inv_map = {}
    for k, v in clusters.items():
        inv_map.setdefault(v, set()).add(k)
    
    # Counting occurences
    occ = {k: len(v) for k, v in inv_map.items()}
    
    # Key with max value in z.
    x = max(occ, key=occ.get)
    
    return inv_map[x]


def addEdges_biggesttoAll(H,k,delta):
    # The constant-factor approximation, guaranteed to work if k <= sqrt(n*delta)-1, but might work in practice for even larger k.
    # delta = max. degree increase per node.
    # We pick the nodes closest to the center of the biggest cluster, each allowing a budget of delta
    
    centers, clusters = k_centers(H, k+1) # Finding k+1 clusters, so we can add k edges
    big = findBiggestCluster(clusters) # set of nodes in biggest cluster
    Hnew = nx.Graph(H) # the augmented graph
    if len(big)*delta<k:
        #raise ValueError("k is too large for constant-factor approximation")
        return nx.empty_graph(1) # returne empty graph (indicating algo doesnt work)
    
    # endpoints of the k clusters
    endpoints = list(set(centers) - big)
    
    # The ceil(k/delta) closest points to the center of the biggest cluster
    centerBig = list(set(centers).intersection(big))[0]
    beginpoints = bfsk(H, centerBig, math.ceil(k/delta))
    
    # Every beginpoint has at most delta edges
    newEdges = [(beginpoints[math.floor(i/delta)],endpoints[i]) for i in range(k)]
    Hnew.add_edges_from(newEdges)
    return Hnew

def addEdges_Random(H, k, delta):
    # Add k random edges to graph H, s.t. each degree does not increase more than delta
    if math.floor(len(H.nodes())*delta/2)<k:
        raise ValueError("k is too large!")
    
    results = []
    Hnew = nx.Graph(H)
    deltacounter = {i: 0 for i in H.nodes()}
    while len(results)!=k:
        u = sample(H.nodes(), 1)[0]
        v = sample(H.nodes(), 1)[0]
        cond1 = u != v
        cond2 = (u,v) not in H.edges()
        cond3 = deltacounter[u]<delta and deltacounter[v]<delta
        if cond1 and cond2 and cond3:
            results.append((u,v))
            deltacounter[u] = deltacounter[u]+1
            deltacounter[v] = deltacounter[v]+1
    Hnew.add_edges_from(results)
    return Hnew

def heuristic_two_sweep(H,k,delta):
    """ Baseline heuristic.
        The idea is to pick the farthest node from a random node
        and connect to the node that is furthest away.
    """
    if math.floor(len(H.nodes())*delta/2)<k:
        raise ValueError("k is too large!")
    
    results = []
    deltacounter = {i: 0 for i in H.nodes()}
    forbidden = set()
    
    Hnew = nx.Graph(H)
    while len(results)!=k:
        # select a random source node
        source = sample(Hnew.nodes(), 1)[0]
        # get the distances to the other nodes
        dist = nx.single_source_shortest_path_length(Hnew, source)
        dist = {k:v for k,v in dist.items() if k not in forbidden}
        # take a node that is (one of) the farthest nodes from the source
        u = max(dist, key=dist.get)
        
        # take the furthest node from this node
        dist2 = nx.single_source_shortest_path_length(Hnew, u)
        dist2 = {k:v for k,v in dist2.items() if k not in forbidden}
        v = max(dist2, key=dist2.get)
        
        # add edge
        results.append((u,v))
        Hnew.add_edges_from(results)
        
        # update forbidden
        deltacounter[u] = deltacounter[u]+1
        deltacounter[v] = deltacounter[v]+1
        
        if deltacounter[u]==delta:
            forbidden.add(u)
        if deltacounter[v]==delta:
            forbidden.add(v)
        
    return Hnew

def logkapprox(H,k,delta):
    # This is a faster implementation for the core-tree building algorithm.
    
    # The returned augmented graph
    Hnew = nx.Graph(H)
    results = []
    
    # We first pick the centers of the cores, then add two random neighbors if possible.
    centers, _ = k_centers(H, k+1)
    centers = set(centers)
    
    # If degree-one vertices are chosen as centers, we take instead take their neighbor as a center.
    newcs = set()
    oldcs = set()
    for c in centers:
        if H.degree(c)==1:
            oldcs.add(c)
            newcs.update(set(H[c]))
    centers.update(newcs)
    centers = centers - oldcs
    
    # We associate disjoint 3-cores with the centers (if possible)       
    cores = {}
    forbidden = set()        
    for c in centers:
        cands = set(H[c])-forbidden
        if len(cands) >= 2 and c not in forbidden:
            two_ngbs = sample(cands,2)
            forbidden.update(two_ngbs)
            forbidden.add(c)
            cores[c] = two_ngbs
        
    
    # Connecting the cores into tree-like structure.
    centers = [c for c in cores]
    root = centers[0]
    
    i1 = 1
    i2 = min(3*delta,len(centers)-1)
    stack = centers[i1:i2+1]
    
    beginpoints = [root,cores[root][0],cores[root][1]]
    newEdges2 = [(beginpoints[math.floor(i/delta)],stack[i]) for i in range(len(stack))]
    results.extend(newEdges2)
    
    teller=1
    while i2<len(centers)-1:
        i1 = i2+1
        i2 = min(i2+3*delta-1,len(centers)-1) #-1 because the center is already shortcut once.
        current_c = centers[teller]
        stack = centers[i1:i2+1]
        
        beginpoints = [cores[current_c][0],cores[current_c][1],current_c]
        newEdges2 = [(beginpoints[math.floor(i/delta)],stack[i]) for i in range(len(stack))]
        results.extend(newEdges2)
        teller = teller + 1
    
    Hnew.add_edges_from(results)
    return Hnew




def _two_sweep_undirected(H):
    """Helper function for finding a lower bound on the diameter
        for undirected Graphs.

        The idea is to pick the farthest node from a random node
        and return its eccentricity.

        paper: 'Fast Computation of Empirically Tight Bounds for the Diameter of Massive Graphs'
    """
    
    repeats = 1
    eccs = [0]*repeats 
    for i in range(repeats):
        # select a random source node
        source = sample(H.nodes(), 1)[0]
        # get the distances to the other nodes
        distances = nx.single_source_shortest_path_length(H, source)
        # if some nodes have not been visited, then the graph is not connected
        if len(distances) != len(H):
            raise nx.NetworkXError("Graph not connected.")
        # take a node that is (one of) the farthest nodes from the source
        node = max(distances, key=distances.get)
        # return the eccentricity of the node
        eccs[i] = nx.eccentricity(H, node)
    
    return max(eccs)


def bfsk(graph, node, k): #BFS-like function that finds k nearest ngbhs.
    visited = [] # List for visited nodes.
    queue = []     #Initialize a queue
    visited.append(node)
    queue.append(node)
    while len(visited)<k:          # Creating loop to visit each node
        m = queue.pop(0) 
        for neighbour in graph[m]:
            if neighbour not in visited:
                visited.append(neighbour)
                queue.append(neighbour)
    visited = visited[:k]
    return visited

def maximal3cores(H):
    cores = {}
    Hnew = nx.Graph(H)
    valid = {v for v in Hnew.nodes() if Hnew.degree(v)>1}
    while valid:
        next_centercore = sample(valid,1)[0] #sampling center vertex of a 3-core.
        two_ngbs = sample(list(Hnew[next_centercore]),2) #sampling two neighbors.
        cores[next_centercore] = two_ngbs
        Hnew.remove_node(next_centercore)
        Hnew.remove_nodes_from(two_ngbs)
        valid = {v for v in Hnew.nodes() if Hnew.degree(v)>1}
        
    return cores     

#%%

def experiment(H):
    writepath = 'test2.txt'
    writepath2 = 'testTime2.txt'
    
    #karray = range(10,210,10)
    karray = [4*2**j for j in range(1,9)] #8,16,...,1024
    delta = 25
    t = len(karray)
    
    # Performance vectors
    r1 = [10000] * t
    r2 = [10000] * t
    r3 = [10000] * t
    r4 = [10000] * t
    
    # Timing vectors
    t1 = [0] * t
    t2 = [0] * t
    t3 = [0] * t
    t4 = [0] * t
    
    repeats = 4
    for i in range(t):
        # constant-factor approximation
        start = timer()
        for j in range(repeats):
            cand = _two_sweep_undirected(addEdges_biggesttoAll(H,karray[i],delta))
            r1[i] = min(cand,r1[i])
        end = timer()
        t1[i] = round((end-start)*100)/100
        print('constantfactor: '+str(t1[i]))
        t1[i] = t1[i]/repeats
        
        # random-edges (multiple repeats, taking best value)
        start = timer()
        for j in range(repeats):           
             cand = _two_sweep_undirected(addEdges_Random(H,karray[i],delta))
             r2[i] = min(cand,r2[i])
        end = timer()
        t2[i] = round((end-start)*100)/100
        print('randomEdges: '+str(t2[i]))
        t2[i] = t2[i]/repeats
        
        # greedy random heuristic
        start = timer()
        for j in range(repeats):  
            cand = _two_sweep_undirected(heuristic_two_sweep(H,karray[i],delta))
            r3[i] = min(cand,r3[i])
        end = timer()
        t3[i] = round((end-start)*100)/100
        print('Greedy: '+str(t3[i]))
        t3[i] = t3[i]/repeats
        
        
        # logk approximation
        start = timer()
        for j in range(repeats):  
            cand = _two_sweep_undirected(logkapprox(H,karray[i],delta))
            r4[i] = min(cand,r4[i])
        end = timer()
        t4[i] = round((end-start)*100)/100
        print('log(k): '+str(t4[i]))
        print('-----------------')
        t4[i] = t4[i]/repeats
    
    #plt.plot(karray, r1, 'b*',karray,r2, 'ro',karray, r3,'k+',karray, r4,'gX',linestyle='dashed')
    c = [karray,r1,r2,r3,r4]
    diam = _two_sweep_undirected(H)
    with open(writepath, 'w') as file:
        file.write(str(0)+"\t"+str(diam)+"\t"+str(diam)+"\t"+str(diam)+"\t"+str(diam)+'\n')
        for x in zip(*c):
            file.write("{0}\t{1}\t{2}\t{3}\t{4}\n".format(*x))
    c2 = [karray,t1,t2,t3,t4]      
    with open(writepath2, 'w') as file:
        for x in zip(*c2):
            file.write("{0}\t{1}\t{2}\t{3}\t{4}\n".format(*x))
    return r1,r2,r3,r4
    
    

    
        
        
