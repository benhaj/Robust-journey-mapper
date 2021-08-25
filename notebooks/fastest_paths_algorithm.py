import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from heapq import heappush, heappop
from itertools import count
import math

LINE_WALKING = 'walking'
LINE_NOTHING = 'nothing'

TRANSPORT_TYPES = ['bus', 'train', 'metro', 'tram']
WALKING_TYPE = 'walking'

def custom_weight_function(edge_attr, current_time):
    if(current_time is None):
        if(edge_attr['type'] == WALKING_TYPE):
            return math.inf
        elif(edge_attr['type'] in TRANSPORT_TYPES):
            return edge_attr['arrival'] - edge_attr['departure']
    else:
        if(edge_attr['type'] == WALKING_TYPE):
            return edge_attr['walk_duration']
        elif(edge_attr['type'] in TRANSPORT_TYPES and current_time <= edge_attr['departure']):
            return edge_attr['arrival'] - current_time
        elif(edge_attr['type'] in TRANSPORT_TYPES and current_time > edge_attr['departure']):
            return math.inf
        
def extract_hashable_edge(raw_edge):
    (edge, edge_data) = raw_edge

    return (edge, (edge_data['departure'], edge_data['route_id'], edge_data['direction_id']))

def extract_hashable_path(raw_path):
    result = [
        extract_hashable_edge(raw_edge) for raw_edge in raw_path
    ]

    return tuple(result)

def extract_change_points(edge_path):
    change_points = []
    
    for idx, edges_tuple in enumerate(zip(edge_path, edge_path[1:])):
        trip_info_succ = (edges_tuple[1][1]['route_id'], edges_tuple[1][1]['direction_id'])
        trip_info_pred = (edges_tuple[0][1]['route_id'], edges_tuple[0][1]['direction_id'])
        
        if(trip_info_succ != trip_info_pred):
            change_points.append((idx, idx+1))
    
    return change_points

def update_ignore_edges(ignore_edges_rules, edge_path, prev_edge_path):
    edge_rule_to_add = None
    
    if(prev_edge_path is not None):
        first_unique_trip_info_prev = (prev_edge_path[0][1]['departure'], prev_edge_path[0][1]['route_id'], prev_edge_path[0][1]['direction_id'])
    else:
        first_unique_trip_info_prev = None
    
    
    first_unique_trip_info_succ = (edge_path[0][1]['departure'], edge_path[0][1]['route_id'], edge_path[0][1]['direction_id'])
    
    if(prev_edge_path is not None and 
       first_unique_trip_info_prev != first_unique_trip_info_succ and
       not((first_unique_trip_info_prev, LINE_NOTHING) in ignore_edges_rules)):
        
        edge_rule_to_add = (first_unique_trip_info_prev, LINE_NOTHING)
        ignore_edges_rules.add(edge_rule_to_add)
        do_change = False   
        
        
    else:
        do_change = True
        change_points = extract_change_points(edge_path)
    
        last_change_point_idx = None

        for idx, change_point in enumerate(reversed(change_points)):
            unique_tuple_succ = (edge_path[change_point[1]][1]['route_id'], 
                                 edge_path[change_point[1]][1]['direction_id'])
            
            unique_tuple_prev = (edge_path[change_point[0]][1]['departure'],
                                 edge_path[change_point[0]][1]['route_id'], 
                                 edge_path[change_point[0]][1]['direction_id'])

            constructed_rule = (unique_tuple_prev, 
                                unique_tuple_succ)

            if(constructed_rule not in ignore_edges_rules):
                edge_rule_to_add = constructed_rule
                last_change_point_idx = idx
                break
                
        ignore_edges_rules.add(edge_rule_to_add)
    
    return do_change

def edge_is_allowed(ignore_edges_rules, edge, pred_edge):
    if pred_edge is None:
        unique_tuple = (edge['departure'], edge['route_id'], edge['direction_id'])
        
        rule = (unique_tuple, LINE_NOTHING)
    else:
        unique_tuple_succ = (edge['route_id'], edge['direction_id'])
        unique_tuple_pred = (pred_edge[1]['departure'], pred_edge[1]['route_id'], pred_edge[1]['direction_id'])
        
        rule = (unique_tuple_pred, unique_tuple_succ)
    
    return not(rule in ignore_edges_rules)

def custom_dijkstra(
    G, source, target, weight, ignore_edges_rules=set(), current_time=None
):
    """Uses Dijkstra's algorithm to find shortest weighted paths

    Parameters
    ----------
    G : NetworkX graph

    source : Starting node for the path.

    weight: function
        Function with (u, v, data) input that returns that edges weight

    target : Ending node for path. Search is halted when target is found.
        

    Returns
    -------
    path : dictionary
        A mapping from node to shortest distance to that node from one
        of the source nodes.

    Raises
    ------
    NodeNotFound
        If any of `sources` is not in `G`.

    """
    if source == target:
        raise nx.NetworkXAlgorithmError('same source and target')

    G_succ = G._succ

    push = heappush
    pop = heappop
    dist = {}  # dictionary of final distances
    seen = {}
    # fringe is heapq with 3-tuples (distance,c,node)
    # use the count c to avoid comparing nodes (may not be able to)
    c = count()
    fringe = []
    paths = {
        node: [] for node in G.nodes
    }
    edge_paths = {
        node: [] for node in G.nodes
    }

    if source not in G:
        raise nx.NodeNotFound(f"Source {source} not in G")
    seen[source] = 0
    
    pred_edge = None

    ####################################
    
    push(fringe, (0, next(c), source, current_time, pred_edge))
        
    ####################################
    
    while fringe:
        ####################################
        
        (d, _, v, current_time, pred_edge) = pop(fringe)
        
        ####################################
        if v in dist:
            continue  # already searched this node.
        dist[v] = d
        if v == target:
            break
        for u, e in G_succ[v].items():
            
            ####################################
        
            multi_edge_data = ((v, u), e)
            
            weights = [
                weight(single_edge_data, current_time) 
                for _, single_edge_data in multi_edge_data[1].items()
                if edge_is_allowed(ignore_edges_rules, single_edge_data, pred_edge)
            ]  
    
            corresponding_edges = [
                (multi_edge_data[0], single_edge_data)
                for _, single_edge_data in multi_edge_data[1].items()
                if edge_is_allowed(ignore_edges_rules, single_edge_data, pred_edge)
            ]
    
            if(len(weights) == 0):
                cost = math.inf
            else :
                min_cost_idx = np.argmin(weights)
                cost = weights[min_cost_idx]
                corr_edge = corresponding_edges[min_cost_idx]
            
            ####################################
            
            if cost == math.inf:
                continue
            vu_dist = dist[v] + cost
            if u in dist:
                u_dist = dist[u]
                if vu_dist < u_dist:
                    raise ValueError("Contradictory paths found:", "negative weights?")

            elif u not in seen or vu_dist < seen[u]:
                seen[u] = vu_dist   
                
                ####################################
            
                if(current_time is None):
                    push(fringe, (vu_dist, next(c), u, corr_edge[1]['departure'] + cost, 
                                  corr_edge))
                else:
                    push(fringe, (vu_dist, next(c), u, current_time + cost, 
                                  corr_edge))
                
                ####################################
                
                paths[u] = paths[v] + [u]
                edge_paths[u] = edge_paths[v] + [corr_edge]
                    
                ####################################


    ####################################
    
    if(len(paths[target]) == 0):
        raise nx.NetworkXNoPath('No path between source and target')
        
    ####################################

    return dist[target], paths[target], edge_paths[target]

def trip_duration(path):
    initial_time = path[0][1]['departure']
    last_time = path[-1][1]['arrival']

    return last_time - initial_time
    
class PathBuffer:
    def __init__(self):
        self.paths = set()
        self.sortedpaths = list()
        self.counter = count()

    def __len__(self):
        return len(self.sortedpaths)

    
    def push(self, cost, path):
        hashable_path = extract_hashable_path(path)
        
        if hashable_path not in self.paths:
            heappush(self.sortedpaths, (cost, next(self.counter), path))
            self.paths.add(hashable_path)

    def pop(self):
        (cost, num, path) = heappop(self.sortedpaths)
        hashable_path = extract_hashable_path(path)
        self.paths.remove(hashable_path)
        return path
    
def all_shortest_simple_paths(G, source, target):
    """Generate all simple paths in the graph G from source to target,
       starting from shortest ones.

    A simple path is a path with no repeated nodes.

    No negative weights are allowed.
    
    Operates on MultiDiGraph

    Parameters
    ----------
    G : NetworkX graph

    source : node
       Starting node for path

    target : node
       Ending node for path

    weight : function
        The weight of an edge is the value returned by the function. 
        The function must accept exactly two positional arguments: 
        The edge under consideration, and the current time at which we are considering the edge. 
        The function must return a number.

    Returns
    -------
    path_generator: generator
       A generator that produces lists of simple paths, in order from
       shortest to longest.
       A path is a sequence of edges, with the first edge having its as first endpoint the source node
       and the last edge having as second endpoint the target node.

    Raises
    ------
    NetworkXNoPath
       If no path exists between source and target.

    NetworkXError
       If source or target nodes are not in the input graph.

    NetworkXNotImplemented
       If the input graph is a Multi[Di]Graph.

    Notes
    -----
    This function is an adaptation (to handle MultiDiGraph)
    of shortest_simple_paths from networkx.algorithms.simple_paths.

    """
    if source not in G:
        raise nx.NodeNotFound(f"source node {source} not in graph")

    if target not in G:
        raise nx.NodeNotFound(f"target node {target} not in graph")

    wt = custom_weight_function
    

    shortest_path_func = custom_dijkstra

    listB = PathBuffer()
    prev_edge_path = None
    ignore_edges_rules = set()
    just_removed_base_edge = False
    while True:
        try:
            length, path, edge_path = shortest_path_func(G, source, target, wt,  
                                                         ignore_edges_rules=ignore_edges_rules)
            listB.push(length, edge_path)
            
            if(just_removed_base_edge):
                just_removed_base_edge = False
            
        except nx.NetworkXNoPath: 
            pass
        except nx.NetworkXAlgorithmError:
            pass

        if listB:
            edge_path = listB.pop()
            yield edge_path
            do_change = update_ignore_edges(ignore_edges_rules, edge_path, prev_edge_path)
            if(do_change):
                prev_edge_path = edge_path
        else:
            break
            
def serialize_path(path):
    result = ''
    
    sep = ';'
    
    for one_edge in path:
        one_edge_string = f"(({one_edge[0][0]},{one_edge[0][1]}),{one_edge[1]['trip_id']})"
        result = result + one_edge_string + sep
        
    return result[:-1]

def serialize_time_agnostic_path(path):
    result = ''
    
    sep = ';'
    
    for one_edge in path:
        one_edge_string = f"(({one_edge[0][0]},{one_edge[0][1]}),({one_edge[1]['route_id']}, {one_edge[1]['direction_id']}))"
        result = result + one_edge_string + sep
        
    return result[:-1]

def get_arrival(path):
    return path[-1][1]['arrival']

def get_departure(path):
    return path[0][1]['departure']

def get_shortest_paths_DFs(G, s, t):
    all_fastest_paths = all_shortest_simple_paths(G, s, t)
    
    paths_dict = {
        'path_id': [],
        'path' : [],
        'length' : [],
        'departure': [],
        'arrival' : [],
        'risk' : [],
        'path_time_agnostic': []
    }

    paths_edges_dict = {
        'path_id' : [],
        'source' : [],
        'target': [],
        'departure': [],
        'route_id': [],
        'direction_id': [],
    }
    
    paths_summarized_dict = {
        'path_id' : [],
        'source' : [],
        'target' : [],
        'type': [],
        'departure': [],
        'arrival': [],
        'route_id' : [],
        'direction_id' : [],
        'departure_next' : [],
        'type_next': [],
        'duration' : []
    }

    for idx, path in enumerate(all_fastest_paths):
        paths_dict['path_id'].append(idx)
        paths_dict['path'].append(serialize_path(path))
        paths_dict['length'].append(trip_duration(path))
        paths_dict['departure'].append(get_departure(path))
        paths_dict['arrival'].append(get_arrival(path))
        paths_dict['risk'].append(0)
        paths_dict['path_time_agnostic'].append(serialize_time_agnostic_path(path))
        
        for one_edge in path:
            paths_edges_dict['path_id'].append(idx)
            paths_edges_dict['source'].append(one_edge[0][0])
            paths_edges_dict['target'].append(one_edge[0][1])
            paths_edges_dict['departure'].append(one_edge[1]['departure'])
            paths_edges_dict['route_id'].append(one_edge[1]['route_id'])
            paths_edges_dict['direction_id'].append(one_edge[1]['direction_id'])
        
        change_points = extract_change_points(path)
        
        first_point = path[0]
        
        last_point = path[-1]
        
        current_point = last_point
       
        for change_point in change_points:
            current_point = path[change_point[0]]
            paths_summarized_dict['path_id'].append(idx)
            paths_summarized_dict['source'].append(first_point[0][0])
            paths_summarized_dict['target'].append(current_point[0][1])
            paths_summarized_dict['departure'].append(first_point[1]['departure'])
            paths_summarized_dict['arrival'].append(current_point[1]['arrival'])
            paths_summarized_dict['type'].append(first_point[1]['type'])
            paths_summarized_dict['route_id'].append(first_point[1]['route_id'])
            paths_summarized_dict['direction_id'].append(first_point[1]['direction_id'])
            paths_summarized_dict['departure_next'].append(path[change_point[1]][1]['departure'])
            paths_summarized_dict['type_next'].append(current_point[1]['type'])
            paths_summarized_dict['duration'].append(first_point[1]['walk_duration'])
            first_point = path[change_point[1]]

        paths_summarized_dict['path_id'].append(idx)
        paths_summarized_dict['source'].append(first_point[0][0])
        paths_summarized_dict['target'].append(last_point[0][1])
        paths_summarized_dict['departure'].append(first_point[1]['departure'])
        paths_summarized_dict['arrival'].append(last_point[1]['arrival'])
        paths_summarized_dict['type'].append(first_point[1]['type'])
        paths_summarized_dict['route_id'].append(first_point[1]['route_id'])
        paths_summarized_dict['direction_id'].append(first_point[1]['direction_id'])
        paths_summarized_dict['departure_next'].append(None)
        paths_summarized_dict['type_next'].append(None)
        paths_summarized_dict['duration'].append(first_point[1]['walk_duration'])
       
    paths_df = pd.DataFrame(paths_dict)
    paths_df.sort_values(['arrival', 'length'], ascending=[False, True], inplace=True)
    
    ta_paths = paths_df.groupby('path_time_agnostic').count().reset_index()
    ta_paths['ta_path_id'] = ta_paths.index
    ta_paths = ta_paths[['path_time_agnostic', 'ta_path_id']]

    paths_df = pd.merge(paths_df, ta_paths, left_on='path_time_agnostic', right_on='path_time_agnostic')
    
    paths_edges_df = pd.DataFrame(paths_edges_dict)
    
    paths_summarized_df = pd.DataFrame(paths_summarized_dict)
        
    return all_fastest_paths, paths_df, paths_edges_df, paths_summarized_df