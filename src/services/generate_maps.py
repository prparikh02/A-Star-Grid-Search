import heapq
import json
import math
import networkx as nx
import os
import random
import sys
import timeit
from networkx.readwrite import json_graph


def generate_map(rows, cols):

    map_config = {}
    G = init_grid(rows, cols)
    map_config['centroids'] = add_hard_to_traverse_cells(G)
    add_highways(G)
    add_blocked_cells(G)
    s, g, r, c = mark_start_and_goal_cells(G)
    map_config['start'] = s
    map_config['goal'] = g
    map_config['rows'] = r
    map_config['cols'] = c

    generate_map_file(G, map_config)

    data = json_graph.node_link_data(G)
    return (G, json.dumps(data, indent=4, separators=(',', ': ')))

"""
A node is represented as a string named as:
    'x-y', where x and y are its grid coordinates
"""
def to_node_name(x, y):
    return '%d-%d' % (x, y)

def init_grid(rows, cols):
        
    G = nx.Graph(rows=rows, cols=cols)
    
    for i in range(cols):
        for j in range(rows):
            G.add_node(to_node_name(i, j), x=i, y=j, cell_type='unblocked', has_highway=False, tag=1)

    # potential edges
    edge_dirs = [[-1, -1], [0, -1], [1, -1], [-1, 0],
             [1, 0], [-1, 1], [0, 1], [1, 1]]
    for n in G.nodes():
        for edge_dir in edge_dirs:
            neighbor = (G.node[n]['x'] + edge_dir[0],
                        G.node[n]['y'] + edge_dir[1])
            if neighbor[0] in range(cols) and neighbor[1] in range(rows):
                G.add_edge(n, to_node_name(neighbor[0], neighbor[1]), weight=1)

    return G

def add_hard_to_traverse_cells(G):

    coords = [(random.randint(0,G.graph['cols']-1), 
               random.randint(0,G.graph['rows']-1)) for i in range(8)]

    # check 31x31 block centered at coords
    radius = 15
    for center in coords:
        for i in range(center[0]-radius, center[0]+radius+1):
            if i not in range(0, G.graph['cols']):
                continue
            for j in range(center[1]-radius, center[1]+radius+1):
                if j not in range(0, G.graph['rows']):
                    continue
                if random.uniform(0, 1) < 0.50:
                    node = G.node[to_node_name(i, j)]
                    if node['cell_type'] != 'hard_to_traverse':
                        node['cell_type'] = 'hard_to_traverse'
                        node['tag'] = 2

    return coords

def add_highways(G):
    
    for i in range(4):
        highways = []
        while not highways:
            highways = create_highway(G)

        for x, y in highways:
            G.node[to_node_name(x, y)]['has_highway'] = True
            if G.node[to_node_name(x, y)]['cell_type'] == 'unblocked':
                 G.node[to_node_name(x, y)]['tag'] = 'a%d' % (i+1)
            elif G.node[to_node_name(x, y)]['cell_type'] == 'hard_to_traverse':
                G.node[to_node_name(x, y)]['tag'] = 'b%d' % (i+1)

def create_highway(G):

    # highways is a dict of (x, y) pairs of highway points
    #   implemented as a dict for O(1) lookup for duplicates    
    highway = {}
    x, y, direction = get_highway_starting_point(G.graph['rows'], G.graph['cols'])
    if G.node[to_node_name(x, y)]['has_highway']:
        return {}

    highway[(x, y)] = True
    steps = 1

    while True:
        for i in range(20):
            x, y = get_next_highway_point(x, y, direction)
            if (G.node[to_node_name(x, y)]['has_highway'] or
                (x, y) in highway):
                return {}
            
            if is_boundary_node(G, G.node[to_node_name(x, y)]):
                if steps >= 100:
                    highway[(x, y)] = True        
                    return highway
                return {}

            highway[(x, y)] = True
            steps += 1

        direction = choose_next_direction(direction)

def choose_next_direction(curr_direction):
    
    if curr_direction in ['n', 's']:
        perp_directions = ['e', 'w']
    else:
        perp_directions = ['n', 's']

    if random.uniform(0, 1) < 0.60:
        return curr_direction
    return random.choice(perp_directions)

def get_next_highway_point(x, y, direction):
    
    if direction == 'n':
        return x, y+1
    elif direction == 's':
        return x, y-1
    elif direction == 'e':
        return x+1, y
    else:
        return x-1, y
        
def get_highway_starting_point(row_count, col_count):
    
    # starting point for the highway should be at boundary,
    #   but not at the corners
    side = random.choice(['n', 's', 'e', 'w'])
    if side == 'n':
        return random.randint(1, col_count-1), row_count-1, 's'
    elif side == 's':
        return random.randint(1, col_count-1), 0, 'n'
    elif side == 'e':
        return col_count-1, random.randint(1, row_count-1), 'w'
    else:
        return 0, random.randint(1, row_count-1), 'e'

def is_boundary_node(G, n):
    
    return (n['x'] in [0, G.graph['cols']-1] or 
            n['y'] in [0, G.graph['rows']-1])

def add_blocked_cells(G):
    
    N = len(G.nodes()) # total nodes in graph
    B = int(N*0.20) # number of nodes to block

    for n in random.sample(G.nodes(), N):
        if B == 0:
            break
        if G.node[n]['has_highway']:
            continue
        G.node[n]['cell_type'] = 'blocked'
        G.node[n]['tag'] = 0
        B -= 1

def mark_start_and_goal_cells(G):

    col_count = G.graph['cols']
    row_count = G.graph['rows']
    valid_x_range = range(0, 20) + range(col_count-20, col_count)
    valid_y_range = range(0, 20) + range(row_count-20, row_count)
    
    while True:
        start_x = random.choice(valid_x_range)
        start_y = random.choice(valid_y_range)
        if G.node[to_node_name(start_x, start_y)]['cell_type'] != 'blocked':
            break
    
    while True:
        goal_x = random.choice(valid_x_range)
        goal_y = random.choice(valid_y_range)
        if (G.node[to_node_name(goal_x, goal_y)]['cell_type'] != 'blocked' and
            (math.sqrt((start_x-goal_x)**2 + (start_y-goal_y)**2)) > 100):
            break

    G.graph['start'] = (start_x, start_y)
    G.graph['goal'] = (goal_x, goal_y)
    G.node[to_node_name(start_x, start_y)]['is_start'] = True
    G.node[to_node_name(goal_x, goal_y)]['is_goal'] = True

    return ((start_x, start_y), (goal_x, goal_y), row_count, col_count)

def generate_map_file(G, map_config):
    
    i = 0
    while os.path.exists('map%s.txt' % i):
        i += 1

    f = open('map%d.txt' % i, 'w')

    f.write('%d,%d\n' % map_config['start'])
    f.write('%d,%d\n' % map_config['goal'])
    f.write('%d,%d\n' % (map_config['rows'], map_config['cols']))

    for j in range(8):
        f.write("%d,%d\n" % map_config['centroids'][j])

    for y in range(G.graph['rows']):
        for x in range(G.graph['cols']):
            if x == G.graph['cols'] - 1:
                f.write("%s\n" % G.node[to_node_name(x, y)]['tag'])
            else:
                f.write("%s," % G.node[to_node_name(x, y)]['tag'])

    f.close()

# Classic A* implementation.
# Inputs: Graph G, starting vertex vs, goal vertex vg
# Output: List of coordinates corresponding to lowest-cost path
def astar(G, vs=None, vg=None, heuristic=None):
    
    if not vs and not vg:
        x, y = G.graph['start']
        vs = to_node_name(x, y)
        x, y = G.graph['goal']
        vg = to_node_name(x, y)
    
    if vs not in G.nodes() or vg not in G.nodes():
        raise ValueError('Starting node and/or goal node not in graph!')

    add_heuristics(G, heuristic)

    fringe = []
    closed = []
    G.node[vs]['g'] = 0
    G.node[vs]['parent'] = vs
    heapq.heappush(fringe, (G.node[vs]['g'] + G.node[vs]['h'], vs))

    while fringe:
        _, v = heapq.heappop(fringe)
        if v == vg:
            return path_trace(G, v)
        closed.append(v)
        for succ in G.neighbors(v):
            if succ not in closed:
                g = G.node[succ]['g'] if 'g' in G.node[succ] else None
                if not g or (g + G.node[succ]['h'], succ) not in fringe:
                    G.node[succ]['g'] = sys.maxint
                    G.node[succ]['parent'] = None
                update_vertex(G, v, succ, fringe)
    return 'no path found'

def update_vertex(G, v, succ, fringe):
    
    g = G.node[succ]['g']
    if G.node[v]['g'] + G.get_edge_data(v, succ)['weight'] < g:
        G.node[succ]['g'] = G.node[v]['g'] + G.get_edge_data(v, succ)['weight']
        G.node[succ]['parent'] = v
        succ_tup = (g + G.node[succ]['h'], succ)
        if succ_tup in fringe:
            fringe.remove(succ_tup)
        heapq.heappush(fringe, (G.node[succ]['g'] + G.node[succ]['h'], succ))

def path_trace(G, v):
    
    trace = []
    C = 0
    while True:
        trace.append({'x': G.node[v]['x'], 'y': G.node[v]['y']})
        p = G.node[v]['parent']
        C += G.get_edge_data(v, p)['weight']
        v = p
        if G.node[v]['parent'] == v:
            trace.append({'x': G.node[v]['x'], 'y': G.node[v]['y']})
            break
    trace.reverse()
    return (trace, C)

def add_heuristics(G, heuristic=None):

    if not heuristic:
        goal_x, goal_y = G.graph['goal']
        for n in G.nodes():
            x = G.node[n]['x']
            y = G.node[n]['y']
            G.node[n]['h'] = math.sqrt((x-goal_x)**2 + (y-goal_y)**2)
            # G.node[n]['g'] = None
            G.node[n]['parent'] = None
