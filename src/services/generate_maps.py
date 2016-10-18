import heapq
import json
import math
import networkx as nx
import os
import random
import sys
import timeit
import types
from networkx.readwrite import json_graph


def generate_map(rows, cols, file=None, start=None, goal=None):

    if file:
        if start and goal:
            G, map_config = import_map(file, start, goal)
        else:
            G, map_config = import_map(file)
    else:
        map_config = {}
        G = init_map(rows, cols)
        map_config['centroids'] = add_hard_to_traverse_cells(G)
        add_highways(G)
        add_blocked_cells(G)
        s, g = mark_start_and_goal_cells(G)
        map_config['start'] = s
        map_config['goal'] = g
        map_config['rows'] = G.graph['rows']
        map_config['cols'] = G.graph['cols']
        generate_map_file(G, map_config)
    
    calculate_edge_weights(G)
    
    data = json_graph.node_link_data(G)
    return (G, json.dumps(data, indent=4, separators=(',', ': ')))

"""
A node is represented as a string named as:
    'x-y', where x and y are its grid coordinates
"""
def to_node_name(x, y):
    return '%d-%d' % (x, y)

def init_map(rows, cols):
        
    G = nx.Graph(rows=rows, cols=cols)
    
    for i in range(cols):
        for j in range(rows):
            G.add_node(to_node_name(i, j), x=i, y=j, cell_type='unblocked', has_highway=False, tag='1', highway_index=0)
    
    create_edges(G)

    return G

def create_edges(G):
    
    rows = G.graph['rows']
    cols = G.graph['cols']

    # potential edges
    edge_dirs = [[-1, -1], [0, -1], [1, -1], [-1, 0],
             [1, 0], [-1, 1], [0, 1], [1, 1]]
    for n in G.nodes():
        for edge_dir in edge_dirs:
            neighbor = (G.node[n]['x'] + edge_dir[0],
                        G.node[n]['y'] + edge_dir[1])
            if neighbor[0] in range(cols) and neighbor[1] in range(rows):
                G.add_edge(n, to_node_name(neighbor[0], neighbor[1]), weight=1)

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
                        node['tag'] = '2'

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
        G.node[n]['tag'] = '0'
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

    return ((start_x, start_y), (goal_x, goal_y))

def calculate_edge_weights(G):
    
    prices = {
        'unblocked': 1,
        'hard_to_traverse': 2,
        'blocked': sys.maxint
    }

    w = lambda u, v: (prices[G.node[u]['cell_type']] + prices[G.node[v]['cell_type']])/2.0

    for u, v, attr in G.edges(data=True):       
        if (G.node[u]['cell_type'] == 'blocked' or 
            G.node[v]['cell_type'] == 'blocked'):
            G[u][v]['weight'] = sys.maxint
            continue
        diag_move = (G.node[u]['x'] != G.node[v]['x'] and G.node[u]['y'] != G.node[v]['y'])
        on_highway = (G.node[u]['has_highway'] and G.node[v]['has_highway']) 
        if diag_move:
            G[u][v]['weight'] = w(u, v)*math.sqrt(2)
        elif on_highway:
            G[u][v]['weight'] = w(u, v)/4.0
        else:
            G[u][v]['weight'] = w(u, v)


def generate_map_file(G, map_config, file=None):
    
    i = 1
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
                if(not file and G.node[to_node_name(x, y)]['tag'] in "ab"):
                    f.write("%s%d\n" % (G.node[to_node_name(x, y)]['tag'], G.node[to_node_name(x, y)]['highway_index']))
                else:
                    f.write("%s\n" % G.node[to_node_name(x, y)]['tag'])
            else:
                if(not file and G.node[to_node_name(x, y)]['tag'] in "ab"):
                    f.write("%s%d," % (G.node[to_node_name(x, y)]['tag'], G.node[to_node_name(x, y)]['highway_index']))
                else:
                    f.write("%s," % G.node[to_node_name(x, y)]['tag'])

    f.close()

def import_map(file, start=None, goal=None):
    
    map_config = {}
    f = open(file, 'r')

    data = (f.readline()).split('\n') # eliminate \n character
    data = data[0].split(',') # delimit coordinate by ','
    map_config['start'] = start if start else (int(data[0]), int(data[1]))

    data = (f.readline()).split('\n')
    data = data[0].split(',')
    map_config['goal'] = goal if goal else (int(data[0]), int(data[1]))

    data = (f.readline()).split('\n')
    data = data[0].split(',')
    map_config['rows'] = int(data[0])
    map_config['cols'] = int(data[1])

    centroids = []
    for i in range(8):
        data = (f.readline()).split('\n')
        data = data[0].split(',')
        centroids.append((int(data[0]), int(data[1])))
    map_config['centroids'] = centroids
    
    tags = {} # key: map symbol, value = ['cell_type', 'has_highway']
    tags['0'] = ['blocked', False]
    tags['1'] = ['unblocked', False]
    tags['2'] = ['hard_to_traverse', False]
    tags['a'] = ['unblocked', True]
    tags['b'] = ['hard_to_traverse', True]

    rows = map_config['rows']
    cols = map_config['cols']
    G = nx.Graph(rows=rows, cols=cols)
    G.graph['start'] = map_config['start']
    G.graph['goal'] = map_config['goal']

    for i in range(rows): # create nodes from file
        data = (f.readline()).split('\n')
        data = data[0].split(',')

        imp = [] # ['tag', 'highwayindex'] (default is zero, and stays zero if no highway)
        for j in range(cols):
            if("a" in data[j] or "b" in data[j]):
                imp.append([data[j][0], int(data[j][1])])
            else:
                imp.append([data[j], 0])

            G.add_node(to_node_name(j, i), x=j, y=i, cell_type=tags[imp[j][0]][0], has_highway=tags[imp[j][0]][1], tag=imp[j][0], highway_index=imp[j][1])
    f.close()

    sx, sy = G.graph['start']
    gx, gy = G.graph['goal']
    G.node[to_node_name(sx, sy)]['is_start'] = True
    G.node[to_node_name(gx, gy)]['is_goal'] = True

    create_edges(G)
    return (G, map_config)

# Versatile A* implementation
# Inputs: Graph G
# Optional Inputs: starting vertex vs, goal vertex vg, weight w, heuristic option 
# Output: List of coordinates corresponding to lowest-cost path
# TODO: Optimize heap
def Astar(G, vs=None, vg=None, w=1.0, heuristic=None):
        
    if not vs and not vg:
        x, y = G.graph['start']
        vs = to_node_name(x, y)
        x, y = G.graph['goal']
        vg = to_node_name(x, y)
    
    if vs not in G.nodes() or vg not in G.nodes():
        raise ValueError('Starting node and/or goal node not in graph!')

    clear_node_search_properties(G)

    h_fun = get_heuristic(G, heuristic)

    expansions = 0
    fringe = []
    closed = set()
    G.node[vs]['g'] = 0
    G.node[vs]['parent'] = vs
    G.node[vs]['h'] = w*h_fun(G, vs)
    G.node[vs]['f'] = G.node[vs]['g'] + G.node[vs]['h']
    heapq.heappush(fringe, (G.node[vs]['g'] + G.node[vs]['h'], vs))

    while fringe:
        _, v = heapq.heappop(fringe)
        expansions += 1
        if v == vg:
            trace, C = path_trace(G, v)
            return (trace, G.node, C, expansions, len(trace))
        closed.add(v)
        for s in G.neighbors(v):
            if s not in closed:
                G.node[s]['h'] = w*h_fun(G, s)
                # TODO: Need a better fringe data structure not dependent on g being None
                g = G.node[s]['g'] if 'g' in G.node[s] else None
                if not g or (g + G.node[s]['h'], s) not in fringe:
                    G.node[s]['g'] = float('inf')
                    G.node[s]['parent'] = None
                update_vertex(G, v, s, fringe)
    return ([], G.node, -1, expansions, 0) # no path found, C = -1

def update_vertex(G, v, s, fringe):
        
    old_g = G.node[s]['g']
    g = G.node[v]['g'] + G.get_edge_data(v, s)['weight']
    if g < old_g:
        G.node[s]['g'] = g
        G.node[s]['parent'] = v
        h = G.node[s]['h']
        f = G.node[s]['f'] = g + h
        # TODO: Need better fringe structure. Removal is currently O(n)
        s_tup = (old_g + h, s)
        if s_tup in fringe:
            fringe.remove(s_tup)
            heapq.heapify(fringe)
        heapq.heappush(fringe, (f, s))

# Sequential Heuristic A* Implementation
# Inputs: Graph G
# Optional Inputs: starting vertex vs, goal vertex vg, weights w1 and w2
# Output: List of coordinates corresponding to lowest-cost path
# TODO: Optimize heap
# TODO: Need better way to pass in w1/w2/N values
def SHAstar(G, vs=None, vg=None, w1=1.25, w2=1.25):
    
    if not vs and not vg:
        x, y = G.graph['start']
        vs = to_node_name(x, y)
        x, y = G.graph['goal']
        vg = to_node_name(x, y)
    
    if vs not in G.nodes() or vg not in G.nodes():
        raise ValueError('Starting node and/or goal node not in graph!')

    clear_node_search_properties(G)

    N = 5
    expansions = [0 for i in range(N)]
    fringe = [[] for i in range(N)]
    closed = [set() for i in range(N)]
    for i in range(N):
        G.node[vs].setdefault('g', []).append(0)
        G.node[vs].setdefault('parent', []).append(vs)
        G.node[vg].setdefault('g', []).append(float('inf'))
        G.node[vg].setdefault('parent', []).append(None)
        heapq.heappush(fringe[i], (keys(G, vs, i, w1=w1), vs)) 

    while fringe[0][0][0] < float('inf'):
        for i in range(1, N):
            if fringe[i][0][0] <= w2*fringe[0][0][0]:
                if G.node[vg]['g'][i] <= fringe[i][0][0]:
                    if G.node[vg]['g'][i] < float('inf'):
                        trace, C = path_trace_many_heuristics(G, vg, i)
                        return trace, G.node, C, expansions[i], len(trace)
                else:
                    _, s = heapq.heappop(fringe[i])
                    expansions[i] += 1
                    expand_state_SHA(G, s, fringe, closed, i, w1)
                    closed[i].add(s)
            else:
                if G.node[vg]['g'][0] <= fringe[0][0][0]:
                    if G.node[vg]['g'][0] < float('inf'):
                        trace, C = path_trace_many_heuristics(G, vg, 0)
                        return trace, G.node, C, expansions[i], len(trace)
                else:
                    _, s = heapq.heappop(fringe[0])
                    expansions[i] += 1
                    expand_state_SHA(G, s, fringe, closed, 0, w1)
                    closed[0].add(s)

    return ([], G.node, -1, expansions[i], 0) # no path found, C = -1

def expand_state_SHA(G, s, fringe, closed, i, w1=1.25, N=5):
    
    # notation: sp === s'
    for sp in G.neighbors(s):
        # Check to see if sp has ever been generated
        if 'g' not in G.node[sp]:
             G.node[sp]['g'] = [[] for j in range(N)]
             G.node[sp]['h'] = [[] for j in range(N)]
             G.node[sp]['f'] = [[] for j in range(N)]
             G.node[sp]['parent'] = [[] for j in range(N)]
        if G.node[sp]['g'][i] == []:
            G.node[sp]['g'][i] = float('inf')
            G.node[sp]['parent'][i] = None

        sp_tup = (keys(G, sp, i, w1), sp)
        c = G.get_edge_data(s, sp)['weight']
        if G.node[sp]['g'][i] > G.node[s]['g'][i] + c:
            G.node[sp]['g'][i] = G.node[s]['g'][i] + c
            G.node[sp]['parent'][i] = s
            if sp not in closed[i]:
                if sp_tup in fringe[i]:
                    fringe[i].remove(sp_tup)
                    heapq.heapify(fringe[i])
                heapq.heappush(fringe[i], (keys(G, sp, i, w1), sp))

# Integrated Heuristic A* Implementation
# Inputs: Graph G
# Optional Inputs: starting vertex vs, goal vertex vg, weights w1 and w2
# Output: List of coordinates corresponding to lowest-cost path
# TODO: Optimize heap
# TODO: Need better way to pass in w1/w2/N values
def IHAstar(G, vs=None, vg=None, w1=1.25, w2=1.25):
    
    if not vs and not vg:
        x, y = G.graph['start']
        vs = to_node_name(x, y)
        x, y = G.graph['goal']
        vg = to_node_name(x, y)
    
    if vs not in G.nodes() or vg not in G.nodes():
        raise ValueError('Starting node and/or goal node not in graph!')

    clear_node_search_properties(G)

    G.node[vs]['g'] = 0
    G.node[vs]['parent'] = vs
    G.node[vg]['g'] = float('inf')
    G.node[vg]['parent'] = None

    N = 5
    expansions = [0 for i in range(N)]
    fringe = [[] for i in range(N)]
    for i in range(N):
        heapq.heappush(fringe[i], (keys(G, vs, i, w1=w1, isIHA=True), vs)) 

    closed = {
        'anchor': set(),
        'inad': set()
    }

    while fringe[0][0][0] < float('inf'):
        for i in range(1, N):
            if fringe[i][0][0] <= w2*fringe[0][0][0]:
                if G.node[vg]['g'] <= fringe[i][0][0]:
                    if G.node[vg]['g'] < float('inf'):
                        trace, C = path_trace_many_heuristics(G, vg, i, isIHA=True)
                        return trace, G.node, C, expansions[i], len(trace)
                else:
                    _, s = heapq.heappop(fringe[i])
                    expansions[i] += 1
                    expand_state_IHA(G, s, fringe, closed, i, w1, w2)
                    closed['inad'].add(s)
            else:
                if G.node[vg]['g'] <= fringe[0][0][0]:
                    if G.node[vg]['g'] < float('inf'):
                        trace, C = path_trace_many_heuristics(G, vg, 0, isIHA=True)
                        return trace, G.node, C, expansions[i], len(trace)
                else:
                    _, s = heapq.heappop(fringe[0])
                    expansions[i] += 1
                    expand_state_IHA(G, s, fringe, closed, 0, w1, w2)
                    closed['anchor'].add(s)

    return ([], G.node, -1, expansions[i], 0) # no path found, C = -1

def expand_state_IHA(G, s, fringe, closed, i, w1=1.25, w2=2.0, N=5):
    
    s_tup = (keys(G, s, i, w1=w1, isIHA=True), s)
    for i in range(N):
        if s_tup in fringe[i]:
            fringe[i].remove(s_tup)
            heapq.heapify(fringe[i])
    
    # notation: sp === s'
    for sp in G.neighbors(s):
        # Check to see if sp has ever been generated
        if 'g' not in G.node[sp]:
             G.node[sp]['g'] = float('inf')
             G.node[sp]['h'] = [[] for j in range(N)]
             G.node[sp]['f'] = [[] for j in range(N)]
             G.node[sp]['parent'] = None

        sp_tup = (keys(G, sp, i, w1=w1, isIHA=True), sp)
        c = G.get_edge_data(s, sp)['weight']
        if G.node[sp]['g'] > G.node[s]['g'] + c:
            G.node[sp]['g'] = G.node[s]['g'] + c
            G.node[sp]['parent'] = s
            if sp not in closed['anchor']:
                if sp_tup in fringe[0]:
                    fringe[0].remove(sp_tup)
                    heapq.heapify(fringe[0])
                heapq.heappush(fringe[0], (keys(G, sp, 0, w1=w1, isIHA=True), sp))
                if sp not in closed['inad']:
                    for i in range(1, N):
                        if keys(G, sp, i, w1=w1, isIHA=True) <= w2*keys(G, sp, 0, w1=w1,isIHA=True):
                            if sp_tup in fringe[i]:
                                fringe[i].remove(sp_tup)
                                heapq.heapify(fringe[i])
                            heapq.heappush(fringe[i], (keys(G, sp, i, w1=w1, isIHA=True), sp))

def keys(G, s, i, w1=1.25, N=5, isIHA=False):
        
    # H[0] is 'anchor' heuristic which must be admissible
    H = ['euclidean', 'chebyshev_diagonal', 'corner_euclidean', 'manhattan', 'inadmissible_euclidean']
    h_fun = get_heuristic(G, H[i])
    h = h_fun(G, s)

    if 'h' not in G.node[s]:
        G.node[s]['h'] = [[] for j in range(N)]
    if 'f' not in G.node[s]:
        G.node[s]['f'] = [[] for j in range(N)]

    G.node[s]['h'][i] = h
    if isIHA:
        f = G.node[s]['g'] + w1*h
    else:
        f = G.node[s]['g'][i] + w1*h
    G.node[s]['f'][i] = f

    return f

# Heuristic must be a lambda
def get_heuristic(G, heuristic):
    
    D = 0.25 # minimum edge cost (highway)

    dx = lambda G, u: abs(G.node[u]['x'] - G.graph['goal'][0])
    dy = lambda G, u: abs(G.node[u]['y'] - G.graph['goal'][1])

    # Standard distance metrics
    minkowski = lambda G, u, p: D*(dx(G, u)**p + dy(G, u)**p)**(1.0/p)
    man_dist = lambda G, u: minkowski(G, u, 1)
    eucl_dist = lambda G, u: minkowski(G, u, 2)

    mean_abs_err = lambda G, u: man_dist(G, u)/2.0
    mean_sq_err = lambda G, u: (eucl_dist(G, u)**2)/(D*2)

    diag_dist = lambda G, u, D2: D*(dx(G, u) + dy(G, u)) + (D2 - 2*D)*min(dx(G, u), dy(G, u))
    diag_cheb_dist = lambda G, u: diag_dist(G, u, 1)
    diag_octile_dist = lambda G, u: diag_dist(G, u, math.sqrt(2))

    xboundary = G.graph['cols'] if (G.graph['goal'][0] > G.graph['cols']/2) else 0
    yboundary = G.graph['rows'] if (G.graph['goal'][1] > G.graph['rows']/2) else 0

    # custom heuristics (inadmissible)
    # heading toward the corner containing the goal
    corner_man_dist = lambda G, u: D*(abs(G.node[u]['x'] - xboundary) +
                                      abs(G.node[u]['y'] - yboundary))

    corner_eucl_dist = lambda G, u: D*math.sqrt((abs(G.node[u]['x'] - xboundary)**2 +
                                                 abs(G.node[u]['y'] - yboundary)**2))
    inadmissible_eucl_dist = lambda G, u: eucl_dist(G, u)/D

    # inverse of cost since start
    inv_path_cost = lambda G, u: D/(1 + math.sqrt(abs(G.node[u]['x'] - G.graph['start'][0])**2 +
                                                  abs(G.node[u]['y'] - G.graph['start'][1])**2))
    
    if heuristic == 'euclidean':
        return eucl_dist
    elif heuristic == 'manhattan':
        return man_dist
    elif heuristic == 'mean_absolute_error':
        return mean_abs_err
    elif heuristic == 'mean_square_error':
        return mean_sq_err
    elif heuristic == 'chebyshev_diagonal':
        return diag_cheb_dist
    elif heuristic == 'octile_diagonal':
        return diag_octile_dist
    elif heuristic == 'corner_manhattan':
        return corner_man_dist
    elif heuristic == 'corner_euclidean':
        return corner_eucl_dist
    elif heuristic == 'inverse_path_cost':
        return inv_path_cost
    elif heuristic == 'inadmissible_euclidean':
        return inadmissible_eucl_dist
    else:
        return eucl_dist

def path_trace(G, v):
    
    trace = []
    C = 0
    while True:
        trace.append({
            'x': G.node[v]['x'],
            'y': G.node[v]['y'],
            'f': G.node[v]['f'],
            'g': G.node[v]['g'],
            'h': G.node[v]['h']
        })
        p = G.node[v]['parent']
        C += G.get_edge_data(v, p)['weight']
        v = p
        if G.node[v]['parent'] == v:
            trace.append({
                'x': G.node[v]['x'],
                'y': G.node[v]['y'],
                'f': G.node[v]['f'],
                'g': G.node[v]['g'],
                'h': G.node[v]['h']
            })
            break

    trace.reverse()
    return trace, C

def path_trace_many_heuristics(G, v, i, isIHA=False):
    
    trace = []
    C = 0
    while True:
        trace.append({
            'x': G.node[v]['x'],
            'y': G.node[v]['y'],
            'f': G.node[v]['f'][i],
            'g': G.node[v]['g'] if isIHA else G.node[v]['g'][i],
            'h': G.node[v]['h'][i]
        })
        p = G.node[v]['parent'] if isIHA else G.node[v]['parent'][i]
        C += G.get_edge_data(v, p)['weight']
        v = p
        pp = G.node[v]['parent'] if isIHA else G.node[v]['parent'][i]
        if pp == v:
            trace.append({
                'x': G.node[v]['x'],
                'y': G.node[v]['y'],
                'f': G.node[v]['f'][i],
                'g': G.node[v]['g'] if isIHA else G.node[v]['g'][i],
                'h': G.node[v]['h'][i]
            })
            break

    trace.reverse()
    return trace, C

def clear_node_search_properties(G):

    keys = ['parent', 'f', 'g', 'h']
    for n in G.nodes():
        for k in keys:
            G.node[n].pop(k, None)