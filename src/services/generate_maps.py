import json
import matplotlib.pyplot as plt
import networkx as nx
import random
from networkx.readwrite import json_graph


def generate_map(rows, cols):

    G = init_grid(rows, cols)
    add_hard_to_traverse_cells(G)
    add_highways(G)

    data = json_graph.node_link_data(G)
    return json.dumps(data, indent=4, separators=(',', ': '))

def init_grid(rows, cols):
        
    G = nx.Graph(rows=rows, cols=cols)
    
    for i in range(cols):
        for j in range(rows):
            G.add_node(to_node_name(i, j), x=i, y=j, cell_type='unblocked', has_highway=False)

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

"""
A node is represented as a string named as:
    'x-y', where x and y are its grid coordinates
"""
def to_node_name(x, y):
    return '%d-%d' % (x, y)

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

def add_highways(G):
    
    for i in range(4):
        highways = []
        while not highways:
            highways = create_highway(G)

        for x, y in highways:
            G.node[to_node_name(x, y)]['has_highway'] = True

def create_highway(G):
    
    highway = []
    x, y, direction = get_highway_starting_point(G.graph['rows'], G.graph['cols'])
    if G.node[to_node_name(x, y)]['has_highway']:
        return []

    highway = [(x, y)]
    steps = 1

    while True:
        for i in range(20):
            x, y = get_next_highway_point(x, y, direction)
            if (G.node[to_node_name(x, y)]['has_highway'] or
                (x, y) in highway):
                return []
            
            if is_boundary_point(G, G.node[to_node_name(x, y)]):
                if steps >= 100:
                    highway.append((x, y))        
                    return highway
                return []

            highway.append((x, y))
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

def is_boundary_point(G, n):
    
    return (n['x'] in [0, G.graph['cols']-1] or 
            n['y'] in [0, G.graph['rows']-1])
