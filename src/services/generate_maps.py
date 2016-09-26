import json
import matplotlib.pyplot as plt
import networkx as nx
import random
from networkx.readwrite import json_graph


def generate_map(rows, cols):

    G = init_grid(rows, cols)
    add_hard_to_traverse_cells(G)

    data = json_graph.node_link_data(G)
    return json.dumps(data, indent=4, separators=(',', ': '))

def init_grid(rows, cols):
        
    G = nx.Graph(rows=rows, cols=cols)
    # TODO: Checks for x,y
    for i in range(cols):
        for j in range(rows):
            G.add_node(to_node_name(i, j), cell_type='unblocked', x=i, y=j)

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

    print G.graph['cols'], G.graph['rows']

    coords = [(random.randint(0,G.graph['cols']-1), 
               random.randint(0,G.graph['rows']-1)) for i in range(8)]
    print coords

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
