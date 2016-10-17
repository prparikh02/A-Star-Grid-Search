import json
import timeit
import sys
sys.path.insert(0, './services/')
from networkx.readwrite import json_graph
from services import generate_maps
from flask import Flask, jsonify, render_template, request


# Subclasses Flask to hold graph G as global variable.
#   This is a temporary solution for session data
class GridServer(Flask):
    
    def __init__(self, *args, **kwargs):
        super(GridServer, self).__init__(*args, **kwargs)
        self.G = []
        self.node_search_data = []

app = GridServer(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/grid')
@app.route('/grid/<int:map_num>/<int:sg_pair_num>')
def grid(map_num=None, sg_pair_num=None):
    rows = 120
    cols = 160
    if map_num and sg_pair_num:
        map_file = './src/services/sample_maps/map%s.txt' % map_num
        start_goal_file = './src/services/sample_maps/startgoal%s.txt' % map_num
        
        if map_num not in range(1,6):
            map_num = 1
        if sg_pair_num not in range(1, 11):
            sg_pair_num = 1
        with open(start_goal_file) as fp:
            for i, line in enumerate(fp):
                if i == sg_pair_num-1:
                    start, goal = line.split('|')
                    start = tuple(int(i) for i in start.split(','))
                    goal = tuple(int(i) for i in goal.split(','))
                    break
        G, graph_data = generate_maps.generate_map(rows, cols, map_file, start, goal)
    else:
        G, graph_data =  generate_maps.generate_map(rows, cols)

    app.G = G

    return render_template('grid.html', 
                            rows=rows, 
                            cols=cols, 
                            graph_data=graph_data)

@app.route('/grid/Astar')
def astar():
    if request.args.get('w'):
        w = float(request.args.get('w'))
    else:
        w = 1.0
    
    start = timeit.default_timer()
    trace, app.node_search_data, C, expansions, moves = generate_maps.astar(app.G, w=w, heuristic=request.args.get('h'))
    elapsed = timeit.default_timer() - start

    return jsonify(trace=trace, cost=C, expansions=expansions, moves=moves, time=elapsed)

@app.route('/grid/SHAstar')
def shastar():
    if request.args.get('w1'):
        w1 = float(request.args.get('w1'))
    else:
        w = 1.0

    if request.args.get('w2'):
        w2 = float(request.args.get('w2'))
    else:
        w2 = 1.0
    
    start = timeit.default_timer()
    trace, app.node_search_data, C, expansions, moves = generate_maps.shastar(app.G, w1=w1, w2=w2)
    elapsed = timeit.default_timer() - start

    return jsonify(trace=trace, cost=C, expansions=expansions, moves=moves, time=elapsed)

@app.route('/grid/astar/node-stats')
def node_stats():
    if not app.node_search_data:
        return 'run a search first!'

    x = int(request.args.get('c'))
    y = int(request.args.get('r'))
    n = generate_maps.to_node_name(x, y)

    f = app.node_search_data[n]['f'] if 'f' in app.node_search_data[n] else None
    g = app.node_search_data[n]['g'] if 'g' in app.node_search_data[n] else None
    h = app.node_search_data[n]['h'] if 'h' in app.node_search_data[n] else None
    return jsonify({
        'f': f,
        'g': g,
        'h': h
    })

if __name__ == "__main__":
    app.run(debug=True)