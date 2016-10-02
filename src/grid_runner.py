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

@app.route('/grid/run-classic-astar')
def run_classic_astar():
    start = timeit.default_timer()
    trace, C, expansions, moves = generate_maps.astar(app.G, w=2.0)
    elapsed = timeit.default_timer() - start
    print C, expansions, moves, elapsed
    return jsonify(nodes=trace, cost=C, expansions=expansions, moves=moves, time_elapsed=elapsed)

if __name__ == "__main__":
    app.run(debug=True)