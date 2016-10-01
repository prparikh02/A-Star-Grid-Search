import json
import timeit
import sys
sys.path.insert(0, './services/')
from networkx.readwrite import json_graph
from services import generate_maps
from flask import Flask, jsonify, render_template


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
def grid():
    rows = 120
    cols = 160
    G, graph_data =  generate_maps.generate_map(rows, cols)
    app.G = G
    return render_template('grid.html', 
                            rows=rows, 
                            cols=cols, 
                            graph_data=graph_data)

@app.route('/grid/run-classic-astar', methods=['GET', 'POST'])
def run_classic_astar():
    start = timeit.default_timer()
    trace, C, expansions, moves = generate_maps.astar(app.G, w=2.0)
    elapsed = timeit.default_timer() - start
    print C, expansions, moves, elapsed
    return jsonify(nodes=trace, cost=C, expansions=expansions, moves=moves, time_elapsed=elapsed)

if __name__ == "__main__":
    app.run(debug=True)