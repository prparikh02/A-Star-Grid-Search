import json
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

@app.route('/grid/highlight-trace', methods=['GET', 'POST'])
def highligh_path():
    trace, C = generate_maps.astar(app.G)
    return jsonify(nodes=trace)

if __name__ == "__main__":
    app.run(debug=True)