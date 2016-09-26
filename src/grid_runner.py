import sys
sys.path.insert(0, './services/')
from services import generate_maps
from flask import Flask, render_template, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/grid')
def grid():
    rows = 120
    cols = 160
    graph_data =  generate_maps.generate_map(rows, cols)
    return render_template('grid.html', rows=rows , cols=cols , graph_data=graph_data)


if __name__ == "__main__":
    app.run(debug=True)