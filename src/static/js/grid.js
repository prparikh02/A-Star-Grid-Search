/* 
    Credit: Slavcho Slavchev
    Cite: https://github.com/slavsan/tetris
    Borrowed and Modified with permission on 09.25.2016 
*/

(function (global) {

    function Cell(params) {
        this.$el = params.$element;
        this.x = params.x;
        this.y = params.y;
        // TODO
        this.type = params.type;
        this.hasHighway = params.hasHighway;
        this.f = []
        this.g = []
        this.h = []
    }

    function Grid(params) {
        this.grid = [];
        this.cells = [];
        this.rowCount = params.rows;
        this.colCount = params.cols;
        this.rows = [];
        this.cols = [];
        if (params.render) {
            this.container = params.render.container;
            this.render();
        }
    }

    Grid.prototype = {

        createCell: function (params) {
            return new Cell(params);
        },

        getCellAt: function (x, y) {

            if (x < 0 || x >= this.colCount || y < 0 || y >= this.rowCount) {
                console.log('coordinates (%i, %i) invalid or out of bounds', x, y);
                return false;
            }
            return this.grid[y][x];
        },

        render: function (params) {

            if (params && params.container) {
                this.container = params.container;
            }

            this.$container = $(this.container);
            if (!this.container || this.$container.length === 0) {
                console.error('container is not present');
                return;
            }

            var $row, $cell, cell;
            for (var i = 0; i < this.rowCount; i += 1) {
                this.grid[i] = [];
                $row = $('<div class="row"></div>').appendTo(this.$container);
                for (var j = 0; j < this.colCount; j += 1) {
                    $cell = $('<div class="cell"></div>').appendTo($row);
                    cell = this.createCell({ $element: $cell, x: j, y: i });
                    this.grid[i].push(cell);
                    this.cells.push(cell);
                }
            }

            // rows
            var self = this;
            this.grid.forEach(function (row) {
                self.rows.push(row);
            });
        }
    };

    global.Grid = Grid;

} (window));