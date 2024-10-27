package masa20;

import java.util.ArrayList;
import java.util.List;
import vacworld.*;

// represents a single cell in the map
class Cell {
    // if the cell is an obstacle
    boolean obstacle = false;
    // if the cell has been explored
    boolean explored = false;
    Pos pos; //cell's x and y position
    Cell n;  // cell to the north
    Cell s;  // cell to the south
    Cell e;  // cell to the east
    Cell w;  // cell to the west

    // construct an observed cell
    Cell(Pos pos, boolean obstacle) {
        this.pos = pos;
        this.obstacle = obstacle;
    }

    // construct blank cell when passing without observing
    Cell() {}

    // get a list of the cell's neighbors
    List<Cell> getNeighbors() {
        List<Cell> neighbors = new ArrayList<>();
        if (n != null) neighbors.add(n);
        if (s != null) neighbors.add(s);
        if (e != null) neighbors.add(e);
        if (w != null) neighbors.add(w);
        return neighbors;
    }

    // gets the neighbor cell that is in a direction relative to the direction the agent is facing
    Cell getRelativeCell(Map map, int relativeDir) {
        switch((map.dir + relativeDir) % 4) {
            case Direction.NORTH:
                if (n == null) {
                    n = map.getCellByPos(pos.getRelativePos(map.dir, relativeDir));
                    if (n == null) n = new Cell();
                }
                return n;
            case Direction.SOUTH:
                if (s == null) {
                    s = map.getCellByPos(pos.getRelativePos(map.dir, relativeDir));
                    if (s == null) s = new Cell();
                }
                return s;
            case Direction.EAST:
                if (e == null) {
                    e = map.getCellByPos(pos.getRelativePos(map.dir, relativeDir));
                    if (e == null) e = new Cell();
                }
                return e;
            case Direction.WEST:
                if (w == null) {
                    w = map.getCellByPos(pos.getRelativePos(map.dir, relativeDir));
                    if (w == null) w = new Cell();
                }
                return w;
        }
        return null;
    }

    // set the cell relative to the direction the player is facing
    // cardinalDir is the direction the player is facing
    // relative dir is the direction relative to cardinalDir of the position to create
    // new cell is the cell to overwrite the found cell with
    void setRelativeCell(int cardinalDir, int relativeDir, Cell newCell) {
        int cellDir = (cardinalDir + relativeDir) % 4;
        switch(cellDir) {
            case Direction.NORTH:
                n = newCell;
                return;
            case Direction.SOUTH:
                s = newCell;
                return;
            case Direction.EAST:
                e = newCell;
                return;
            case Direction.WEST:
                w = newCell;
                return;
        }
    }
}
