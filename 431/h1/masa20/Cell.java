package masa20;

import java.util.ArrayList;
import java.util.List;
import vacworld.*;

// represents a single cell in the map
class Cell {
    boolean obstacle = false; // if the cell is an obstacle
    boolean explored = false; // if the cell has been explored
    Pos pos; // cell's x and y position
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
    // creates and indexes a new neighbor Cell when that square has never been seen.
    Cell getRelativeCell(Map map, int relativeDir) {
        int absDir = (map.dir + relativeDir) % 4;
        Cell neighbour = null;
        switch (absDir) {
            case Direction.NORTH: neighbour = n; break;
            case Direction.SOUTH: neighbour = s; break;
            case Direction.EAST: neighbour = e; break;
            case Direction.WEST: neighbour = w; break;
        }
        if (neighbour == null) {
            Pos relPos = pos.getRelativePos(map.dir, relativeDir);

            neighbour = map.getCellByPos(relPos);
            if (neighbour == null) {
                neighbour = new Cell();
                map.index.put(relPos, neighbour);
            }

            switch (absDir) {
                case Direction.NORTH: n = neighbour; break;
                case Direction.SOUTH: s = neighbour; break;
                case Direction.EAST: e = neighbour; break;
                case Direction.WEST: w = neighbour; break;
            }
        }
        return neighbour;
    }

    // set the cell relative to the direction the player is facing
    // cardinalDir is the direction the agent is facing
    // relative dir is the direction relative to cardinalDir of the position to create
    // new cell is the cell to overwrite the found cell with
    void setRelativeCell(int cardinalDir, int relativeDir, Cell newCell) {
        int cellDir = (cardinalDir + relativeDir) % 4;
        switch(cellDir) {
            case Direction.NORTH: n = newCell; break;
            case Direction.SOUTH: s = newCell; break;
            case Direction.EAST: e = newCell; break;
            case Direction.WEST: w = newCell; break;
        }
    }
}
