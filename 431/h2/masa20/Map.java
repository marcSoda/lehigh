package masa20;

import pacworld.*;

class Map {
    Cell[][] map; // agent's representation of the map
    Coord me = new Coord(-1,-1); // location of this agent
    FreqMap pacMap; // frequency map for package locations
    VisiblePackage held = null; // held package
    int size; // size of square map
    Brain brain; // agent's brain

    Map(int size, Brain brain) {
        this.brain = brain;
        this.size = size;
        this.map = new Cell[size][size];
        this.pacMap = new FreqMap();
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                map[i][j] = new Cell(CellType.UNEXPLORED);
            }
        }
    }

    // update the map given a percept
    void update(PacPercept p) {
        // parse all messages
        for (String m : p.getMessages()) brain.coms.parse(m);
        // add package to map if it is valid
        for (VisiblePackage vp : p.getVisPackages()) {
            Coord loc = new Coord(vp.getY(), vp.getX());
            Coord dest = new Coord(vp.getDestY(), vp.getDestX());
            if (!loc.inBounds(size) || !dest.inBounds(size) || (held != null && vp.getId() == held.getId())) continue;
            addPack(vp.getId(), loc);
        }
        setExploredCells(me); // update the map to reflext that this location was explored
        // System.out.println(this); // print the agent's map
    }

    // clear visible agents from the map
    void clearAgents() {
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                if (map[i][j].type == CellType.AGENT)
                    map[i][j].type = CellType.EMPTY;
            }
        }
    }

    // add package to the map and freqmap
    void addPack(int id, Coord loc) {
        pacMap.addElm(id, loc, 1);
        this.clearCellsByID(id);
        Coord pac = pacMap.getMostFrequentCoord(id);
        if (pacMap.getMaxCountForCoord(pac) > 1)
            this.map[pac.x][pac.y] = new Cell(CellType.PACK, id);
    }

    // get the coordinate of the held package
    Coord getHeldCoord() {
        if (this.held == null) return null;
        return new Coord(this.held.getY(), this.held.getX());
    }

    // get the destination of the held package
    Coord getHeldDest() {
        if (this.held == null) return null;
        return new Coord(this.held.getDestY(), this.held.getDestX());
    }

    // check if coord is of type type
    boolean coordIsType(Coord coord, CellType type) {
        if (!coord.inBounds(size) || map[coord.x][coord.y].type != type) return false;
        return true;
    }

    // check if agent will bump if it moves in direction
    public boolean willBump(int direction) {
        int x = me.x;
        int y = me.y;
        switch (direction) {
            case 0: x--; break;
            case 2: x++; break;
            case 1: y++; break;
            case 3: y--; break;
            default: throw new IllegalArgumentException("Invalid direction");
        }
        if (map[x][y].type == CellType.PACK || !new Coord(x, y).inBounds(size)) {
            return true; // will bump into package
        }
        return false;
    }

    // get the closest unexplored coordinate
    public Coord getClosestUnexploredCoord() {
        Coord closestUnexploredCoord = null;
        double closestDistance = Double.MAX_VALUE;
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                if (map[i][j].type == CellType.UNEXPLORED) {
                    Coord currentCoord = new Coord(i, j);
                    double distance = me.dist(currentCoord);
                    if (distance < closestDistance) {
                        closestDistance = distance;
                        closestUnexploredCoord = currentCoord;
                    }
                }
            }
        }
        return closestUnexploredCoord;
    }

    // set all cells in radius of 5 from center to be explored
    void setExploredCells(Coord center) {
        for (int i = center.x-5; i <= center.x+5; i++) {
            for (int j = center.y-5; j <= center.y+5; j++) {
                if (new Coord(i, j).inBounds(size) && map[i][j].type == CellType.UNEXPLORED)
                    map[i][j].type = CellType.EMPTY;
            }
        }
    }

    // set cells with id to be empty
    void clearCellsByID(int id) {
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                if (map[i][j].id == id) {
                    map[i][j].type = CellType.EMPTY;
                }
            }
        }
    }

    // set the coord to be of type type
    void setCoordType(Coord c, CellType type) {
        map[c.x][c.y].type = type;
    }

    // erase a package from the map and the freq map
    void erasePackage(Coord c) {
        if (c == null || !c.inBounds(size)) return;
        pacMap.del(map[c.x][c.y].id);
        pacMap.del(c);
        setCoordType(c, Map.CellType.EMPTY);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("  ");
        for (int j = 0; j < size; j++) {
            sb.append(String.format("%02d", j));
        }
        sb.append("\n");

        for (int i = 0; i < size; i++) {
            // Add row number on the left
            sb.append(String.format("%02d", i));
            for (int j = 0; j < size; j++) {
                Cell cell = map[i][j];
                if (i == me.x && j == me.y) {
                    sb.append("\u001B[32m\u25A0 \033[0m");
                } else if (cell.type == CellType.UNEXPLORED) {
                    sb.append("- ");
                } else if (cell.type == CellType.EMPTY) {
                    sb.append("+ ");
                } else if (cell.type == CellType.AGENT) {
                    sb.append("\u001B[31m\u25A0 \033[0m");
                } else if (cell.type == CellType.PACK) {
                    sb.append("P ");
                }
            }
            sb.append("\n");
        }
        return sb.toString();
    }

    enum CellType {
        UNEXPLORED,
        EMPTY,
        PACK,
        AGENT,
    }

    // representation of a cell
    public class Cell {
        int id = -1;
        CellType type;

        Cell(CellType type) {
            this.type = type;
        }

        Cell(CellType type, int id) {
            this.type = type;
            this.id = id;
        }
    }
}
