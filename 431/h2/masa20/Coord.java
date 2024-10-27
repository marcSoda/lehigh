package masa20;

import pacworld.*;
import java.util.*;

public class Coord {
    int x, y;

    Coord(int x, int y) {
        this.x = x;
        this.y = y;
    }

    // squared euclidean distance between coords
    public int dist(Coord other) {
        int dx = this.x - other.x;
        int dy = this.y - other.y;
        return dx * dx + dy * dy;
    }

    // returns the adjacent coordinate in the given direction
    public Coord getAdjacentCoord(int direction) {
        switch (direction) {
            case Direction.NORTH: return new Coord(this.x-1, this.y);
            case Direction.SOUTH: return new Coord(this.x+1, this.y);
            case Direction.EAST: return new Coord(this.x, this.y+1);
            case Direction.WEST: return new Coord(this.x, this.y-1);
            default: return this;
        }
    }

    // get coords around this
    ArrayList<Coord> getAdjacentCoords() {
        ArrayList<Coord> adj = new ArrayList<>();
        for (int d = 0; d < 4; d++)
            adj.add(d, this.getAdjacentCoord(d));
        return adj;
    }

    // get the direction of the coord relative to this
    public int getAdjacentDir(Coord c) {
        if (c.x == this.x - 1 && c.y == this.y) return Direction.NORTH;
        else if (c.x == this.x + 1 && c.y == this.y) return Direction.SOUTH;
        else if (c.x == this.x && c.y == this.y + 1) return Direction.EAST;
        else if (c.x == this.x && c.y == this.y - 1) return Direction.WEST;
        else return -1;
    }

    // check if coord is in the bounds of the map
    public boolean inBounds(int size) {
        if (this.x >= 0 && this.x < size && this.y >= 0 && this.y < size) return true;
        return false;
    }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof Coord)) return false;
        Coord c = (Coord)o;
        if (c.x == this.x && c.y == this.y) return true;
        else return false;
    }

    @Override
    public int hashCode() {
        return Objects.hash(x, y);
    }

    @Override
    public String toString() {
        return "[" + this.x + "," + this.y + "]";
    }

    public static Coord fromString(String s) throws IllegalArgumentException {
        if (s == null || !s.matches("\\{\\s*\\d+\\s*,\\s*\\d+\\s*\\}")) {
            throw new IllegalArgumentException("Invalid coordinate string: " + s);
        }
        String[] parts = s.replaceAll("\\{\\s*|\\s*\\}", "").split(",");
        int x = Integer.parseInt(parts[0].trim());
        int y = Integer.parseInt(parts[1].trim());
        return new Coord(x, y);
    }
}
