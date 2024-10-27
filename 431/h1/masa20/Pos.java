package masa20;

import vacworld.Direction;

// represents static locations in the map
class Pos {
    int x;
    int y;

    Pos(int x, int y) {
        this.x = x;
        this.y = y;
    }

    // get a position relative to a cardinal direction
    // cardinalDir is the direction the player is facing
    // relative dir is the direction relative to cardinalDir of the position to create
    Pos getRelativePos(int cardinalDir, int relativeDir) {
        int cellDir = (cardinalDir + relativeDir) % 4;
        switch(cellDir) {
            case Direction.NORTH:
                return new Pos(x, y+1);
            case Direction.SOUTH:
                return new Pos(x, y-1);
            case Direction.EAST:
                return new Pos(x+1, y);
            case Direction.WEST:
                return new Pos(x-1, y);
        }
        return null;
    }

    public boolean equals(Object obj) {
        if (!(obj instanceof Pos)) return false;
        Pos comp = (Pos)obj;
        if (this.x == comp.x && this.y == comp.y) return true;
        return false;
    }
}
