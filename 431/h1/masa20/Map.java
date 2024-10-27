package masa20;

import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Queue;
import vacworld.*;

// the internal state of the Agent
class Map implements Iterable<Cell> {
    Cell head;
    int dir;

    Map() {
        head = new Cell(new Pos(0, 0), false);
        dir = Direction.NORTH;
    }

    // build out a more complete picture of the environment
    // returns true if dirt is under the cell. false otherwise
    boolean process(final VacPercept p) {
        // observe and link the cell in front of the head
        Cell forward = head.getRelativeCell(this, Direction.NORTH);
        forward.pos = head.pos.getRelativePos(dir, Direction.NORTH);
        forward.obstacle = p.seeObstacle();
        forward.setRelativeCell(dir, Direction.SOUTH, this.head);
        // link the cell beind the head
        Cell backward = head.getRelativeCell(this, Direction.SOUTH);
        backward.pos = head.pos.getRelativePos(dir, Direction.SOUTH);
        backward.setRelativeCell(dir, Direction.NORTH, this.head);
        // link the cell to the west of the head
        Cell left = head.getRelativeCell(this, Direction.WEST);
        left.pos = head.pos.getRelativePos(dir, Direction.WEST);
        left.setRelativeCell(dir, Direction.EAST, this.head);
        // link the cell to the east of the head
        Cell right = head.getRelativeCell(this, Direction.EAST);
        right.pos = head.pos.getRelativePos(dir, Direction.EAST);
        right.setRelativeCell(dir, Direction.WEST, this.head);
        head.explored = true;
        if (p.seeDirt()) return true;
        return false;
    }

    // move the head forward
    void goForward() {
        head = head.getRelativeCell(this, Direction.NORTH);
    }

    // update the direction to the left
    void turnLeft() {
        dir = (dir + 3) % 4;
    }

    // update the direction to the right
    void turnRight() {
        dir = (dir + 1) % 4;
    }

    // search the map for a cell based on its position
    Cell getCellByPos(Pos pos) {
        for (Cell c : this) {
            if (c.pos.equals(pos)) {
                return c;
            }
        }
        return null;
    }

    @Override
    public Iterator<Cell> iterator() {
        return new MapIter(head);
    }

    // iterator over all cells in the map
    // traverses the map graph in a breadth-first manner
    private class MapIter implements Iterator<Cell> {
        private Queue<Cell> queue;
        private HashSet<Cell> seen;

        public MapIter(Cell start) {
            this.queue = new LinkedList<>();
            this.seen = new HashSet<>();
            this.queue.add(start);
            this.seen.add(start);
        }

        @Override
        public boolean hasNext() {
            return !queue.isEmpty();
        }

        @Override
        public Cell next() {
            Cell next = queue.poll();
            for (Cell c : next.getNeighbors()) {
                if (!seen.contains(c)) {
                    queue.add(c);
                    seen.add(c);
                }
            }
        return next;
        }
    }
}
