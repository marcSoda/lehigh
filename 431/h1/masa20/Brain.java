package masa20;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Queue;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.PriorityQueue;
import agent.Action;
import vacworld.*;

// serves to update and store the map and agent state
// and coordinate the agent's moves

class Brain {
    Map map; // Agent's env map
    List<Action> moves; // optimal set of Actions for agent to take
    boolean suckNow = false; // override next Action in moves with suck action

    Brain() { map = new Map(); }

    // update the environment, instructs the agent to suck now if need be
    void learn(VacPercept p) {
        suckNow = map.process(p);
    }

    // returns an Action that describes what the agent should do next
    Action think() {
        if (suckNow) return new SuckDirt();
        // get moves if there are none
        if (moves == null || moves.isEmpty())
            moves = getMoves();
        // shutoff if moves is still null
        if (moves == null) return new ShutOff();
        Action move = moves.remove(0);
        if (move instanceof GoForward) {
            // ensure that the agent never hits an obstacle by recalculating the moves
            if (map.head.getRelativeCell(map, Direction.NORTH).obstacle) {
                moves.clear();
                return think();
            }
            map.goForward();
        } else if (move instanceof TurnRight) {
            map.turnRight();
        } else if (move instanceof TurnLeft) {
            map.turnLeft();
        }
        return move;
    }


    // a* algorithm to get the optimal path between two cells
    // note that the returned path has no information about direction
    // the returned path will need to be operated on to get a sequence
    // of Actions that included turn info
    ArrayList<Cell> getOptimalPath(Cell from, Cell target) {
        HashMap<Cell, Integer> gScore = new HashMap<>();
        PriorityQueue<Cell> open = new PriorityQueue<>(
            Comparator.comparingInt(c -> gScore.get(c) + estimateTotalCost(c, target)));
        HashMap<Cell, Cell> parent = new HashMap<>();
        gScore.put(from, 0);
        open.add(from);
        while (!open.isEmpty()) {
            Cell current = open.poll();
            if (current == target) {
                ArrayList<Cell> path = new ArrayList<>();
                for (Cell c = current; c != null; c = parent.get(c))
                    path.add(0, c);
                return path;
            }
            int curDir = parent.containsKey(current)
                        ? getDirectionTowards(parent.get(current), current)
                        : map.dir;
            for (Cell neighbor : current.getNeighbors()) {
                if (neighbor == null || neighbor.obstacle) continue;
                int stepDir = getDirectionTowards(current, neighbor);
                int d = Math.abs(curDir - stepDir);
                int turnCost = d == 3 ? 1 : d;
                int tentativeG = gScore.get(current) + 2 + turnCost;
                if (tentativeG < gScore.getOrDefault(neighbor, Integer.MAX_VALUE)) {
                    parent.put(neighbor, current);
                    gScore.put(neighbor, tentativeG);
                    open.add(neighbor);
                }
            }
        }
        return new ArrayList<>(); // unreachable
    }

    // iterates over the map (BFS) and returnes the cell that can be reached with the lowest cost
    Cell getLowestCostUnexplored() {
        Queue<Cell> q = new LinkedList<>();
        HashSet<Cell> seen = new HashSet<>();
        q.add(map.head);
        seen.add(map.head);

        while (!q.isEmpty()) {
            Cell cur = q.poll();
            if (!cur.obstacle && !cur.explored)
                return cur;
            for (Cell neighbor : cur.getNeighbors()) {
                if (neighbor == null || neighbor.obstacle || seen.contains(neighbor))
                    continue;
                seen.add(neighbor);
                q.add(neighbor);
            }
        }
        return null;
    }

    // estimates the cost to travel between two cells
    int estimateTotalCost(Cell start, Cell end) {
        return estimateTurnCost(end) + (estimateMoveCost(start, end) * 2);
    }

    // estimates the cost of the turn(s) requred to travel between two cells
    int estimateTurnCost(Cell targetCell) {
        int targetDirection = getDirectionTowards(map.head, targetCell);
        int turnDirection = (targetDirection - map.dir + 4) % 4;
        return Math.min(turnDirection, 4 - turnDirection);
    }

    // gets the relative direction from one cell to another
    int getDirectionTowards(Cell current, Cell target) {
        int x = target.pos.x - current.pos.x;
        int y = target.pos.y - current.pos.y;
        if (Math.abs(x) >= Math.abs(y))
            return x > 0 ? Direction.EAST : Direction.WEST;
        else
            return y > 0 ? Direction.NORTH : Direction.SOUTH;
    }

    // estimates the cost of the move(s) required to travel between two cells
    int estimateMoveCost(Cell start, Cell end) {
        int x = start.pos.x - end.pos.x;
        int y = start.pos.y - end.pos.y;
        return Math.abs(x) + Math.abs(y);
    }

    // get the optimal series of moves that the agent should take to move from the
    // head to the lowestCostCell
    // the algorithm basically walks the optimal path and adds turns when they are needed
    LinkedList<Action> getMoves() {
        Cell target = getLowestCostUnexplored();             // next cell to traverse to
        if (target == null) return null;                     // indicates graph has been fully explored
        List<Cell> path = getOptimalPath(map.head, target);  // the path to take from head to target
        LinkedList<Action> moves = new LinkedList<Action>(); // set of actions to be built
        int dir = map.dir;                                   // hypothetical direction
        // walk the path
        for (int i = 0; i < path.size() - 1; i++) {
            Cell cur = path.get(i);
            Cell next = path.get(i + 1);
            int towards = getDirectionTowards(cur, next); // the direction that the next cell is in
            // decide how to add turn(s)
            while (dir != towards) {
                int diff = (towards - dir + 4) % 4;
                if (diff == 1) {
                    moves.add(new TurnRight());
                    dir = (dir + 1) % 4;
                } else {
                    moves.add(new TurnLeft());
                    dir = (dir + 3) % 4;
                }
            }
            moves.add(new GoForward());
            dir = towards;
        }
        return moves;
    }
}
