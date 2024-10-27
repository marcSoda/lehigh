package masa20;

import agent.*;
import pacworld.*;
import java.util.*;

class Brain {
    static final int thresh = 5; // number of times a package needs to have been reported at a single location for it to appear on the map
    int id = -1; // the id of this agent
    Map map = new Map(50, this); // the agent's copy of the map
    Random r = new Random(10); // used to generate random numbers for decision making
    Communicator coms = new Communicator(this); // the communication assembler/parser
    List<Integer> moves; // the list of directions in the agents calculated move set
    int bump = 0; // bump counter. if 2 then we are stuck
    boolean tryPickup = false; // if we just tried to pickup a package
    int dir; // direction of held package relative to agent
    int lastDir = -1; // the last direction that the agent moved in
    Pickup pickupDestBlock; // the pickup action that the agent will take after it drops its package. used when there is a package covering the dropoff location
    int eDropDir = -1; // direction of emergency drop
    boolean idle = false; // whether or not the agent should idle
    Coord phanCoord = null; // if the agent tries to pickup a package but fails, it will broadcast this coordinate to the other agents for removal from the map
    Coord intended; // the coord that the agent intends to go to

    int bumpCount = 0;
    int phanCount = 0;

    Brain(int id) {
        this.id = id;
    }

    // handles updating the environment and communication
    void learn(PacPercept p) {
        map.clearAgents(); // clear visible agents from the last iteration
        for (VisibleAgent a : p.getVisAgents()) {
            // parse the visible agents ids
            int aid = Integer.parseInt(a.getId().replaceAll("[^\\d]", ""));
            // update this agent's location on the map
            if (aid == id) {
                map.me = new Coord(a.getY(), a.getX());
                continue;
            }
            // set the coordinate of the visible agent to avoid
            Coord ac = new Coord(a.getY(), a.getX());
            map.setCoordType(ac, Map.CellType.AGENT);
        }
        // let the agent know it felt a bump
        if (p.feelBump()) {
            bump++;
            bumpCount++;
        }
        map.held = p.getHeldPackage();
        // if our last action was to pickup a package
        if (tryPickup) {
            // if pickup failed, it must have been a phantom
            if (map.held == null) {
                phanCoord = map.me.getAdjacentCoord(this.dir);
                map.erasePackage(phanCoord);
                phanCount++;
            }
            else map.erasePackage(map.getHeldCoord());
            tryPickup = false;
        }
        // update the map
        map.update(p);
    }

    // choose next action
    Action think() {
        // communicate phantom coords for deletion from all agent maps
        if (phanCoord != null) {
            Coord pc = phanCoord;
            phanCoord = null;
            return coms.del(pc);
        }
        // chance that agent will communicate its stored package locations
        if (r.nextDouble() < .008) {
            Say say = coms.packs();
            if (say != null) return say;
        }
        // chance that agent will trim the package locations (delete potential phantoms)
        if (r.nextDouble() < .1) {
            map.pacMap.trim(map);
        }
        // idle if agent should idle
        if (idle) {
            idle = false;
            return new Idle();
        };
        // if a pickup action is queued, return it
        if (pickupDestBlock != null) {
            Pickup pdbc = pickupDestBlock;
            pickupDestBlock = null;
            tryPickup = true;
            return pdbc;
        }
        // pickup a package if there is one to pickup
        if (map.held == null) {
            if (r.nextDouble() < .9) {
                Pickup pickup = tryPickup();
                if (pickup != null) {
                    if (moves != null) moves.clear();
                    return pickup;
                }
            }
        // dropoff a package if we are near a dropoff location
        } else {
            int ddir = tryDropoff();
            if (ddir != -1) {
                if (moves != null) moves.clear();
                return new Dropoff(ddir);
            }
        }
        // if there are no moves or we just bumped
        if (moves == null || moves.isEmpty() || bump > 0) {
            moves = getMoves();
            bump = 0;
            // if moves is still empty, drop the package and try to find a new one
            if (moves == null || moves.isEmpty()) {
                if (map.held != null) {
                    eDropDir = dropAnywhere();
                    return new Dropoff(eDropDir);
                }
                return think();
            }
        }
        int dir = moves.remove(0);
        // if the next move will cause a bump, try again
        if (map.willBump(dir)) {
            moves = getMoves();
            return think();
        }
        lastDir = dir;
        return new Move(dir);
    }

    // try to pickup a package
    Pickup tryPickup() {
        ArrayList<Coord> adjs = map.me.getAdjacentCoords();
        Collections.shuffle(adjs, this.r);
        for (Coord c : adjs) {
            int d = map.me.getAdjacentDir(c);
            if (eDropDir != -1 && d == eDropDir) {
                eDropDir = -1;
                continue;
            }
            Pickup pickup = pickup(c, d);
            if (intended != null && !c.equals(intended)) continue;
            intended = null;
            if (pickup != null) return pickup;
            d++;
        }
        return null;
    }

    // returns a pickup action if there is a valid package adjacent to the agent
    // null otherwise
    Pickup pickup(Coord c, int dir) {
        if (c.inBounds(map.size) && map.coordIsType(c, Map.CellType.PACK)) {
            this.dir = dir;
            this.tryPickup = true;
            return new Pickup(dir);
        }
        return null;
    }

    // try to dropoff a package
    // if there is a package covering the dropoff location, drop tha package randomly and pickup the covering package
    int tryDropoff() {
        int d = 0;
        for (Coord c : map.me.getAdjacentCoords()) {
            if (c.equals(map.getHeldDest())) {
                Pickup pickup = pickup(c, d);
                if (pickup != null) {
                    this.pickupDestBlock = pickup;
                    return dropAnywhere();
                }
                return d;
            }
            d++;
        }
        return -1;
    }

    // drop package in empty adjacent cell
    int dropAnywhere() {
        ArrayList<Coord> adjs = map.me.getAdjacentCoords();
        Collections.shuffle(adjs, this.r);
        for (Coord c : adjs) {
            int d = map.me.getAdjacentDir(c);
            if (map.coordIsType(c, Map.CellType.EMPTY)) {
                return d;
            }
            d++;
        }
        return map.me.getAdjacentDir(map.getHeldCoord());
    }

    // get a series of directions representing the agent's next actions
    LinkedList<Integer> getMoves() {
        // Get coord to navigate to
        Coord c;
        if (map.held == null) {
            // 80% chance to get the closest package 20% for a random package
            // helps with high contention situations
            if (r.nextDouble() < .8)
                c = map.pacMap.getClosestPackageCoord(map.me, map);
            else
                c = map.pacMap.getRandomPackageCoord();
            // if we didn't find package, go to an unexplored coordinate
            if (c == null) c = map.getClosestUnexploredCoord();
            else intended = c;
        // goto dest if holding a package
        } else c = getNonPackageCoordAroundDest(map.getHeldDest());
        // if we still couldn't find a coordinate to navigate to, that mean's we're done
        if (c == null) {
            idle = true;
            return null;
        };

        // Get the path to the target coord
        ArrayList<Coord> optimalPath = getOptimalPath(map.me, c);
        if (optimalPath == null) {
            // if there is no optimal path, trigger emergency drop
            if (map.held != null) {
                eDropDir = dropAnywhere();
                return null;
            };
            map.erasePackage(c);
            return null;
        }
        // Convert path to a list of move directions
        ArrayList<Integer> actions = new ArrayList<Integer>();
        for (int i = 1; i < optimalPath.size(); i++) {
            Coord current = optimalPath.get(i);
            Coord prev = optimalPath.get(i - 1);
            if      (current.x < prev.x) actions.add(Direction.NORTH);
            else if (current.x > prev.x) actions.add(Direction.SOUTH);
            else if (current.y < prev.y) actions.add(Direction.WEST);
            else if (current.y > prev.y) actions.add(Direction.EAST);
        }
        return new LinkedList<Integer>(actions);
    }

    // A* to get optimal path
    public ArrayList<Coord> getOptimalPath(Coord from, Coord to) {
        if (to == null) return null;
        // goto the closest adjacent cell around the destination
        to = getClosestNonSelfDestOrPackageCoord(to);

        // nodes to explore
        PriorityQueue<Node> toExplore = new PriorityQueue<Node>(Comparator.comparingDouble(Node::getF));
        // explored nodes
        HashSet<Node> explored = new HashSet<Node>();
        // holds parents of optimal path nodes
        HashMap<Node, Node> parent = new HashMap<Node, Node>();

        Node start = new Node(from, null, 0, from.dist(to));
        toExplore.add(start);

        while (!toExplore.isEmpty()) {
            // node with lowest f val
            Node current = toExplore.poll();
            // if it's the target, path is calculated
            if (current.coord.equals(to)) {
                ArrayList<Coord> path = new ArrayList<Coord>();
                while (current.parent != null) {
                    // if the optimal path will cause a bump, return null to cause emergency drop
                    if (current.getF() == Double.MAX_VALUE) {
                        if (r.nextDouble() < .5)
                            return null; // will cause emergency drop
                    }
                    path.add(current.coord);
                    current = current.parent;
                }
                path.add(from);
                Collections.reverse(path);
                return path;
            }
            // add the current node to the explored set and expand its neighbors
            explored.add(current);
            for (Coord neighbor : getNeighbors(current.coord)) {
                // convert each neighbor to a node
                double g = current.g + current.coord.dist(neighbor);
                double h = neighbor.dist(to);
                // the location that the held package would be in at this point in the path
                Coord hp = neighbor.getAdjacentCoord(this.dir);
                // set g and h to infinity if bump will be caused at this coord
                if ((map.held != null && !hp.inBounds(map.size)) ||
                    (map.held != null && map.coordIsType(hp, Map.CellType.PACK)) ||
                    map.coordIsType(neighbor, Map.CellType.PACK) ||
                    map.coordIsType(neighbor, Map.CellType.AGENT) ||
                    (bump == 1 && neighbor.equals(map.me.getAdjacentCoord(lastDir)))) {
                    g = Double.MAX_VALUE;
                    h = Double.MAX_VALUE;
                }
                Node node = new Node(neighbor, current, g, h);
                // skip explored node
                if (explored.contains(node)) continue;
                // add to explore list
                if (!toExplore.contains(node)) toExplore.add(node);
                // neighbor is already in the to explore list, update its G value
                else {
                    Node existing = toExplore.stream().filter(n -> n.equals(node)).findFirst().get();
                    if (node.g < existing.g) {
                        toExplore.remove(existing);
                        toExplore.add(node);
                    }
                }
                // update parent
                parent.put(node, current);
            }
        }
        // no path found
        return new ArrayList<Coord>();
    }

    //get closest adjacent coord to the target coord
    public Coord getClosestNonSelfDestOrPackageCoord(Coord from) {
        Coord closest = null;
        double closestDist = Double.POSITIVE_INFINITY;
        for (int i = 0; i < map.size; i++) {
            for (int j = 0; j < map.size; j++) {
                Coord to = new Coord(i, j);
                if (!to.equals(map.getHeldDest()) &&
                    !to.equals(map.me)  &&
                    !map.coordIsType(to, Map.CellType.PACK) &&
                    from.dist(to) < closestDist) {
                    closest = to;
                    closestDist = from.dist(to);
                }
            }
        }
        return closest;
    }

    // get closest adjacent coord to dropoff
    Coord getNonPackageCoordAroundDest(Coord dest) {
        for (Coord c : dest.getAdjacentCoords()) {
            if (!map.getHeldDest().equals(c) &&
                !map.getHeldCoord().equals(c) &&
                !map.coordIsType(c, Map.CellType.PACK) &&
                !map.coordIsType(c.getAdjacentCoord(this.dir), Map.CellType.PACK)) {
                return c;
            }
        }
        return null;
    }

    // get cell neighbors
    private ArrayList<Coord> getNeighbors(Coord coord) {
        ArrayList<Coord> neighbors = new ArrayList<Coord>();
        for (int d = 0; d < 4; d++){
            Coord adjCoord = coord.getAdjacentCoord(d);
            if (adjCoord.inBounds(map.size)) {
                neighbors.add(adjCoord);
            }
        }
        return neighbors;
    }

    // node class used for A*
    public class Node {
        public Coord coord;
        public Node parent;
        public double g;
        public double h;

        public Node(Coord coord, Node parent, double g, double h) {
            this.coord = coord;
            this.parent = parent;
            this.g = g;
            this.h = h;
        }

        public double getF() {
            if (g == Double.MAX_VALUE || h == Integer.MAX_VALUE)
                return Double.MAX_VALUE;
            return g + h;
        }

        @Override
        public boolean equals(Object obj) {
            if (obj == null || getClass() != obj.getClass())
                return false;
            Node other = (Node) obj;
            return coord.equals(other.coord);
        }

        @Override
        public int hashCode() {
            return coord.hashCode();
        }
    }
}
