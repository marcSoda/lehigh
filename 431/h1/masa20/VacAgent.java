package masa20;

import agent.*;
import vacworld.*;

/*
  * Intro and Benchmarking
    This VacAgent implement is, I think, very efficient. I benchmarked it with the BASH script
    that I provided with this code (called bench). My implementation is only 375 lines of code.
    I tested it on 5000 seeds (0-4999) and it achieved an average score of 896, never once
    dipped below a score of 820 and only dipped below 840 about 4 times. Note that I didn't
    rigorously follow all of the production-level Java conventions because there was no reason
    to and I don't like the style of it. For example, I don't have a getter and a setter for every
    state var nor did I privatize all of the functions and variables that may be able to be privatized.
  * Overview:
    Everything is processed in the Brain class (described later). It contains a reference
    to the Map of the environment, and uses that to calculate a series of moves that the
    agent should take.
  * Cell: Each time the agent moves, it further constructs and updates a graph of cells.
    Each cell has references to the cells around it as well as the position of the cell,
    whether or not the cell has been explored, and whether or not the cell is an obstacle.
    Note that it is not at all necessary to store whether or not the cell has dirt on it
    because when dirt is detected, the Agent's path is interrupted immediately. It is
    impossible for the Agent to pass over a cell that contains dirt without sucking it up.
    This interconnection of cells creates a graph that can be traversed nicely.
  * Map:
    The Map object (stored by the brain) has a reference to the cell that the agent is currently
    on (head) and the direction of the agent. The map is all that is required to effectively
    represent the agent's internal state. Map's `process` function takes a percept and applies
    the information to the Map by adding cells to the graph and propagating updated information.
    The Map has a nice iterator which is used to traverse the entire graph in a breadth-first
    manner.
  * Pos:
    The `Pos` class simply stores the position of the Cell. It's not all that necessary. When
    I started building this out, I thought it would be more instrumental, but now it could
    probably be removed with some light refactoring. It like it thought. Makes things a bit more
    concise.
  * Brain:
    The meat of the program is in the Brain. The Brain stores a reference to the Map and a LinkedList
    of the moves that the Agent intends to take. These moves are not always followed though. For example,
    if the brain previously calculated that it wants to visit or should pass through a cell which has since
    been realized to be an obstacle, the agent will not hit the obstacle, it will recalculate the move set.
    Also, if the Agent is over dirt, it will ignore the move list and sick it up, resuming reading from the
    move set on the next iteration. To generate a set of moves, the Brain simply calls getMoves() which will
    grab the lowest cost node that is not explored and not an obstacle. It does so by leveraging the
    `estimateTotalCost()` function which estimates the cost of all of the moves and turns the agent will need
    to take to get to each cell. The cell with the lowest total cost is returned. Note that if there are no
    remaining unexplored cells, getMoves returns null which triggers the Agent to shutoff. It then gets a list
    of cells which represents the most optimal path from the head to the target cell. Note that this path is
    simply a list of cells, the move set is calculated later. The most optimal path to the least-cost cell
    is calculated using a variation of the A* algorithm. Once the most optimal path has been calculated, it
    is translated into a set of Actions that the agent may take to reach the target cell.
*/

public class VacAgent extends Agent {
    // stores the state of the agent and environment. coordinates the agent's moves
    Brain brain = new Brain();

    // update state
    public void see(Percept p) {
        brain.learn((VacPercept)p);
    }

    // get the next action the agent should take
    public Action selectAction() {
        return brain.think();
    }

    public String getId() {
        return "CreamMachine";
    }
}
