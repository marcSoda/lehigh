/*  Marc Soda
  * Overview
    My PacAgent implementation seems to work very well in most situations. It was very difficult to produce
    a balanced agent that works well in every possible situation. Performace sacrifices needed to be made in
    more optimal situations to allow the agent to function well in less optimal situations. My main approach
    to creating a dynamic agent was to write it in such a way that it responds optimally to as many situations
    as possible. After doing that, the agent had trouble coping with high-contention situations. To fix this,
    I incorporated a bit of randomness into most of the important decisions that the agent makes. For example,
    sometime the agent will plot a course to closest package, and sometimes it will opt to navigate to a package
    that is further away. There are many such example of this randomness in the code. This approach boosted
    performace for high-contention situations by a significant margin. For example, agents no longer get stuck
    in infinite loops if an optimal path to one coordinate does not exits, they no longer will get caught in a
    loop when trying to navigate around an agent, etc, etc. You will see some occurances in which an agent will
    appear to get caught in a loop for a few moves. It will eventually either choose a different path or
    receive an improved copy of the map which will get it unstuck. The following sections will go into some
    high-level detail about each individual class. If you require more low-level information, each class is
    well-commented and will provide more detail.
  * Coord
    The coord class is used to represent a coordinate within the 50x50 map. Its only members are x and y,
    but it holds a few very useful functions for validating coords and returning coord(s) and directions
    relative to itself.
  * FreqMap
    FreqMap is probably my favorite class. It provides a unique but dynamic solution to the noise issue.
    The packages are represented by a hashmap of hashmaps which map the packageId to a hashmap of coordinates,
    each of which map to the frequency that that coordinate was reported for that id. This allows us to keep
    track of how many times a package was reported at a specific coordinate. The most frequently reported
    coordinate for a package is the likely location of that package. This structure significantly reduces
    if not eliminates the issue of phantom packages, which was a significant obstacle in designing the agent.
    It employs a custom comparator which ensures that the freqmap is always sorted with the closest package at
    the front of the array. The FreqMap has many functions for balancing the maps, deleting possible phantoms,
    retrieving packages given some stipulations, etc.
  * Map
    The map class represents the agent's view of the world. The main data structure is a 50x50 array of cells.
    The map has a sublcass called Cell which contains an enum of PACK, UNEXPLORED, EMPTY, or AGENT which
    describes the state of the cell. If the cell is a PACK, it will contian the id of the package that it
    represents. The map class also keeps track of the coordinate of the agent, the held package, and the
    freqmap. The map class has many useful methods for updating the state of the map.
  * Communicator
    The Communicator class is pretty simple. It serves as a message assembler and parser. When the agent
    wants to send a message, it calls the corresponding function in the communicator which will assemble the
    message and return a Say action. When the agent receives a message, it parses the message and updates
    the state of the agent based on the contents of the message.
  * Brain
    The Brain class is the meat of the program. It keeps track of a ton of state. There are two main methods,
    learn() which takes a percepts and updates the Brain and Map state based on the percept, and think()
    which uses the state of the Brain and Map to produce the agent's next action. It is best to read the
    comments in the Brain class to get a good understanding of the complex interactions happening within
    it. Everything is quite clearly laid out. The Brain class uses A* to navigate from the agent's current
    position to the agent's destination. If the agent is holding a package, the agent will navigate to
    closest adjacent cell of the dropoff location for the currently held package. If it is not holding a
    package, there is a 80% chance that it will navigate to the closest package and pick it up and a 20%
    chance that it will navigate to a random package and pick it up. If at any point during the traversal
    of the calculated path the agent thinks it is about to bump into something, it will try to calculate
    a new path to the destination. If it cannot find a valid path to its destination, it will drop the
    current package (if it has one) and try again. I identified several different situations that an
    individual agent could get into at any time, such as getting stuck in an infinte loop untill the map
    changes, or when an agent boxes itself into a corner, or when two agents are trying to navigate around
    each other and they keep plotting the same conflicting paths. Most of these sorts of niche situations
    are dealt with by incorporating a bit of randomness into the agent's decisions. There are a few other
    major obstacles which have a clear procedure for avoidance such as when the agent is trying to dropoff
    a package but there's another package covering it. When this happens, the agent will drop the package
    in any avalable adjacent cell, pickup the package covering the dropoff location, and deliver it. The
    brain class is pretty complext and has many moving parts which cannot all be described in this overview.
*/

package masa20;
import agent.*;
import pacworld.*;

public class PacAgent extends Agent {
    Brain brain; // agent's brain

    public PacAgent(int id){
        super(id);
        brain = new Brain(id);
    }

    public void see(Percept p){
        brain.learn((PacPercept)p);
    }

    public Action selectAction() {
        return brain.think();
    }
}
