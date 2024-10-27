package vacworld;

import agent.*;
import java.io.*;

/** A simulator for the vacuum cleaning world environment. This environment
    is inaccessible, deterministic, static and discrete. */
public class VacuumWorld extends Environment{

   protected static final int MAX_ACTIONS = 200;

   // scoring information
   protected int numMoves = 0;
   protected int numTurns = 0;
   protected int numSucks = 0;
   protected int numBumps = 0;
   protected int totalScore = 0;

   protected boolean interactive = true;
   protected PrintStream output;

   public VacuumWorld() {
      output = System.out;
      interactive = true;
   }

   public VacuumWorld(boolean interactive) {
      output = System.out;
      this.interactive = interactive;
   }

   public VacuumWorld(PrintStream output, boolean interactive) {

      this.output = output;
      this.interactive = interactive;
   }

   /** Add a new agent to the environment. Since the vacuum cleaning
      world is a single agent environment, this method should only
      be called once per object. */
   public void addAgent(Agent agent) {

      if (agents.isEmpty() == false) {
         output.println("ERROR - tried to add a second agent to a single agent environment");
      }
      else {
         agents.add(agent);
      }
   }


   /** The simulation is complete when the robot has performed its ShutOff
      action. */
   protected boolean isComplete() {

      if (((VacuumState)state).isRobotOff() || timedOut())
         return true;
      else
         return false;
   }

   protected boolean timedOut() {
      if (!interactive && getNumActions() >= MAX_ACTIONS)
         return true;
      else return false;
   }

   /** Create a percept for an agent. This implements the see: E -> Per
      function. */
   protected Percept getPercept(Agent a) {

      VacPercept p;

      if (state instanceof VacuumState) {
         p = new VacPercept((VacuumState)state, a);
         output.println("Pecept: " + p.toString());
         return p;
      }
      else {
         output.println("ERROR - state is not a VacuumState object.");
         return null;
      }
   }


   /** Execute an agent's action and update the environment's state. */
   protected void updateState(Agent a, Action action) {

      if (action instanceof GoForward || action instanceof ShutOff ||
            action instanceof SuckDirt || action instanceof TurnLeft ||
            action instanceof TurnRight)
         super.updateState(a, action);
      else
         System.err.println("ERROR: Invalid action: " + action.toString());

      // update scoring data
      if (action instanceof GoForward)
         numMoves++;
      else if (action instanceof TurnRight || action instanceof TurnLeft)
         numTurns++;
      else if (action instanceof SuckDirt)
         numSucks++;
      if (((VacuumState)state).bumped())
         numBumps++;

      output.println("Action: " + action.toString());
      output.println();
   }


   /** Starts the simulation. We override the method in agent.Environment so
      that VacuumWorld states can be displayed to a selected output stream. */
   public void start(State initState) {

      Percept p;
      Action action;
      Agent a;

      // re-initialize all performance tracking variables
      numBumps = 0;
      numMoves = 0;
      numSucks = 0;
      numTurns = 0;
      totalScore = 0;

      state = initState;
      ((VacuumState)state).display(output);
      if (interactive == true)
         waitForUser();

      a = (Agent)agents.get(0);   // the first agent is the only agent
      boolean quit = false;
      while (!isComplete() && !quit) {
         p = getPercept(a);
         a.see(p);
         action = a.selectAction();
         updateState(a, action);
         ((VacuumState)state).display(output);
         if (interactive == true)
            quit = waitForUser();
      }
      totalScore = getPerformanceMeasure();
   }


   /** Pause simulation until user has pressed a key. Returns true if the user has
    * chosen to quit the simulation. */
   protected boolean waitForUser() {

      System.out.println("Press ENTER to continue. Press Q to quit");

      BufferedReader console = new BufferedReader(new InputStreamReader(System.in));
      try {
         String key = console.readLine();
         if (key.equals("Q") || key.equals("q"))
            return true;
      }
      catch (IOException e) {
         System.out.println(e.getMessage());
         return true;
      }
      return false;
   }

   /** Runs the program. The usage is:
    * <pre>
    * java vacworld.VacuumWorld [-batch] [-rand integer] agentpack
    * </pre>
    * The package containing the agent code (the VacAgent class)
    * must be specified as a command-line parameter. By default, it runs
    * in interactive mode, requiring the user to press a key after each
    * action. To run in batch mode, use the -batch switch. To test the
    * agent in different configurations, use the -rand <i>seed</i>
    * argument, where <i>seed</i> is an integer to be used by the
    * pseudo-random number generator. */
   public static void main(String[] args) {

      VacuumWorld world;
      Agent agent = null;
      VacuumState initState;
      boolean interactive = true;
      boolean randomState = false;
      int randSeed = 0;

      for (int i=0; i < args.length; i++) {
         if (args[i].equals("-batch"))
            interactive = false;
         else if (args[i].equals("-rand")) {
            i++;
            if (i >= args.length) {
               System.err.println("ERROR: Must specify an integer seed when using '-rand'");
               System.exit(1);
            }
            try {
               randSeed = Integer.parseInt(args[i]);
               randomState = true;
            } catch (NumberFormatException ex) {
               System.err.println("ERROR: Seed passed with '-rand' must be an integer.");
               System.exit(1);
            }
         }
         else if (agent == null) {
         	// dynamically load agent class from the package specified on the command line
         	String agentName = null;
         	try {
         		agentName = args[i] + ".VacAgent";
         		ClassLoader myClassLoader = ClassLoader.getSystemClassLoader();
         		Class myClass = myClassLoader.loadClass(agentName);
         		Object o = myClass.newInstance();
         		agent = (Agent)o;
         	} catch (Exception ex) {
         		System.err.println("ERROR: Loading class " + agentName);
         		System.exit(1);
         	}
         	i++;
         }
         else {
            System.err.println("ERROR: Invalid command line arguments.");
            System.err.println("Usage: java vacworld.VacuumWorld [-batch] [-rand integer] agentpack");
            System.exit(1);
         }
      }
      System.out.println("The Vacuum Cleaner World Agent Testbed");
      System.out.println("--------------------------------------");
      System.out.println();

      world = new VacuumWorld(interactive);

      if (randomState) {
         initState = VacuumState.getRandomState(randSeed);
         System.out.println("State generated with seed " + randSeed);
         System.out.println();
      }
      else
         initState = VacuumState.getInitState();
      world.addAgent(agent);
      world.start(initState);
      System.out.println();

      if (world.timedOut()) {
         System.out.println("*** Timeout. Test halted! ***");
         System.out.println();
      }

      world.printScore(System.out);
   }

   protected int getNumDirtyLocs() {

      return ((VacuumState)state).getNumDirtyLocs();
   }

   /** Returns true if the robot has turned itself off in its origin
      square. This method is not used, although it could be used
      in variations of the task that require the agent to return
      home. This is not used for this version of the simulator */
   /*
   public boolean gotHome() {

      VacuumState vstate;

      vstate = (VacuumState)state;
      if (((VacuumState)state).isRobotOff() &&
            vstate.getAgentX() == 1 && vstate.getAgentY() == 1)
         return true;
      else
         return false;
   }
*/

   /** Returns the number of actions the agent has executed. */
   protected int getNumActions() {

      return numMoves + numTurns + numSucks;
   }

   /** Returns the performance measure of an agent team in the current
      environment. Since there is only one agent, we just return that
      agent's performance measure. */
   public int getTeamPerformanceMeasure() {
      return getPerformanceMeasure();
   }

   /** Returns the performance measure of the agent in the current
      environment. Since there is only one agent, the ag parameter
      is ignored. */
   public int getPerformanceMeasure(Agent ag) {
      return getPerformanceMeasure();
   }

   /** Returns the performance measure of the agent in the current
      environment. */
   protected int getPerformanceMeasure() {

      int score;
      // score = 1000 + getMovesScore() + getTurnsScore() + getSucksScore() +
      //    getBumpsScore() + getDirtScore() + getHomeScore();
      score = 1000 + getMovesScore() + getTurnsScore() + getSucksScore() +
      	getBumpsScore() + getDirtScore();
      if (timedOut())
         score = score - 100;
      if (score < 0)
         score = 0;
      return score;
   }

   protected int getMovesScore() {
      return numMoves * -2;
   }

   protected int getTurnsScore() {
      return numTurns * -1;
   }

   protected int getSucksScore() {
      return numSucks * -2;
   }

   protected int getBumpsScore() {
      return numBumps * -10;
   }

   protected int getDirtScore() {
      return getNumDirtyLocs() * -100;
   }

   /*
   protected int getHomeScore() {
      if (gotHome() == false)
         return -200;
      else
         return 0;
   }
*/

   protected void printScore(PrintStream out) {

      out.println("Evaluation:");
      out.println("----------");
      out.println("  Base score: \t1000");
      // out.println("  Home penalty: " + getHomeScore());
      if (getNumActions() >= MAX_ACTIONS)
         out.println("  Loop penalty: -100");
      out.println("  Dirt left: \t" + getDirtScore() + "\t(" +
            getNumDirtyLocs() + ")");
      out.println("  Bumps: \t" + getBumpsScore() + "\t(" + numBumps + ")");
      out.println("  Moves: \t" + getMovesScore() + "\t(" + numMoves + ")");
      out.println("  Turns: \t" + getTurnsScore() + "\t(" + numTurns + ")");
      out.println("  Sucks: \t" + getSucksScore() + "\t(" + numSucks + ")");
      out.println("  ----------------------------");
      out.println("  TOTAL SCORE: \t" + totalScore);
      out.println();
   }


}
