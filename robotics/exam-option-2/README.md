# CSE 298 - Foundations of Robotics - Exam Option 2
Due by: 8/16/2021 by 5PM EST

## Ethics Contract

**FIRST**: Please read the following carefully:

- I have not received, I have not given, nor will I give or receive, any assistance to another student taking this exam, including discussing the exam with students in another section of the course. Do not discuss the exam after you are finished until final grades are submitted.
- If I use any resources (including text books and online references and websites), I will cite them in my assignment.
- I will not ask question about how to debug my exam code on Piazza or any other platform.
- I will not plagiarize someone else's work and turn it in as my own. If I use someone else's work in this exam, I will cite that work. Failure to cite work I used is plagiarism.
- I understand that acts of academic dishonesty may be penalized to the full extent allowed by the [Lehigh University Code of Conduct][0], including receiving a failing grade for the course. I recognize that I am responsible for understanding the provisions of the Lehigh University Code of Conduct as they relate to this academic exercise.

If you agree with the above, type your full name in the following space along with the date. Your exam **will not be graded** without this assent. When you are done, **make your first commit with the commit message: `I, <your full name here>, agree to the ethics contract`.**

Write your name and date between the lines below

---------------------------------------------
Marcantonio Soda Jr. 8/16/21
---------------------------------------------

### Part 1

Implement one of the following algorithms:

- A* - Implement the A* algorithm in Matlab. The input to the algorithm can be an adjacency matrix, which is easy to implement in Matlab. The graph will represent a grid world, where each cell connects to its 8 neighbors, with `1` dist to the adjacent ones and `2^.5` distance to the diagonal ones. Represent obstacles by removing edges to neighboring cells. The input to the algorithm should be the start and goal cells. Make a figure that shows the start, goal, any obstacles, and the path found by the algorithm.

- FastSLAM algorithm, a good summary of which is in the Probabilistic Robotics book on page 450. Homework 4 would be a good starting place for this. You can use the same bearing only cameras, or you can add the range in there if you want. It should work either way. You can also use the same robot model and the same noise model. Or you can just add Gaussian noise to everything. Explore how the algorithm performs as you vary the number of particles in the filter. Compare to the EKF filter.

- Particle Filter - Implement a particle filter algorithm in Matlab. Homework 4 might be a good place to start with this one. You can keep the same cameras and the same bearing measurements, or you can also add in range. The robot model can be the same too. But instead of tracking a mean and covariance, you'll track a number of hypothetical robot particles. The spread of those robots will represent uncertainty, and you can take the mean of their positions as a guess for where they are. You can find a good particle filter algorithm on page 96 of the Probabilistic Robotics book.  Explore how the algorithm performs as you vary the number of particles in the filter. Compare to the EKF filter.

- Trajectory Rollout - Implement the Trajectory Rollout algorithm described by [Gerkey and Konolige](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.330.2120&rep=rep1&type=pdf). Also described in [Lecture 8](https://www.youtube.com/watch?v=buEfiJftc0E&list=PL4A2v89SXU3SUUNrwKcE-yy2SX6YQOg_p&index=9&t=0s).

- Q Learning - Implement the Q Learning algorithm for a simple 2D grid-world robot. The robot can take 3 actions: move one cell forward, turn left or turn right. Its objective is to reach a goal cell in the grid-world. You can set up a reward system where the robot recieives a reward of 100 if it enters the goal cell, and zero otherwise. Implement the Q learning algorithm as covered in class. Once the algorithm is implemented, initialize your agent with a zeroed Q-function (or a random Q-function, it doesn't matter as the Q-function will converge either way... that's the great part about Q learning!). Now you have to train your agent. First, use an off-policy approach, meaning the robot will take random actions only (one of move forward, turn left, turn right). After each action, apply the appropriate award according the schedule. After enough random actions, the agent will eventually make its way to the goal cell. When it does, reset the robot to a random cell, and repeat the process. Soon enough, the Q-function will begin to converge, meaning that the robot is learning how to reach the goal. Switch the behavior of the robot to be "on-policy" meaning it will make the decision of which action to take based on the expect reward implied by the Q-function.

### Part 2

Demonstrate the algorithm working with a video. It's easy to [record your screen using zoom](https://support.zoom.us/hc/en-us/articles/201362473-Local-Recording). Either give me a voice annotation of what's going on, or write it out in this README with timestamps indicating what is happening.
