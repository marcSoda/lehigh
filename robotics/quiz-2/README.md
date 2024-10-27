# CSE 298 - Quiz 2

Remember, make at least 1 commit per question. You should have at least 3 commits for this quiz.

# Question 1

What is localization and why is it important in robotics?
- Localization is the process by which a robot determines where it is in its environment.
- Localization is extremly important for local robots. If the robot can't localize itself, it almost certainly won't be able to effectively perform its task.
  * For example, if a self-driving car has trouble localizing itself, it could think it's on the road when really it's on the sidewalk.

# Question 2

Describe how predictor-corrector algorithms better estimate the state of a system compared to dead reconing. As part of your answer, describe how dead reckoning works.
- Dead reckoning is the process by which a robot attempts to estimate its position using internal sensors. It is typically very innacurate as it does not take landmarks into account.
- Predictor-corrector algorithms are superior to dead reckoning because it allows the robot use landmarks to better localize itself. It employs a form of dead reckoning to estimate its position, but then finds a landmark to significantly reduce the error in localization.

# Question 3

Compare and contrast the following algorithms:

- Kalman Filter
  * a predictor-corrector localization algorithm that assumes that everything is gaussian and all models are linear.
- Extended Kalman Filter
  * a predictor-corrector localization algorithm with relaxed assumptions when compared to the Kalman filter. This allows it to be applied to non-linear systems.
  * this is done by applying the algorithm in small (linear-like) increments.
- Particle Filter
  * a predictor-corrector localization algorithm.
  * non-linear, like EKF
  * Sample particles are taken from the distribution, assigned a weight, then resampled in order.
  * Like the Kalman Filters, the particle filter relies heavily on landmarks to reduce error.
  * Unlike the Kalman Filters, the particle filter solves the "kidnapped robot" problem
