# CSE 298 - Quiz 3

Due: 7/28 by end of day

Answer the following questions. Remember, make one commit per question.

# Question 1

What is the benefit of open loop control over feedback control? What is the benefit of feedback control over open loop control?

- Open loop control systems are typically simpler and therefore cheaper. Many systems do not require feedback control.
- Feedback control allows for the controller to receive feedback from the system to make edjustments. Open loop control systems do not take feedback into account.

# Question 2

What is the input to a feedback controller? What process provides this input?
The input to the feedback controller is the output of the feedback controller. The plant process provides this input.

# Question 3

Pretend you are an control systems engineer for Ford Mototor Company. You are test driving a new cruise control system, and you are driving around the test track at 63 miles per hour. You set the cruise control, which is implemented using a PID controller, to 65 miles per hour. The cruise controller takes over and speeds up to 68 miles per hour, then slows to 63 miles per hour, and finally settles at 66 miles per hour. What kind of behavior is the controller exhibiting? What adjustments might you want to make to the controller?

The controller is exhibiting "Integral Windup". It could be solved by placing a limit on the integral term in the u equation.
