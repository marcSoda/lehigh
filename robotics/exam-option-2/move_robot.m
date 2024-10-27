function robot = move_robot(robot, x, y, th)

delete(robot.h)
robot = make_robot(x, y, th, robot.w, robot.size, robot.color, robot.trail);
