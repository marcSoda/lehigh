function robot  = make_robot(x, y, th, w, size, color, trail)

% Default values
robot.x = x;
robot.y = y;
robot.th = th;
robot.w = w;
robot.size = size;
robot.color = color;
robot.trail = trail;

fig_coords = [-robot.size/2 -robot.size/2 robot.size ; -robot.size/2 robot.size/2 0];
R = [cos(th) -sin(th); sin(th) cos(th)];
fig_coords = R*fig_coords;
robot.h = patch(fig_coords(1,:)+x, fig_coords(2,:)+y, robot.color);
if robot.trail, plot(robot.x, robot.y, strcat(robot.color,'.')); end
