function robot  = make_robot(x, y, theta, varargin)
%MAKE_ROBOT creates a robot
%   MAKE_ROBOT( x, y, theta ) creates a robot at the given pose
%
%   MAKE_ROBOT( x, y, theta, 'Property1', 'PropertyValue1', ... )
%   sets the corresponding property values of the robot.  These include:
%   - 'SIZE' which is the length of the robot in axis units (default is 1)
%   - 'COLOR', which is the robot's color (default is red)
%   - 'MAKE_TRAIL' is whether to draw the robot's path (default is false)
%

% Default values
robot.x = x;
robot.y = y;
robot.theta = theta;
robot.size = 1;
robot.trail = false;
robot.color = 'r';

if nargin>3
    for i=4:2:nargin
        prop = varargin{i-3};
        val = varargin{i-2};
        if strcmpi( prop, 'size' ), robot.size = val;
        elseif strcmpi( prop, 'make_trail' ), robot.trail = val;
        elseif strcmpi( prop, 'color' ), robot.color = val;
        else error('Unrecognized argument for make_robot'); end
    end
end

s = robot.size;
fig_coords = [-s s 2*s s -s; -s -s 0 s s];
w1 = [-0.8*s -0.1*s -0.1*s -0.8*s; -s -s -1.25*s -1.25*s];
w2 = [-0.8*s -0.1*s -0.1*s -0.8*s; s s 1.25*s 1.25*s];
w3 = [0.1*s 0.8*s 0.8*s 0.1*s; -s -s -1.25*s -1.25*s];
w4 = [0.1*s 0.8*s 0.8*s 0.1*s; s s 1.25*s 1.25*s];
R = [cos(theta) -sin(theta); sin(theta) cos(theta)];

robot.fig_coords = R*fig_coords;
robot.w1 = R*w1;
robot.w2 = R*w2;
robot.w3 = R*w3;
robot.w4 = R*w4;

robot.h = zeros(6,1);
robot.h(1) = patch(robot.fig_coords(1,:)+x, robot.fig_coords(2,:)+y, robot.color);
robot.h(2) = patch(robot.w1(1,:)+robot.x, robot.w1(2,:)+robot.y, 'k');
robot.h(3) = patch(robot.w2(1,:)+robot.x, robot.w2(2,:)+robot.y, 'k');
robot.h(4) = patch(robot.w3(1,:)+robot.x, robot.w3(2,:)+robot.y, 'k');
robot.h(5) = patch(robot.w4(1,:)+robot.x, robot.w4(2,:)+robot.y, 'k');
robot.h(6) = plot(x,y,'k*');

% If trail is on, plot the point
if robot.trail
    plot( robot.x, robot.y, strcat(robot.color,'.') );
end
