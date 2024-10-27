function hwk4_skeleton
global V W DT_ODOM SIGMA_BEARING Q;

V=1;          % Robot velocity in m/s
W=pi/8;       % Angular velocity bound in radians/s
DT_ODOM = 1;  % Update rate for odometry (10 Hz)
SIGMA_BEARING = 3*pi/180;   % Std dev for bearing estimates in radians
MAX_RANGE = 15; %max range of cameras

close all; figure; axis equal; axis([0 120 0 80]); hold on;

x0 = 15;  y0 = 15;  theta0 = 0;

robot = make_robot( x0, y0, theta0, 'size', 1.5, 'color', 'g', 'make_trail', 1 );
robot_hat = make_robot( x0+10*randn, y0+10*randn, theta0+pi/8*randn, 'size', 1.5, 'color', 'r', 'make_trail', 1 );

rectangle( 'position', [15 15 90 50],'linestyle',':', 'edgecolor', 'k' );

p_cov = plot(0, 0);
P = [10 0 0; 0 10 0; 0 0 (pi/4)^2];

%Make Cameras
num_cameras = 0;
button = 1;
while 1
    [ x, y, button ] = ginput(1);
    if button==3, break; end
    num_cameras = num_cameras+1;
    camera(num_cameras) = make_camera(x, y, MAX_RANGE);
end

%Begin Simulation
v = V;
w = 0;
current_leg=1;
while current_leg~=9
     %MeasurementUpdate for every camer in range
     for i=1:num_cameras
        [ camera(i), bearing ] = test_camera( camera(i), robot );
         if ~isempty( bearing )
             [ robot_hat, P ] = MeasurementUpdate( robot_hat, P, camera(i), bearing );
        end
    end

    for j=1:1/DT_ODOM
        [ v, w, current_leg ] = checkPos(robot, v, w, current_leg);
        [ robot, robot_hat, P ] = TimeUpdate( robot, robot_hat, P, v, w );
        cove = make_cove(P, [robot_hat.x robot_hat.y]);
        set(p_cov, 'xdata', cove(1,:), 'ydata', cove(2,:));
        drawnow;
    end
end

function [ robot, robot_hat, P ] = TimeUpdate( robot, robot_hat, P, v, w )
global DT_ODOM;
    dt = DT_ODOM;
    rx = robot.x;
    ry = robot.y;
    rth = robot.theta;
    hx = robot_hat.x;
    hy = robot_hat.y;
    hth = robot_hat.theta;

    M = [ .02 0; 0 pi/128]; %noise multipliers

    u_t = [v+M(1,1)*randn(1) w + M(1,1)*randn(1)]; %v and w with random noise

    if w == 0, w = .00001; end %avoid divide by 0 errors

    %Update robot
    ru = [(rx+((-v/w)*sin(rth)+(v/w)*sin(rth+w*dt)));
          (ry +((v/w)*cos(rth)-(v/w)*cos(rth+w*dt)));
          (rth+w*dt)+M(2,2)*randn(1)];

    %Update robot_hat
    g = [ (hx+((-u_t(1)/u_t(2))*sin(hth)+(u_t(1)/u_t(2))*sin(hth+u_t(2)*dt)));
          (hy +((u_t(1)/u_t(2))*cos(hth)-(u_t(1)/u_t(2))*cos(hth+u_t(2)*dt)));
          (hth+u_t(2)*dt)];

    %Update covariance matrix:
    G = [ 1 0 (v/w)*(cos(w*dt + rth)-cos(rth));
          0 1 (v/w)*(sin(w*dt+rth)-sin(rth));
          0 0 1 ];

    V = [ ((-sin(rth)+sin(rth+w*dt))/w) ((v*(sin(rth)-sin(rth+w*dt)))/w^2)+((v*cos(rth+w*dt)*dt)/w);
         ((cos(rth)-cos(rth+w*dt))/w) (-(v*(cos(rth)-cos(rth+w*dt)))/w^2)+((v*sin(rth+w*dt)*dt)/w);
         0 dt ];

    P = G*P*G'+ V*M*V';

    %Move robots
    robot = move_robot(robot, ru(1), ru(2), ru(3));
    robot_hat = move_robot( robot_hat, g(1), g(2), g(3) );

function [ robot_hat, P ] = MeasurementUpdate( robot_hat, P, camera, z )
  global SIGMA_BEARING;
  hx = robot_hat.x;
  hy = robot_hat.y;
  hth = robot_hat.theta;
  cx = camera.x;
  cy = camera.y;

  mu = [ hx hy hth ]';
  q = (cx-hx)^2 + (cy-hy)^2;        %Book Step 11
  z_hat = atan2(cy-hy, cx-hx)-hth;  %Step 12
  H = [(cy-hy)/q -(cx-hx)/q -1];    %Step 13
  S = H*P*H' + SIGMA_BEARING;       %Step 14
  K = P*H'/S;                       %Step 17
  mu = mu + K*(z-z_hat);            %Step 18
  P = (eye(3)-K*H)*P;               %Step 19

  robot_hat.x = mu(1);
  robot_hat.y = mu(2);
  robot_hat.theta = mu(3);

%This could probably be more concise, but it works.
function [ v, w, current_leg ] = checkPos( robot, v, w, current_leg )
global V W
    if current_leg == 1
        if robot.x >= 105
            v = 0;
            w = W;
            current_leg++;
        end
    elseif current_leg == 2
        if robot.theta >= pi/2
            v = V;
            w = 0;
            current_leg++;
        end
    elseif current_leg == 3
        if robot.y >= 65
            v = 0;
            w = W;
            current_leg++;
        end
    elseif current_leg == 4
        if robot.theta >= pi
            v = V;
            w = 0;
            current_leg++;
        end
    elseif current_leg == 5
        if robot.x <= 15
            v = 0;
            w = W;
            current_leg++;
        end
    elseif current_leg == 6
        if robot.theta >= (3*pi)/2
            v = V;
            w = 0;
            current_leg++;
        end
    elseif current_leg == 7
        if robot.y <= 15
            v = 0;
            w = W;
            current_leg++;
        end
    elseif current_leg == 8
        if robot.theta >= 0
            v = 0;
            w = 0;
            current_leg++;
        end
    end

%The following functions were taken directly from recitation 6:
function p = make_cove(P, pos)
  p = [];
  N = 50;
  inc = 2*pi/N;
  phi = 0:inc:2*pi;
  circ = 2*[cos(phi); sin(phi)];
  p = zeros(2, N+2);
  ctr = 1;
  ii = ctr:(ctr+N+1);
  p(:,ii) = make_ellipse(pos(1:2), P(1:2,1:2), circ);
  ctr = ctr+N+2;

function p = make_ellipse(x, P, circ)
  r = sqrtm_2by2(P);
  a = r*circ;
  p(2,:) = [a(2,:)+x(2) NaN];
  p(1,:) = [a(1,:)+x(1) NaN];

function X = sqrtm_2by2(A)
  [Q, T] = schur(A);
  R = zeros(2);
  R(1,1) = sqrt(T(1,1));
  R(2,2) = sqrt(T(2,2));
  R(1,2) = T(1,2) / (R(1,1) + R(2,2));
  X = Q* R*Q';
