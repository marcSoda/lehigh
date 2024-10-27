function particle_filter
global V W DT VAR;

V=8;            %linear velocity
W=pi/2;         %angular velocity
DT = .2;        %rate
MAX_RANGE = 20; %max range of cameras
N = 40;         %number of particles
VAR = [.2 pi/64 .05 pi/128]; %variances
%      xy th     v   w

close all; figure; axis equal; axis([0 120 0 80]); hold on;

x0 = 15;  y0 = 15;  th0 = 0;

robot = make_robot( x0, y0, th0, 0, 3, 'g', true);     %actual robot
robot_hat = make_robot( x0, y0, th0, 0, 2, 'b', true); %hypothetical robot

rectangle('position', [15 15 90 50],'linestyle',':', 'edgecolor', 'k');

%Initial sample
for i = 1:N
    px = x0+VAR(1)*randn;
    py = y0+VAR(1)*randn;
    pth = th0+VAR(1)*randn;
    P(i) = make_robot(px, py, pth, 0, 1, 'r', false);
end

%Make Cameras
num_cameras = 0;
button = 1;
while 1
    [ x, y, button ] = ginput(1);
    if button==3, break; end
    num_cameras = num_cameras+1;
    cameras(num_cameras) = make_camera(x, y, MAX_RANGE);
end

%Begin Simulation
v = V;
w = 0;
current_leg=1;
while current_leg~=9
    %Move robot and particles 1/DT times per iteration
    for i=1:1/DT
        [v, w, current_leg] = checkPos(robot, v, w, current_leg);
        [robot, robot_hat, P] = move(robot, robot_hat, P, v, w);
    end
    %Update and resample once per camera per iteration
    for i = 1:num_cameras
        [cameras(i), z] = test_camera(cameras(i), robot);
        if isempty(z), continue; end
        P = update(P, cameras(i), z);
        P = resample(P);
    end
end

function [robot, robot_hat, P] = move(robot, robot_hat, P, v, w)
global DT VAR;
    dt = DT;
    rx = robot.x;
    ry = robot.y;
    rth = robot.th;

    if w == 0, w = .00001; end %avoid divide by 0 errors

    %Move particles
    for i = 1:length(P)
        p = P(i);
        u_v = v+VAR(3)*randn;  %variance in linear velocity
        u_w =  w+VAR(4)*randn; %variance in angular velocity

        px = (p.x+((-u_v/u_w)*sin(p.th)+(u_v/u_w)*sin(p.th+u_w*dt))); %x update
        py = (p.y+((u_v/u_w)*cos(p.th)-(u_v/u_w)*cos(p.th+u_w*dt)));  %y update
        pth = (p.th+u_w*dt);                                          %th update
        P(i) = move_robot(p, px+VAR(1)*randn, py+VAR(1)*randn, pth+VAR(2)*randn);
    end

    %move robot
    rx = (rx+((-v/w)*sin(rth)+(v/w)*sin(rth+w*dt))); %x update
    ry = (ry +((v/w)*cos(rth)-(v/w)*cos(rth+w*dt))); %y update
    rth = (rth+w*dt)+VAR(4)*randn;                   %th update

    %draw new robot
    robot = move_robot(robot, rx, ry, rth);
    %draw avg
    robot_hat = move_robot(robot_hat, mean([P(:).x]), mean([P(:).y]), mean([P(:).th]));
    drawnow; %refresh

function P = update(P, camera, z)
    %weigh particles
    for i = 1:length(P)
        pdist = sqrt((camera.x-P(i).x)^2 + (camera.y-P(i).y)^2);
        pbear = atan2(camera.y-P(i).y, camera.x-P(i).x)-P(i).th;
        w1 = (1/sqrt(2*pi*.5)) * exp(-(z(1)-pdist)^2/(2*.5));
        w2 = (1/sqrt(2*pi*.5)) * exp(-(z(2)-pbear)^2/(2*.5));
        P(i).w = w1*w2;
    end
    [P.w] = deal([P.w]./sum([P.w])); %standardize weights

function P = resample(P)
    for i = 1:length(P)
        p = P(find(rand <= cumsum([P.w]),1));
        P(i) = move_robot(P(i), p.x, p.y, p.th);
    end

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
        if robot.th >= pi/2
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
        if robot.th >= pi
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
        if robot.th >= (3*pi)/2
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
        if robot.th >= 0
            v = 0;
            w = 0;
            current_leg++;
        end
    end
