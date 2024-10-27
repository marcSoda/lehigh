function [ cam, bearing ] = test_camera( cam, robot )

bearing = [];
try, delete(cam.line); end;

dist = sqrt((cam.x-robot.x).^2+(cam.y-robot.y).^2);

if dist > cam.range, return; end;

cam.line = plot([cam.x robot.x], [cam.y robot.y]);
bearing = atan2((cam.y - robot.y), (cam.x - robot.x))-robot.theta;
