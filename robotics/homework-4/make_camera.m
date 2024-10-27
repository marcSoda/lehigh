function camera  = make_camera(x, y, max_range)

camera.x = x;
camera.y = y;
camera.range = max_range;
camera.line = [];
camera.fig_coords = [-1 0 1 ; -1 3 -1];
camera.h = patch(camera.fig_coords(1,:)+x, camera.fig_coords(2,:)+y,'FaceColor', [.5 .5 .5]);
