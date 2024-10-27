function camera  = make_camera(x, y, max_range)

camera.x = x;
camera.y = y;
camera.range = max_range;
camera.line = [];
fig_coords = [-1 0 1 ; -1 3 -1];
camera.h = patch(fig_coords(1,:)+x, fig_coords(2,:)+y,'FaceColor', [0 135/255 215/255]);
