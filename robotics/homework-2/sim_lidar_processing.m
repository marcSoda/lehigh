function [x y] = sim_lidar_processing( ranges )
    [num_rows, num_cols] = size( ranges );
    t = [1:num_rows]';
    x = max(ranges.*cos( repmat(t, 1, num_cols)*pi/180),0)
    y = max(ranges.*sin( repmat(t, 1, num_cols)*pi/180),0)
