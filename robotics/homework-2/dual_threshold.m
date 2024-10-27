function y = dual_threshold( x, x_min, x_max )
  x(x >= x_min & x <= x_max)  = 1
