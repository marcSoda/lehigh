function delta_t = benchmark( fun_handle, num_el, num_times )
%=========================================================================
% function delta_t = benchmark( fun_handle, num_el, num_times )
% - Takes a function handle, the number of elements of the test vector, and
% the number of times to execute the function as arguments.
% - Returns the time "delta_t" required to execute the function "num_times" 
% - Sample usage:  This call will execute function "my_fun" 10 times with
%   random vector argument 100 elements long
%   >> dt = benchmark( @my_fun, 100, 10 );
%=========================================================================

% This creates an array "x" that has "num_el" rows and "num_times" columns 
% and fills it with randomly generated numbers between 0 and 1  
x = rand(181,num_el );

% Tic and toc enable you to measure the elapsed time for a block of code. 
tic;

% Executes the function num_times
for i=1:num_times
    % This passes the ith column of x to the function
    y = fun_handle( x );
end

delta_t = toc;
