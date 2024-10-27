# CSE298 - Foundations of Robotics - Homework 2

**Due Date: 7/19/2021 EOD**

## Objectives:

The purpose of this assignment is to provide you with exercises on writing efficient, matrix based Matlab code.

## The Assignment

*Note to evaluate the performance of some of these functions, I have supplied a `benchmark()` function which will allow you to compare the execution time of the original function compared to your own implementation.*

1. Rewrite the function `slow_norm` without using a for loop. Benchmark the performance with your new function compared to `slow_norm`, as well as with the built-in Matlab norm function (which you may not use). Commit your progress to Gitlab when you are done.
2. Repeat this exercise for the `non_negative_filter` function. Commit your progress when you are done.
3. Repeat this exercise for `dual_threshold` except you do not need to benchmark the two functions. Commit your progress when you are done.
4. You can profile your code in Matlab to identify bottlenecks. The process for doing this is:

```
> profile on;
> <Run your function here>
> profile off;
> profile viewer;
```

The profile viewer provides a detailed breakdown of where your function spent its execution time.Generate a large random array of ranges (e.g., 181x1000) in the workspace. See the `rand` function and/or the `benchmark` function provided for how this can be done. Then turn on the profiler and call the `sim_lidar_processing` function

```
> [x,y] = sim_lidar_processing( ranges );
```

Where ranges is the array that you created. Examine the profiler results.

Now modify the function `sim_lidar_processing` replacing `cosd` and `sind` with `cos` and `sin` functions which take radians as arguments. Re-profile the code. Did anything change? If so, what? Commit your progress when you are done.

*Q4 Answer: This is already done in the provided code. However, I replaced cos and sin with cosd and sind. I observed that cosd and sind take significantly more (around 70x) time than cos and sin. Interestingly, the calls to pi take around double the time with cos and sin than cosd and sind even though it is called the same number of times. I'm not sure why.


5. Rewrite the function sim_lidar_processing without using a for loop. Note that the current implementation uses a pair of nested for loops. You can easily get rid of both of these. Hint: look at the repmat command. Commit your progress when you are done.

## Deliverables

Submit all of your code via Gitlab. A brief write-up addressing question 4 is also required. You must have at least 1 commit per step in the assignment (5 commits total at least).
