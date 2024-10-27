# CSE411 - Advanced Programming Techniques

# Homework 3

**Due Date: 10/24/2023 EOD**

## Learning Outcomes:

- Use different parallel programming techniques to make correct code run faster
- Determine the effect of the number of threads on the performance of a parallel program

## Description

The Gaussian Elimination algorithm is a method for solving linear systems of equations of the form A.x = B, where A is the coefficient matrix (NxN) and B is the column vector (N). x is the variable vector (N) to be determined by the algorithm.

You are provided with a sequential Rust implementation of the algorithm that has a time complexity O(n3). 
We do not want to reduce the complexity of the algorithm, but we want you to explore different parallel implementations.

Measure the performance of each parallel implementation for different matrix sizes and number of threads.

## Implementation

You are provided with code for a sequential Gauss function. You need to create new versions of the function `gauss()` for each parallel implementation. It is recommended to start with simple parallel techniques and create layers of parallelism on top of each other.

Measure the execution time (using cargo bench) of each suggested parallel implementation for different sizes of the matrix A (256, 512, 1024, 2048, 4096).

Measure the execution time of each implementation for different numbers of threads as well.

## Report

Write a short report (3-4 pages) to explain each parallel technique you implemented. Discuss the results obtained for each implementation and the effect of the number of threads on the performance.  You may use graphs to show the speedups as a function of the size of the matrix and the performance as a function of the number of threads.

## Grade distribution

	- Code of the parallel implementations  (4 at least) 50%
        - Benchmarks to show performance varying threads and size
        - Tests to show correctness versus single threaded implementation
	- Report 50%
        - Background on Gauss
        - Discussion on theory of parallel techniques implemented
        - Implementation and test setup
        - Results
        - Discussion on results and conclusions
