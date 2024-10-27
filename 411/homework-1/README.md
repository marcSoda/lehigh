# CSE 411 - Advanced Programming Techniques - Fall 2023

# Homework 1

⏰ **Due Date: 9/14/2023 EOD**

## Instructions 

**Read thoroughly before starting your project:**

1. Fork this repository into your CSE411 project namespace. [Instructions](https://docs.gitlab.com/ee/workflow/forking_workflow.html#creating-a-fork)
2. Clone your newly forked repository onto your development machine. [Instructions](https://docs.gitlab.com/ee/gitlab-basics/start-using-git.html#clone-a-repository) 
3. As you are writing code you should commit patches along the way. *i.e.* don't just submit all your code in one big commit when you're all done. Commit your progress as you work. 

**💥IMPORTANT: You must commit frequently as you work on your project.**
 
## Project Learning Outcomes:

- Implement an integer set using four different implementations
- Measure the performance of the set operations for each  implementation
- Compare the four implementations of the set
- Write a report to describe the implementation and discuss the results

## Project Description

A set is a collection of unique keys (integers) that can be manipulated using three basic operations below:

- *insert* - returns false if the key is already in the set, otherwise it inserts the key and returns true.
- *remove* - returns false if the key is not in the set, otherwise it removes the key and returns true.
- *find* - returns true if the key is in the set or false otherwise, without changing the set.

## Implementation

You will implement four different implementations of the set interface as listed below:

- *ArraySet* implements the set using a dynamic array
- *ListSet* implements the set using a linked list
- *TreeSet* implements the set using a balanced binary search tree
- *HashSet* implements the set using a hash table

You should not implement  the data structures listed above from scratch. Use the data structures already available in the standard template library.

### Program interface

Your program should be invoked with four command-line arguments as listed below:

- i operations (integer)
- k max_key_value (int) (minimum is 0)
- d data_structure (string) (array, list, tree, or hashtable)
- r read-only_ratio (int) (percentage of find operations, the remaining will be equally split between insert and remove)

## Evaluation

### Experiment setup

 Your program should be set up to:

- Pre-initialize the set to be 50% full. Keys should be unique and randomly selected. 
- Measure the execution time of each operation using Rust's `bench` command.
- The number of operations should be big enough to be valid.
- Each experiment should perform #operations, randomly chosen among insert, find, and remove, each using a random key in the range [0, max_key_value-1].

### Benchmark results

- Set up a CI pipline that builds, tests, and benchmarks your code.

- Benchmark the results for different sizes (1K, 10K, 100K, 1M) of each set implemenation and different read-only ratios (0%, 20%, 50%, 80%, 100%). Present the results using clear plots.

## Final report

Write a report that should include:

- Brief summary of each implementation of the set (not the code). Describe what distinguishes each implementation.
- Testing setup (fixed parameters(operations and runs) and tested configurations (set sizes and percentage of finds)).
- Results for each implementation and configuration using plots.
- Discussion of the results using your own words.

Do not include your source code in the report.

## Grading

- Code - 50%
- Writeup - 50%

- Only files under version control in your forked assignment repository will be graded. Local files left untracked on your computer will not be considered.

- Only code committed *and pushed* prior to the time of grading will be accepted. Locally committed but unpushed code will not be considered.

- Your assignment will be graded according to the [Programming Assignment Grading Rubric](https://drive.google.com/open?id=1V0nBt3Rz6uFMZ9mIaFioLF-48DFX0VdkbgRUDM_eIFk).

