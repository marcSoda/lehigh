# Homework 5 - Functional Programming

Due: **12/8/23 EOD**

```rust
let data = vec![
    ("Alice", 50),
    ("Bob", 80),
    ("Charlie", 60),
    ("David", 75),
    ("Eve", 90),
    ("Frank", 55),
    ("Grace", 45),
    ("Hannah", 85),
    ("Ivy", 95),
    ("Jack", 40)
];
```

1. Filter and Map: From the dataset, create a list of names who have a score greater than 70.
2. Enumerate and For_Each: Print each record with an index.
3. Zip and Filter_Map: Assuming you have another vector remarks parallel to data, zip these two and create a list of remarks for students with scores more than 75.
```rust
let remarks = vec!["Good", "Excellent", "Average", "Good", "Outstanding", "Fair", "Poor", "Very Good", "Exceptional", "Poor"];
```
4. Chain and Count: Chain the data with another vector of records and count the total number of records.

```rust
let additional_data = vec![
    ("Kyle", 65),
    ("Liam", 70)
];
```

5. Flat_Map and Take: Consider data as a vector of vectors. Flatten this structure and take the first 5 elements.
```rust
let nested_data = vec![vec![data[0], data[1]], vec![data[2], data[3]], vec![data[4], data[5]]];
```

6. Cycle and Take: Create an infinite iterator cycling through data and take the first 15 elements.

7. All and Any: Check if all records have a score above 40 and if any record has a score above 90.

## Additional Problems

```rust
let activities = vec![
    vec!["Read", "Write", "Code"],
    vec!["Draw", "Paint"],
    vec!["Code", "Debug"],
    vec!["Paint", "Sculpt"],
    vec!["Read", "Research"],
    vec!["Write", "Blog"],
    vec!["Code", "Review"],
    vec!["Draw", "Design"],
    vec!["Research", "Experiment"],
    vec!["Sculpt", "Model"],
];
```

8. Create a list of names of students whose score is above 60 and also list their activities.

9. Generate a list of unique activities of students who scored more than 70.

10. Pair each student with their corresponding activity, but only include students up until (and including) the first one who scored less than 50.

11. Create an iterator that cycles through the data but skips all records until a student with a score less than 60 is found, then fuses the iterator after the first student with a score over 90.

12. Chain the data and activities vectors and create a HashMap where the key is the student's ID and the value is a tuple of their name, score, and activities.

13. Create a list of names of students who are involved in 'Code' activity and have a score above 65, along with their score.
