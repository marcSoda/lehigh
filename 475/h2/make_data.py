#!/usr/bin/env python3
import sys
import random
from sklearn.datasets import make_blobs
import numpy as np

# Get command line arguments
A = int(sys.argv[1])
B = int(sys.argv[2])
C = int(sys.argv[3])
D = int(sys.argv[4])
E = str(sys.argv[5])

data, centers = make_blobs(n_samples=A, centers=C, n_features=B)
# data = np.round(data).astype(int)
# data += abs(data.min()) + 1

# Save data points to file
with open(E, "w") as f:
    f.write(str(A) + " " + str(B) + " " + str(C) + " " + str(D) + " " + "\n")
    for i in range(A):
        datapoint = " ".join(str(x) for x in data[i])
        f.write(datapoint + "\n")
