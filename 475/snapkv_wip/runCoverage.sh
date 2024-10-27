#!/bin/bash

cd build

for FILE in bin/cov*; do
    ./"$FILE"
done

cd ..

gcovr -r .
