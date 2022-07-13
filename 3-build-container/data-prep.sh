#!/bin/bash

echo "Shared path - ${1}"

mkdir -p ${1}
cd ${1}
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xzf cifar-10-python.tar.gz
rm cifar-10-python.tar.gz
chown -R 1000:100 ${1}

