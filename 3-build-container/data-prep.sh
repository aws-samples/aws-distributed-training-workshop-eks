#!/bin/bash

echo ""
echo "Shared path - ${1}"

mkdir -p ${1}
cd ${1}

echo ""
echo "Downloading data ..."
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xzf cifar-10-python.tar.gz
rm cifar-10-python.tar.gz

echo ""
echo "Setting permissions ..."
chown -R 1000:100 ${1}

echo ""
echo "Done"
echo ""

