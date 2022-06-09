#!/bin/bash

echo ""
echo "Showing cifar10-test log ..."
echo ""

kubectl logs -f $(kubectl get pods | grep cifar10-test | cut -d ' ' -f 1 | head -n 1)

