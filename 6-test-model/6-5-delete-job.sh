#!/bin/bash

echo ""
echo "Deleting test job ..."
kubectl delete -f ./test.yaml
