#!/bin/bash

echo ""
echo "Describing data prep pod ..."
kubectl describe pod efs-data-prep-pod

echo ""
echo "Showing status of data prep pod ..."
kubectl get pods | grep data-prep

