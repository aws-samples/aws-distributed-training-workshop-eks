#!/bin/bash

# Deploy Kuberntes Packages

# Metrics server
echo ""
echo "Deploying Kubernetes Metrics Server ..."
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

# Torch Elastic
echo ""
echo "Deploying Torch Elastic ..."
kubectl apply -k torch-elastic/default

# Etcd
echo ""
echo "Deploying etcd ..."
kubectl apply -f torch-elastic/etcd.yaml

