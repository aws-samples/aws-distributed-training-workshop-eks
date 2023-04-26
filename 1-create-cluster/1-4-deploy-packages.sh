#!/bin/bash

# Deploy Kuberntes Packages

# Metrics server
echo ""
echo "Deploying Kubernetes Metrics Server ..."
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

# Kubeflow Training Operator
echo ""
echo "Deploying Kubeflow Training Operator ..."
pushd ./kubeflow-training-operator
./deploy.sh
popd

# Etcd
echo ""
echo "Deploying etcd ..."
kubectl apply -f etcd/etcd-deployment.yaml

