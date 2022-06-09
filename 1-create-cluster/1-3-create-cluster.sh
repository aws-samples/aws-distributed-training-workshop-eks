#!/bin/bash

. ../.env

echo ""
echo "Creating EKS cluster ..."
echo ""
echo "... using configuration from ./eks.yaml ..."
echo ""
cat ./eks.yaml
echo ""
date
CMD="eksctl create cluster -f ./eks.yaml"
echo "${CMD}"
${CMD}
echo ""
date
echo "Done creating EKS cluster"

echo ""
echo "Updating kubeconfig ..."
aws eks update-kubeconfig --name $CLUSTER_NAME
echo ""

echo ""
echo "Displaying cluster nodes ..."
kubectl get nodes

