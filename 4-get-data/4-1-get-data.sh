#!/bin/bash

. ../.env

echo ""
echo "Generating pod manifest ..."
cat efs-data-copy.yaml.template | envsubst > efs-data-copy.yaml

echo ""
echo "Creating efs-data-prep pod ..."
kubectl apply -f efs-data-copy.yaml
sleep 3
kubectl get pods | grep data-prep

