#!/bin/bash

echo ""
echo "Showing test job status ..."
kubectl get pods | grep test

