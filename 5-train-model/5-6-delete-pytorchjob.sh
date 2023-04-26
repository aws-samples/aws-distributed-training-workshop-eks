#!/bin/bash

echo ""
echo "Deleting PyTorchJob ..."
kubectl delete -f ./train.yaml

echo ""
echo "Restarting etcd ..."
kubectl delete pod $(kubectl get pods | grep etcd | cut -d ' ' -f 1)

echo ""
echo "Cleaning up model checkpoint ..."
echo ""
kubectl apply -f ./cleanup.yaml
echo ""
while true; do
	JOB="$(kubectl get job | grep cleanup)"
	COMPLETED=$(echo $JOB | awk -e '{print $2}' | cut -d '/' -f 1)
	if [ "$COMPLETED" == "1" ]; then
		kubectl logs $(kubectl get pods | grep cleanup | cut -d ' ' -f 1)
		break;
	else
		echo "$JOB"
		sleep 1
	fi
done
echo ""
kubectl delete -f ./cleanup.yaml
echo ""

