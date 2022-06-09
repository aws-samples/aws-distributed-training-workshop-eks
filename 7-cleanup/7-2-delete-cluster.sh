#!/bin/bash

. ../.env

echo ""
read -p "Deleting cluster $CLUSTER_NAME. Proceed? [Y/n]: " PROCEED
if [ "$PROCEED" == "Y" ]; then
        echo "Confirmed ..."
        eksctl delete cluster -f ../1-create-cluster/eks.yaml
        echo "Please note that the cluster will be fully deleted when the Cloud Formation stack completes its removal"
        echo "Only after the process in Cloud Formation is finished, you will be able to create a new cluster with the same name"
elif [ "$PROCEED" == "n" ]; then
        echo "Cancelling. Cluster will not be deleted."
else
        echo "$PROCEED is not a valid response"
        echo "Please run the script again and choose Y or n (case sensitive)"
fi
echo ""

