#!/bin/bash

. ../.env

# Create registry if needed
REGISTRY_COUNT=$(aws ecr describe-repositories | grep ${IMAGE} | wc -l)
if [ "$REGISTRY_COUNT" == "0" ]; then
	aws ecr create-repository --repository-name ${IMAGE}
fi

# Login to container registry
echo "Logging in to $REGISTRY ..."
aws ecr get-login-password | docker login --username AWS --password-stdin $REGISTRY

# Push image to registry
echo "Pushing ${IMAGE}${TAG} to registry ..."
docker push ${REGISTRY}${IMAGE}${TAG}

