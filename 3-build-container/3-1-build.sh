#!/bin/bash

. ../.env

# Build Docker image
docker image build -f Dockerfile-${PROCESSOR} -t ${REGISTRY}${IMAGE}${TAG} .

