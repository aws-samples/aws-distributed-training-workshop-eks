#!/bin/bash

. ../.env

# Build Docker image
docker image build -t ${REGISTRY}${IMAGE}${TAG} .

