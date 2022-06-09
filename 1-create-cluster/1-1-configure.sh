#!/bin/bash

. ../.env

echo ""
echo "Configuring AWS client ..."
aws configure --profile $AWS_PROFILE

echo ""
echo "Generating cluster configuration eks.yaml ..."
cat eks.yaml.template | envsubst > eks.yaml

