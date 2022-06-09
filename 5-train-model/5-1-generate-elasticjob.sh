#!/bin/bash

. ../.env

echo ""
echo "Generating ElasticJob manifest ..."
cat train.yaml.template | envsubst > train.yaml
echo ""
echo "Generating Checkpoint Cleanup job ..."
cat cleanup.yaml.template | envsubst > cleanup.yaml
echo ""
echo "ElasticJob Manifest:"
echo ""
cat train.yaml
echo ""

