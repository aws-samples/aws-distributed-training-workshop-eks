#!/bin/bash

. ../.env

echo ""
echo "Generating test job manifest ..."
cat test.yaml.template | envsubst > test.yaml
cat test.yaml
echo ""
