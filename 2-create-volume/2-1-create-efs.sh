#!/bin/bash

. ../.env

# This script follows the following eks workshop
# https://www.eksworkshop.com/beginner/190_efs/launching-efs/

# Assume the cluster name is the first cluster in the list
echo ""
echo 'Cluster name: ' $CLUSTER_NAME
VPC_ID=$(aws eks describe-cluster --name $CLUSTER_NAME --query "cluster.resourcesVpcConfig.vpcId" --output text)
CIDR_BLOCK=$(aws ec2 describe-vpcs --vpc-ids $VPC_ID --query "Vpcs[].CidrBlock" --output text)

echo 'VPC: ' $VPC_ID
echo 'CIDR: ' $CIDR_BLOCK

echo ""
echo "Creating security group ..."
MOUNT_TARGET_GROUP_NAME="eks-efs-group-${CLUSTER_NAME}"
MOUNT_TARGET_GROUP_DESC="NFS access to EFS from EKS worker nodes"
aws ec2 create-security-group --group-name $MOUNT_TARGET_GROUP_NAME --description "$MOUNT_TARGET_GROUP_DESC" --vpc-id $VPC_ID
sleep 5

MOUNT_TARGET_GROUP_ID=$(aws ec2 describe-security-groups --filter Name=vpc-id,Values=$VPC_ID Name=group-name,Values=$MOUNT_TARGET_GROUP_NAME --query 'SecurityGroups[*].[GroupId]' --output text)
echo $MOUNT_TARGET_GROUP_NAME $MOUNT_TARGET_GROUP_DESC $MOUNT_TARGET_GROUP_ID

aws ec2 authorize-security-group-ingress --group-id $MOUNT_TARGET_GROUP_ID --protocol tcp --port 2049 --cidr $CIDR_BLOCK
sleep 2

echo ""
echo "Creating EFS volume ..."
FILE_SYSTEM_ID=$(aws efs create-file-system | jq --raw-output '.FileSystemId')
echo $FILE_SYSTEM_ID
sleep 10

TAG1=tag:alpha.eksctl.io/cluster-name
TAG2=tag:kubernetes.io/role/elb
SUBNETS=$(aws ec2 describe-subnets --filter Name=$TAG1,Values=$CLUSTER_NAME Name=$TAG2,Values=1 --query 'Subnets[*].SubnetId' --output text)
echo $SUBNETS

for subnet in ${SUBNETS}
do
    echo "Creating mount target in subnet " $subnet " , security group " $MOUNT_TARGET_GROUP_ID " ,for efs id " $FILE_SYSTEM_ID
    aws efs create-mount-target --file-system-id $FILE_SYSTEM_ID --subnet-id $subnet --security-groups $MOUNT_TARGET_GROUP_ID
    sleep 2
done
sleep 30

echo ""
echo "Mount points state ..."
aws efs describe-mount-targets --file-system-id $FILE_SYSTEM_ID | jq --raw-output '.MountTargets[].LifeCycleState'

echo ""
echo "Done."
echo ""
