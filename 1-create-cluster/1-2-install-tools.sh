#!/bin/bash

# Install tools

# eksctl
echo ""
echo "Installing eksctl ..."
curl --location "https://github.com/weaveworks/eksctl/releases/download/v0.66.0/eksctl_$(uname -s)_amd64.tar.gz" | tar xz -C /tmp
sudo mv /tmp/eksctl /usr/local/bin
eksctl version

# kubectl
echo ""
echo "Installing kubectl ..."
curl -o kubectl https://amazon-eks.s3.us-west-2.amazonaws.com/1.19.6/2021-01-05/bin/linux/amd64/kubectl
chmod +x ./kubectl
sudo mv ./kubectl /usr/local/bin
kubectl version --client

# kubectx
echo ""
echo "Installing kubectx ..."
pushd /tmp
git clone https://github.com/ahmetb/kubectx
sudo mv kubectx /opt
sudo ln -s /opt/kubectx/kubectx /usr/local/bin/kubectx
sudo ln -s /opt/kubectx/kubens /usr/local/bin/kubens
popd

# kubetail
echo ""
echo "Installing kubetail ..."
curl -o /tmp/kubetail https://raw.githubusercontent.com/johanhaleby/kubetail/master/kubetail
chmod +x /tmp/kubetail
sudo mv /tmp/kubetail /usr/local/bin/kubetail

# kubeshell
echo ""
echo "Installing kubeshell ..."
curl -LO https://github.com/kvaps/kubectl-node-shell/raw/master/kubectl-node_shell
chmod +x ./kubectl-node_shell
sudo mv ./kubectl-node_shell /usr/local/bin/kubectl-node_shell

# jq
echo ""
echo "Installing jq ..."
sudo yum install -y jq

# yq
echo ""
echo "Installing yq ..."
pip3 install yq

# Set up aliases
echo ""
echo "Setting up aliases ..."
cat << EOF >> ~/.bashrc
alias ll='ls -alh --color=auto'
alias k='kubectl'
alias kc='kubectx'
alias kn='kubens'
alias kt='kubetail'
alias ks='kubectl node-shell'
EOF

echo ""
echo "Done setting up tools."
echo ""


