#!/bin/bash
set -e

if command -v aws &> /dev/null; then
    echo "AWS CLI already installed, skipping."
    exit 0
fi

apt-get update -qq && apt-get install -y -qq unzip
curl -s https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip -o /tmp/awscliv2.zip
unzip -q /tmp/awscliv2.zip -d /tmp
/tmp/aws/install
rm -rf /tmp/aws /tmp/awscliv2.zip

if [ -n "$AWS_ACCESS_KEY_ID" ] && [ -n "$AWS_SECRET_ACCESS_KEY" ]; then
    aws configure set aws_access_key_id "$AWS_ACCESS_KEY_ID"
    aws configure set aws_secret_access_key "$AWS_SECRET_ACCESS_KEY"
    aws configure set default.region us-east-2
    echo "AWS credentials configured."
else
    echo "AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY not set — skipping aws configure."
fi
