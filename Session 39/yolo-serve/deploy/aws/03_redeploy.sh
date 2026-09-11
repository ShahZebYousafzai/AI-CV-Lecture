#!/usr/bin/env bash
# Ship a code change to a running instance.
#
# Run on your machine after 01_push_to_ecr.sh has pushed a NEW tag.
# The instance pulls the new image and systemd swaps the container.
#
#   ./deploy/aws/03_redeploy.sh i-0abc123def456 1.0.1

set -euo pipefail

INSTANCE_ID="${1:?usage: 03_redeploy.sh <instance-id> <image-tag>}"
IMAGE_TAG="${2:?usage: 03_redeploy.sh <instance-id> <image-tag>}"
AWS_REGION="${AWS_REGION:-ap-south-1}"
REPO_NAME="${REPO_NAME:-yolo-serve}"

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
IMAGE_URI="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${REPO_NAME}:${IMAGE_TAG}"

# SSM Run Command instead of SSH: no key file, no open port 22, and the
# whole thing is logged in CloudTrail. Needs AmazonSSMManagedInstanceCore
# on the instance role.
echo "==> Updating ${INSTANCE_ID} to ${IMAGE_URI}"

COMMAND_ID="$(aws ssm send-command \
    --region "${AWS_REGION}" \
    --instance-ids "${INSTANCE_ID}" \
    --document-name "AWS-RunShellScript" \
    --comment "redeploy yolo-serve" \
    --parameters "commands=[
        \"sed -i 's|${REPO_NAME}:.*|${REPO_NAME}:${IMAGE_TAG}|g' /etc/systemd/system/yolo-serve.service\",
        \"systemctl daemon-reload\",
        \"systemctl restart yolo-serve\",
        \"sleep 45\",
        \"curl -fs http://localhost/health/ready\"
    ]" \
    --query "Command.CommandId" --output text)"

echo "==> Command ${COMMAND_ID}, waiting"
sleep 60

aws ssm get-command-invocation \
    --region "${AWS_REGION}" \
    --command-id "${COMMAND_ID}" \
    --instance-id "${INSTANCE_ID}" \
    --query "{Status:Status,Output:StandardOutputContent,Error:StandardErrorContent}" \
    --output json
