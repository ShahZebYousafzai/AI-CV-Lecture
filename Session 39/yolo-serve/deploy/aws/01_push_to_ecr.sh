#!/usr/bin/env bash
# Build the image and push it to Amazon ECR.
#
# Run this on your own machine. It assumes `aws configure` has already been
# done with a user or role that can create and push to an ECR repository.
#
#   ./deploy/aws/01_push_to_ecr.sh
#
# Everything below is idempotent: run it again after a code change and it
# rebuilds and pushes a new tag.

set -euo pipefail

# ---- settings you may want to change --------------------------------------
AWS_REGION="${AWS_REGION:-ap-south-1}"
REPO_NAME="${REPO_NAME:-yolo-serve}"
IMAGE_TAG="${IMAGE_TAG:-1.0.0}"
# ---------------------------------------------------------------------------

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
REGISTRY="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
IMAGE_URI="${REGISTRY}/${REPO_NAME}:${IMAGE_TAG}"

echo "==> Account ${ACCOUNT_ID}, region ${AWS_REGION}"

# 1. Create the repository if it does not exist yet.
#    Scan on push is free and tells you about known CVEs in your base image.
if ! aws ecr describe-repositories --repository-names "${REPO_NAME}" \
        --region "${AWS_REGION}" >/dev/null 2>&1; then
    echo "==> Creating ECR repository ${REPO_NAME}"
    aws ecr create-repository \
        --repository-name "${REPO_NAME}" \
        --region "${AWS_REGION}" \
        --image-scanning-configuration scanOnPush=true \
        --image-tag-mutability IMMUTABLE >/dev/null
fi

# 2. Log the local Docker daemon into ECR. The token lasts 12 hours.
echo "==> Logging Docker into ECR"
aws ecr get-login-password --region "${AWS_REGION}" \
    | docker login --username AWS --password-stdin "${REGISTRY}"

# 3. Build.
#    --platform matters: an Apple Silicon Mac builds arm64 by default, and an
#    arm64 image will not start on an x86 EC2 instance. It fails with
#    "exec format error", which is a confusing way to learn about CPU
#    architectures at 2am.
echo "==> Building ${IMAGE_URI}"
docker build --platform linux/amd64 -t "${REPO_NAME}:${IMAGE_TAG}" .

# 4. Tag and push.
docker tag "${REPO_NAME}:${IMAGE_TAG}" "${IMAGE_URI}"
echo "==> Pushing ${IMAGE_URI}"
docker push "${IMAGE_URI}"

echo
echo "Pushed: ${IMAGE_URI}"
echo "Use that URI in deploy/aws/02_ec2_user_data.sh"
