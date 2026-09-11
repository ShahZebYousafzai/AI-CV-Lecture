#!/bin/bash
# EC2 user data. Paste this into "Advanced details -> User data" when
# launching the instance, or attach it with `aws ec2 run-instances
# --user-data file://02_ec2_user_data.sh`.
#
# User data runs ONCE, as root, on first boot. Anything here happens before
# you can SSH in, which is the point: the instance is disposable and its
# whole configuration lives in this file, in version control.
#
# Progress lands in /var/log/cloud-init-output.log on the instance.

set -euxo pipefail

# ---- fill these in --------------------------------------------------------
AWS_REGION="ap-south-1"
IMAGE_URI="<account-id>.dkr.ecr.ap-south-1.amazonaws.com/yolo-serve:1.0.0"
# ---------------------------------------------------------------------------

# 1. Docker. Amazon Linux 2023 ships it in the default repositories.
dnf update -y
dnf install -y docker
systemctl enable --now docker

# 2. Let the instance pull from ECR.
#    This works because the instance has an IAM role attached with the
#    AmazonEC2ContainerRegistryReadOnly policy. No keys on the box, ever:
#    the role hands out short lived credentials through the metadata service.
ACCOUNT_ID="$(echo "${IMAGE_URI}" | cut -d. -f1)"
aws ecr get-login-password --region "${AWS_REGION}" \
    | docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

# 3. Pull the image.
docker pull "${IMAGE_URI}"

# 4. Run it under systemd rather than a bare `docker run`.
#    A bare docker run dies with your SSH session and does not come back
#    after a reboot. systemd restarts it on failure and starts it at boot.
cat >/etc/systemd/system/yolo-serve.service <<EOF
[Unit]
Description=YOLOv8 detection API
After=docker.service
Requires=docker.service

[Service]
Restart=always
RestartSec=5
# Clean up any container left over from a previous run.
ExecStartPre=-/usr/bin/docker rm -f yolo-serve
ExecStartPre=/usr/bin/docker pull ${IMAGE_URI}
ExecStart=/usr/bin/docker run --rm --name yolo-serve \\
    -p 80:8000 \\
    -e APP_ENV=aws \\
    -e APP_LOG_LEVEL=INFO \\
    -e APP_TORCH_THREADS=2 \\
    -e APP_INFERENCE_CONCURRENCY=2 \\
    --memory 1800m \\
    --log-driver=awslogs \\
    --log-opt awslogs-region=${AWS_REGION} \\
    --log-opt awslogs-group=/yolo-serve \\
    --log-opt awslogs-create-group=true \\
    ${IMAGE_URI}
ExecStop=/usr/bin/docker stop yolo-serve

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now yolo-serve

# 5. Wait for the model to load, then prove the service answers.
for _ in $(seq 1 30); do
    if curl -fs http://localhost/health/ready; then
        echo "yolo-serve is ready"
        break
    fi
    sleep 5
done
