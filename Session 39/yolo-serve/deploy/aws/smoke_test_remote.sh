#!/usr/bin/env bash
# Prove the deployed service works from outside AWS.
#
#   ./deploy/aws/smoke_test_remote.sh 13.234.56.78
#
# Run this immediately after every deploy. "It came up" is not the same as
# "it works", and the difference is usually a security group rule.

set -euo pipefail

HOST="${1:?usage: smoke_test_remote.sh <public-ip-or-dns>}"
BASE="http://${HOST}"
IMAGE="${2:-samples/bus.jpg}"

echo "==> 1. readiness"
curl -fsS "${BASE}/health/ready" | python3 -m json.tool

echo
echo "==> 2. metadata"
curl -fsS "${BASE}/metadata" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['model_name'], d['class_count'], 'classes')"

echo
echo "==> 3. single prediction"
curl -fsS -X POST -F "file=@${IMAGE}" "${BASE}/predict" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); r=d['results'][0]; print(f\"{r['detection_count']} objects in {d['inference_ms']} ms\"); [print(' ', x['class_name'], round(x['confidence'],2)) for x in r['detections']]"

echo
echo "==> 4. batch prediction"
curl -fsS -X POST -F "files=@${IMAGE}" -F "files=@${IMAGE}" "${BASE}/predict/batch" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['image_count'], 'images in', d['inference_ms'], 'ms')"

echo
echo "==> 5. interactive docs"
echo "    ${BASE}/docs"
