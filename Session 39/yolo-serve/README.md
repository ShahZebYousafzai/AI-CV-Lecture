# yolo-serve

A YOLOv8 object detector served over HTTP with FastAPI, containerised with
Docker, and deployed to AWS. Built for Session 39 of the AI and Computer
Vision curriculum.

The same folder runs unchanged in three places. Only the environment
variables change.

```
laptop          uvicorn app.main:app
docker          docker run -p 8000:8000 yolo-serve
aws             systemd runs the same container on an EC2 instance
```

---

## What is in here

```
yolo-serve/
├── app/
│   ├── main.py                 app factory, lifespan, middleware
│   ├── config.py               every setting, overridable by APP_* env vars
│   ├── api/
│   │   ├── deps.py             dependency providers
│   │   └── routes/
│   │       ├── health.py       liveness, readiness, metadata
│   │       ├── predict.py      single, batch, base64, annotated, jobs
│   │       └── demo.py         the deliberately blocking teaching route
│   ├── schemas/detection.py    request and response contracts
│   ├── services/
│   │   ├── detector.py         the only file that imports Ultralytics
│   │   ├── inference.py        async bridge to the blocking model
│   │   ├── image_io.py         safe decoding, annotation
│   │   └── jobs.py             in memory async job store
│   └── core/                   errors, logging
├── deploy/aws/                 ECR push, EC2 user data, IAM, smoke test
├── scripts/                    model download, benchmarks, demos
├── tests/                      20 tests, run against the real model
├── Dockerfile                  multi stage, CPU only, non root
├── docker-compose.yml
├── requirements*.txt
└── Makefile
```

Why this shape rather than one `main.py`: routes change weekly, models change
monthly, and the two should not be able to break each other. `services/`
knows nothing about HTTP, `api/` knows nothing about Ultralytics, and the
test suite can exercise either half on its own.

---

## Part 1: run it locally

```bash
cd yolo-serve

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements-torch-cpu.txt
pip install -r requirements-dev.txt

python scripts/download_model.py     # 6.5 MB of weights into models/
python scripts/download_samples.py   # two test photos into samples/

uvicorn app.main:app --reload --port 8000
```

Startup logs should read:

```
Starting YOLOv8 Detection Service v1.0.0 (env=local)
Loaded yolov8n.pt on cpu in 0.04s (80 classes)
Warmup inference took 1194 ms
Service ready
```

Those three lines are the whole lesson about model loading. The weights load
once, the first slow inference happens before any user arrives, and only then
does the service report ready.

Open http://localhost:8000/docs for the generated OpenAPI page.

### Call it

```bash
# liveness and readiness are different questions
curl localhost:8000/health/live
curl localhost:8000/health/ready

# one image
curl -X POST -F "file=@samples/bus.jpg" localhost:8000/predict

# many images, one forward pass
curl -X POST -F "files=@samples/bus.jpg" -F "files=@samples/zidane.jpg" \
     localhost:8000/predict/batch

# get a picture back instead of JSON
curl -X POST -F "file=@samples/bus.jpg" \
     localhost:8000/predict/annotated -o out.png

# submit a job, poll it
curl -X POST -F "files=@samples/bus.jpg" -F "files=@samples/zidane.jpg" \
     localhost:8000/jobs
curl localhost:8000/jobs/<job-id>
```

A single prediction on `bus.jpg` returns six detections in roughly 78 ms on a
two core CPU.

### Run the tests

```bash
pytest
# 20 passed
```

These run the real lifespan and the real model. Slower than mocking, and
worth it, because startup is exactly where a model server breaks.

---

## Part 2: the three ideas worth the session

### 1. Load the model once

`app/main.py` loads the weights in the `lifespan` context manager, before the
server accepts its first connection, and hands the loaded object to
`app.state`. Every request then costs one forward pass and nothing else.

Loading inside the handler instead would add the load time to every single
request and hold N copies of the weights in memory under concurrency.

### 2. `async def` does not make CPU work concurrent

This is the bug that hurts real services:

```python
@router.post("/predict")
async def predict(...):
    return model.predict(image)      # blocks the event loop for 300 ms
```

While that line runs, this worker answers nothing. Not the next request, not
the load balancer's health check. The fix is one call:

```python
predictions = await anyio.to_thread.run_sync(lambda: detector.predict(images))
```

Measure it yourself:

```bash
APP_ENABLE_DEMO_ROUTES=true uvicorn app.main:app --port 8000
python scripts/blocking_demo.py --n 8
```

Measured on a two core CPU box:

| Handler | Health checks answered during inference | Worst health latency |
|---|---|---|
| Model called inside `async def` | 2 | 736 ms |
| Model pushed to a worker thread | 32 | 6 ms |

Same model, same machine. That worst case number is what your load balancer
sees, and what decides whether it pulls the instance out of service.

### 3. Batch, but know what it buys you

```bash
python scripts/benchmark.py --n 8
```

Measured output on two CPU cores:

```
1. Same size images, n=8
  N separate /predict calls    wall  598 ms | model 530 ms |  74.8 ms/image
  one /predict/batch call      wall  619 ms | model 579 ms |  77.4 ms/image

2. Mixed size images, n=8
  N separate /predict calls    wall  585 ms | model 518 ms |  73.1 ms/image
  one /predict/batch call      wall  960 ms | model 923 ms | 120.0 ms/image
```

Two honest conclusions that most tutorials skip:

* On a saturated CPU, batching is roughly break even for compute. The wins
  are one HTTP round trip instead of eight, one queue slot instead of eight,
  and per call framework overhead paid once. On a GPU, where batch size 1
  leaves most of the device idle, batching is worth 3x to 5x.
* Mixing image sizes in one batch makes it slower per image. Ultralytics only
  uses rectangular inference when every image in the batch has the same
  shape; otherwise everything is padded to a full square. Resize before you
  batch.

Measure on the hardware you will deploy to. Do not quote the slide.

---

## Part 3: Docker

```bash
docker build -t yolo-serve:1.0.0 -t yolo-serve:latest .
docker run --rm -p 8000:8000 yolo-serve:latest

# or
docker compose up --build
```

Then the same curl commands as before, against port 8000.

### What the Dockerfile is doing, and why

| Choice | Reason |
|---|---|
| Two stages, builder and runtime | Compilers and pip caches never reach the shipped image |
| Dependencies copied before source | Editing a route does not rebuild the four minute pip layer |
| PyTorch from the CPU index | PyPI's `torch` drags in about 2 GB of CUDA that a CPU box never loads |
| `libgl1`, `libglib2.0-0` installed | Ultralytics imports OpenCV, which needs them. Without them the container dies on import |
| Weights baked in at build time | No downloads at startup, no dependency on the internet from inside the VPC |
| Pruning polars, matplotlib, pip | Transitive dependencies inference never imports |
| `USER appuser` | A container running as root is root on the host kernel if anything escapes |
| `HEALTHCHECK` hits `/health/ready` | A container whose model is still loading must not be sent traffic |
| `CMD` in exec form | uvicorn becomes PID 1 and gets SIGTERM directly, so shutdown is clean |

Sizes measured on this build:

| Build | venv | Image |
|---|---|---|
| CPU index for torch | 1.5 GB | 2.57 GB |
| CPU index plus pruning | 932 MB | 1.77 GB |

Not measured here, but worth knowing: the default PyPI `torch` wheel plus the
`nvidia-*` runtime packages it depends on are roughly 2.5 GB of extra download
on top of the first row. On a CPU instance, none of it is ever loaded.

If 1.77 GB is still too much, the next step is exporting to ONNX and serving
with `onnxruntime` alone. That drops PyTorch entirely and lands near 300 MB,
which is the Session 36 optimisation material meeting this one.

### Configuration

Nothing is hardcoded. Every value in `app/config.py` reads from an `APP_`
prefixed environment variable:

```bash
docker run --rm -p 8000:8000 \
  -e APP_CONF_THRESHOLD=0.4 \
  -e APP_TORCH_THREADS=4 \
  -e APP_LOG_LEVEL=DEBUG \
  yolo-serve:latest
```

Same image, different behaviour. That is the property that makes one artefact
promotable from dev to staging to production without a rebuild.

---

## Part 4: deploy to AWS

The path: **build locally, push to ECR, run on EC2 under systemd.**

### Before you start

* An AWS account and `aws configure` done locally
* An EC2 key pair in your chosen region, or SSM Session Manager
* Region set consistently. Everything below uses `ap-south-1`. Change it in
  one place: the variables at the top of the scripts.

### Step 1: push the image to ECR

```bash
export AWS_REGION=ap-south-1
./deploy/aws/01_push_to_ecr.sh
```

The script creates the repository if needed, logs Docker into ECR, builds
with `--platform linux/amd64` and pushes. It prints the image URI, which
looks like:

```
123456789012.dkr.ecr.ap-south-1.amazonaws.com/yolo-serve:1.0.0
```

> The `--platform linux/amd64` flag is not optional on an Apple Silicon Mac.
> An arm64 image on an x86 instance fails with `exec format error`.

### Step 2: create the IAM role for the instance

Console: IAM, Roles, Create role, AWS service, EC2. Attach:

* `AmazonEC2ContainerRegistryReadOnly` so the instance can pull
* `AmazonSSMManagedInstanceCore` so you can get a shell without opening
  port 22

Name it `yolo-serve-ec2-role`. `deploy/aws/iam-ec2-role-policy.json` has the
hand written least privilege equivalent if you prefer an inline policy.

No access keys go on the instance. The role hands out short lived
credentials through the instance metadata service, and they rotate on their
own.

### Step 3: security group

Create `yolo-serve-sg` with:

| Direction | Port | Source | Why |
|---|---|---|---|
| Inbound | 80 | your IP, or `0.0.0.0/0` for the demo | the API |
| Outbound | all | `0.0.0.0/0` | pulling from ECR |

Do not leave port 22 open to the world. With the SSM role attached you do not
need it at all.

### Step 4: launch the instance

| Setting | Value |
|---|---|
| AMI | Amazon Linux 2023 |
| Instance type | `t3.small` (2 vCPU, 2 GB) for yolov8n on CPU |
| Key pair | optional, SSM covers shell access |
| Security group | `yolo-serve-sg` |
| IAM instance profile | `yolo-serve-ec2-role` |
| Storage | 20 GB gp3, the image alone is 1.77 GB |
| User data | paste `deploy/aws/02_ec2_user_data.sh` after editing the two variables at the top |

`t3.micro` has 1 GB of RAM and will be killed by the OOM reaper while the
model loads. `t3.small` is the smallest thing that works.

### Step 5: watch it come up

User data runs once, as root, on first boot. Follow it with:

```bash
aws ssm start-session --target i-0abc123def456
sudo tail -f /var/log/cloud-init-output.log
```

Expect roughly three minutes: instance boot, Docker install, a 1.77 GB pull,
then model load.

### Step 6: prove it works from outside

```bash
./deploy/aws/smoke_test_remote.sh <public-ip>
```

That hits readiness, metadata, a single prediction and a batch prediction
over the public internet, and prints the detections. Open
`http://<public-ip>/docs` to hand the browser to your client.

### Step 7: ship a change

```bash
IMAGE_TAG=1.0.1 ./deploy/aws/01_push_to_ecr.sh
./deploy/aws/03_redeploy.sh i-0abc123def456 1.0.1
```

systemd pulls the new tag and restarts the container. Roll back by
redeploying the previous tag: the old image is still in ECR, which is the
entire reason for immutable tags.

### Where the logs are

The systemd unit uses the `awslogs` Docker log driver, so container stdout
lands in CloudWatch under the log group `/yolo-serve`. This is why
`app/core/logging_config.py` logs to stdout and never to a file: containers
are disposable, and anything written inside one disappears with it.

---

## Where this stops being production ready

Worth saying out loud, because the gap is the interesting part:

| Gap | What production does |
|---|---|
| One instance, one availability zone | Auto Scaling group behind an Application Load Balancer across two AZs |
| Plain HTTP | ACM certificate on the load balancer, HTTP redirected to HTTPS |
| Jobs live in process memory | Redis or SQS, with a separate worker process |
| Anyone can call it | API key on the load balancer, or Cognito, or a VPC only listener |
| No metrics | `prometheus-fastapi-instrumentator`, or CloudWatch custom metrics |
| Deploy by hand | GitHub Actions on push to main: test, build, push, redeploy |
| One worker per container | `--workers N` sized to vCPUs, or more containers behind the load balancer |

Each of those is one small step from where the code already is, which is the
point of structuring it this way in the first place.

---

## Endpoint reference

| Method | Path | Purpose |
|---|---|---|
| GET | `/health/live` | process is up |
| GET | `/health/ready` | model is loaded and can serve |
| GET | `/metadata` | model name, thresholds, class list |
| POST | `/predict` | one image, multipart, returns JSON |
| POST | `/predict/batch` | many images, one forward pass |
| POST | `/predict/base64` | many images as a JSON body |
| POST | `/predict/annotated` | one image, returns a PNG with boxes drawn |
| POST | `/jobs` | submit a batch, returns 202 and a job id |
| GET | `/jobs/{job_id}` | poll a job |
| GET | `/docs` | interactive OpenAPI page |

Every response carries `X-Request-ID` and `X-Process-Time-Ms`. Every error
has the same shape:

```json
{"error": {"code": "invalid_image", "message": "...", "request_id": "..."}}
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `libGL.so.1: cannot open shared object file` | OpenCV's system libraries missing | `apt-get install libgl1 libglib2.0-0` in the runtime stage |
| `exec format error` on EC2 | arm64 image on an x86 instance | rebuild with `--platform linux/amd64` |
| Container killed during startup | not enough RAM | use `t3.small` or larger |
| `/health/ready` returns `loading` forever | weights missing from the image | check `RUN python scripts/download_model.py` succeeded in the build |
| Connection times out from your laptop | security group | open port 80 inbound |
| `denied: requested access to the resource is denied` | ECR login expired | rerun the `aws ecr get-login-password` and `docker login` pair |
| Health checks slow while predicting | blocking handler | run the model with `anyio.to_thread.run_sync` |
