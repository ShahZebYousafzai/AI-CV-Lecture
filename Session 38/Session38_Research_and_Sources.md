# Session 38 · Model Optimization: research notes and sources

Everything on the slides and in the transcript traces back to one of the sources below.
Where a number appears on a slide, the source is named on the slide itself as well.
Research was carried out on 1 September 2026 against the current published documentation.

---

## 1. ONNX export

**What it is.** ONNX (Open Neural Network Exchange) is an open standard for representing
machine learning models as a computation graph. The ONNX project describes it as a common
Intermediate Representation whose goal is to "empower developers to select the framework
that works best for their project, at any stage of development or deployment," and notes
that a standard representation "enables hardware vendors to streamline optimizations for
their platforms."

**Structure of an `.onnx` file.** A protobuf containing a directed acyclic graph. Nodes are
standard operators (Conv, Relu, MatMul, Resize, ...), edges are named tensors, and the
trained weights are stored in the same file as initializers.

**Opsets.** Operators are versioned into numbered operator sets. Exporting at a given opset
pins the model to the operator definitions frozen in that release, which is what allows a
runtime to load the model years later without a matching framework install.

**How PyTorch export works.** `torch.onnx.export` traces the model: one dummy input is pushed
through, every operator that actually executes is recorded, and each is mapped to its ONNX
equivalent at the requested opset. Consequences taught on slide 7:

- Control flow is resolved at trace time. A branch that was not taken does not exist in the
  exported graph, and Python loops are unrolled to the number of iterations that ran.
- `dynamic_axes` must be set for batch / height / width, otherwise the graph is fixed to the
  dummy input's shape.
- Verification against the PyTorch model on real inputs is the standard practice after export.

**Free speedup from the runtime.** ONNX Runtime applies graph optimizations at session load,
in three levels: *Basic* ("semantics-preserving graph rewrites which remove redundant nodes
and redundant computation": constant folding, Identity/Slice/Unsqueeze/Dropout elimination,
Conv+Add and Conv+BatchNorm fusion), *Extended* (GELU fusion, Layer Normalization fusion,
Attention fusion), and *Layout* (NCHWc on CPU).

Sources:
- [Introduction to ONNX](https://onnx.ai/onnx/intro/) · onnx.ai
- [ONNX Overview](https://github.com/onnx/onnx/blob/main/docs/Overview.md) · onnx/onnx
- [ONNX IR specification](https://onnx.ai/onnx/repo-docs/IR.html) · onnx.ai
- [torch.onnx](https://docs.pytorch.org/docs/stable/onnx.html) · PyTorch documentation
- [ONNX Runtime graph optimizations](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html)

---

## 2. TensorRT

**Two phases.** NVIDIA's documentation splits TensorRT into a *build phase*, in which a
builder compiles the network and selects the fastest GPU-specific kernel for each layer on
the target device, and a *runtime phase*, in which the resulting engine (also called a plan
file) is loaded and executed. The engine is a serialized binary containing the graph, the
chosen CUDA tactics, plugin pointers and the weights.

**What the builder does.**
- *Layer and tensor fusion*, including pointwise fusion and Q/DQ fusion. The arithmetic is
  unchanged; what is removed is kernel-launch overhead and the intermediate tensors written
  to and read back from GPU memory.
- *Kernel auto-tuning / tactic selection*: multiple candidate kernel implementations are
  timed on the actual GPU at the actual shapes, and the fastest is kept.
- *Precision control*: mixed-precision strategies across FP32 / FP16 / INT8, decided layer by
  layer, with a documented accuracy-versus-performance tradeoff.
- *Memory management and parallelism*: workspace allocation during the build, plus batching,
  CUDA graphs and within-/cross-inference multi-streaming at runtime.

**Portability.** NVIDIA states that "serialized engines are only guaranteed to work correctly
when used with the same OS, CPU architectures, GPU models, and TensorRT versions used to
serialize the engines." This is the basis for the "build on the deployment device" rule on
slide 8.

**Precisions supported.** Beyond FP32/FP16, TensorRT documents INT8 (signed 8-bit integer),
INT4 (weight-only), FP8E4M3 and FP4E2M1, using a symmetric quantization scheme in which
"both activations and weights are mapped to quantized values centered around zero."

Sources:
- [How TensorRT works](https://docs.nvidia.com/deeplearning/tensorrt/latest/architecture/how-trt-works.html)
- [Best practices / optimization](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html)
- [Working with quantized types](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-with-quantized-types.html)
- [TensorRT SDK product page](https://developer.nvidia.com/tensorrt)

---

## 3. Quantization

**Number formats** (slide 10). Bit layouts and properties are standard IEEE 754 / bfloat16 /
int8 facts:

| Format | Sign | Exponent | Mantissa | Bytes | Range | Notes |
|---|---|---|---|---|---|---|
| FP32 | 1 | 8 | 23 | 4 | ±3.4e38 | ~7 decimal digits; the training default |
| FP16 | 1 | 5 | 10 | 2 | ±65504 | ~3 decimal digits |
| BF16 | 1 | 8 | 7 | 2 | same as FP32 | less precision, fewer overflow issues |
| INT8 | 1 | none | 7 magnitude | 1 | -128 to +127 | 256 levels, needs calibration |

**The mapping** (slide 11). ONNX Runtime states the 8-bit linear quantization relation as
`val_fp32 = scale * (val_quantized - zero_point)`, with

- asymmetric: `scale = (data_range_max - data_range_min) / (quant_range_max - quant_range_min)`
- symmetric: `scale = max(abs(data_range_max), abs(data_range_min)) * 2 / (quant_range_max - quant_range_min)`

The zero point exists so that floating-point zero is exactly representable, which matters for
CNNs that use zero padding. TensorRT uses symmetric quantization.

**Representation.** Two forms: *QOperator* (dedicated quantized ops such as QLinearConv,
MatMulInteger) and *QDQ* (QuantizeLinear/DequantizeLinear pairs inserted around the original
operators).

**PTQ vs QAT.** Static post-training quantization computes activation ranges offline from
calibration data using MinMax, Entropy or Percentile calibrators; dynamic quantization
computes them at inference time. ONNX Runtime recommends dynamic for RNNs and transformers,
static for CNNs. Quantization-aware training simulates the rounding during the forward pass
so weights adapt to it.

**Hardware caveat.** Gains depend on hardware support: "x86-64 with VNNI, GPU with Tensor Core
int8 support and Arm-based processors with dot-product instructions can get better performance
in general." On older hardware the quantize/dequantize overhead can outweigh the benefit.

Sources:
- [ONNX Runtime quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
- [TensorRT: working with quantized types](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-with-quantized-types.html)

---

## 4. Pruning

**The idea and the loop** (slide 12). Han, Pool, Tran & Dally (2015) introduced the
train / prune / retrain pipeline: train the network to learn which connections matter, remove
all connections whose weights fall below a threshold, then retrain the remaining sparse
network. The paper states explicitly that "if the pruned network is used without retraining,
accuracy is significantly impacted."

**Unstructured vs structured** (slide 13). PyTorch's pruning API implements pruning as a mask:
the parameter `weight` is replaced by `weight_orig` plus a binary `weight_mask` buffer, and
`weight` is recomputed by multiplying the two in a forward pre-hook. Unstructured pruning
(`L1Unstructured`) zeros individual entries; structured pruning zeros whole channels along a
chosen dimension. Local pruning ranks within one tensor; global pruning ranks across the whole
model, giving uneven per-layer sparsity at a fixed global rate. Note that the PyTorch tutorial
covers the mechanics of producing sparsity, not inference speedup. The sparsity-is-not-speed
point on slide 13 comes from the hardware side, below.

**2:4 semi-structured sparsity.** NVIDIA's Ampere and later architectures support a pattern in
which "in each contiguous block of four values, two values must be zero" (50% sparsity). Sparse
Tensor Cores "operate only on the nonzero values in the compressed matrix," using metadata to
locate the matching operands, giving roughly half the work for the matrix multiply. NVIDIA's
recipe (ASP / Automatic SParsity) is: start from a trained dense model, apply one-shot
magnitude pruning to satisfy the 2:4 constraint, then retrain with the original
hyperparameters, and they report sparse models matching dense baseline accuracy across
ResNet-50, BERT, transformers and detection networks.

Sources:
- [Han, Pool, Tran & Dally (2015), *Learning both Weights and Connections for Efficient Neural Networks*, arXiv:1506.02626](https://arxiv.org/abs/1506.02626)
- [PyTorch pruning tutorial](https://docs.pytorch.org/tutorials/intermediate/pruning_tutorial.html)
- [Accelerating Inference with Sparsity Using Ampere and TensorRT](https://developer.nvidia.com/blog/accelerating-inference-with-sparsity-using-ampere-and-tensorrt/) · NVIDIA Technical Blog

---

## 5. Combining the techniques (slide 15)

Han, Mao & Dally (2016), *Deep Compression*, ICLR 2016, chains pruning, trained quantization
and Huffman coding. Reported results, quoted on the slide:

- pruning alone: 9x to 13x fewer weights
- pruning + quantization: 27x to 31x reduction
- all three stages: 35x to 49x, "without affecting their accuracy"
- AlexNet 240 MB to 6.9 MB (35x); VGG-16 552 MB to 11.3 MB (49x)
- 3x to 4x layerwise speedup and 3x to 7x better energy efficiency across CPU, GPU and mobile GPU

The teaching point taken from this is not the exact ratios but that the techniques attack
different redundancies and therefore compose.

Source: [Han, Mao & Dally (2016), *Deep Compression*, arXiv:1510.00149](https://arxiv.org/abs/1510.00149)

---

## 6. Worked example (slide 16)

Ultralytics' TensorRT integration documentation publishes YOLO26 export benchmarks on COCO
detection. The figures used on slide 16:

| GPU | Precision | Inference time (ms) | mAP50 | mAP50-95 |
|---|---|---|---|---|
| NVIDIA A100 | FP32 | 0.52 | 0.52 | 0.37 |
| NVIDIA A100 | FP16 | 0.34 | 0.52 | 0.37 |
| NVIDIA A100 | INT8 | 0.28 | 0.47 | 0.33 |
| RTX 3080 12GB | FP32 | 1.06 | 0.52 | 0.37 |
| RTX 3080 12GB | FP16 | 0.62 | 0.52 | 0.37 |
| RTX 3080 12GB | INT8 | 0.52 | 0.47 | 0.33 |

The same page notes that INT8 export uses post-training quantization, that NVIDIA recommends
"at least 500 calibration images that are representative of the data for your model," and that
"it is critical to ensure that the same device that will use the TensorRT model weights for
deployment is used for exporting with INT8 precision, as the calibration results can vary
across devices."

Source: [Ultralytics · TensorRT export](https://github.com/ultralytics/ultralytics/blob/main/docs/en/integrations/tensorrt.md)

---

## Figure attributions

Two figures in the deck are original figures from publicly available arXiv papers, reproduced
for teaching with attribution, cropped to the figure region and otherwise unaltered. Each
carries a visible credit line on its slide. No AI image generation was used anywhere in this
session.

| File | Slide | Source | Figure | Licence |
|---|---|---|---|---|
| `figures/fig_pruning_synapses_han2015.png` | 12 | Han, Pool, Tran & Dally (2015), [arXiv:1506.02626](https://arxiv.org/abs/1506.02626) | Figure 3, synapses and neurons before and after pruning | arXiv non-exclusive licence |
| `figures/fig_deepcompression_pipeline_han2016.png` | 15 | Han, Mao & Dally (2016), [arXiv:1510.00149](https://arxiv.org/abs/1510.00149), ICLR 2016 | Figure 1, the three stage compression pipeline with per-stage reduction ratios | arXiv non-exclusive licence |

How they were obtained: each PDF was downloaded from `arxiv.org/pdf/<id>` and the figure region
was cropped at 300 DPI with PyMuPDF (`page.get_pixmap(dpi=300, clip=...)`).

| File | Page (0-indexed) | Clip rect (PDF points) |
|---|---|---|
| `fig_pruning_synapses_han2015.png` | 2 | 300, 72, 508, 182 |
| `fig_deepcompression_pipeline_han2016.png` | 1 | 102, 64, 510, 212 |

Two further figures were **drawn for this session** from the data and definitions cited above,
using matplotlib (`make_charts.py`):

| File | Slide | What it shows |
|---|---|---|
| `figures/fig_quantization_mapping.png` | 11 | A synthetic FP32 weight distribution against a 256-level INT8 grid, illustrating scale, level snapping and the resulting information loss. Follows the ONNX Runtime quantization definition. |
| `figures/fig_precision_benchmark.png` | 16 | The Ultralytics YOLO26 latency and mAP figures in the table above. |

Every other diagram in the deck (the pipeline flow, the ONNX interoperability comparison, the
build/runtime chain, the layer fusion before-and-after, the FP32/FP16/BF16/INT8 bit-field bars,
and the three sparsity grids) is drawn natively in PowerPoint shapes, in the same convention
used for the Session 34 decks.

---

## Deliverables

| File | What it is |
|---|---|
| `Session38_Model_Optimization.pptx` | 17-slide deck for the 30-minute theory session |
| `Session38_Delivery_Transcript.docx` | Word-for-word delivery script with timings and stage cues |
| `Session38_Research_and_Sources.md` | This file |
| `figures/` | The four images used in the deck |
| `make_charts.py` | Regenerates the two matplotlib figures |
