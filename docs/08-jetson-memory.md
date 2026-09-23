# 08 -- Memory on the Jetson AGX Orin

Everything in this chapter was measured on `jarvita-agx` (64 GB AGX Orin,
JetPack 7.2) during the GLiNER work in September 2026, and it is the reason
this server is shaped the way it is.

## Unified memory changes the rules

On a desktop GPU, video memory is a separate pool: a model that overshoots
gets a CUDA out-of-memory error and the rest of the machine is unaffected. On
the Orin, CPU and GPU share one 61.35 GiB pool. A GPU allocation is RAM the
kernel cannot give to anything else. Consequences:

* `docker stats` and `free` *do* see GPU memory (it shows up as the
  container's "kernel" memory).
* Overshooting does not produce a clean CUDA error. With zero swap the kernel
  OOM-killer takes the largest process -- on 2026-08-28 that was the LLM
  server, and it took sshd down with it for 18 minutes. Zram swap is now on,
  but GPU pages are not swappable, so it cushions the CPU side only.
* PyTorch's out-of-memory path calls NVML, which Tegra does not properly
  support, and asserts instead of raising. The process wedges. (candle does
  not use NVML; a CUDA allocation failure surfaces as an `Err`, which is a
  real advantage of this stack on this board.)

## The budget this server has to live in

| Resident | GiB |
|---|---|
| Nemotron-3.5 vLLM (weights + pinned KV pool) | ~40 |
| GLiNER extraction API, at rest | ~1.4 |
| GLiNER, worst-case request | ~8 |
| System | ~6 |
| **Free at the worst moment** | **~6** |

So the embedder gets about 3-4 GiB, all in. That is the target for
`MAX_BATCH_TOKENS` and the dtype choice below.

## Where an embedding model's memory goes

**Weights.** pplx-embed is 0.6 B parameters: 2.4 GB in f32, 1.2 GB in f16.
BGE-base is 110 M: 0.44 / 0.22 GB. Load in f16 on the Jetson (`DTYPE=f16`,
the default); use f32 only for `verify` on a CPU.

**Activations.** Per layer, `(batch, seq, hidden)` plus the MLP's
`(batch, seq, 3 x hidden)` -- linear in seq, freed layer by layer.

**Attention scores.** `(batch, heads, seq, seq)`. This is the quadratic
term and the only one that can surprise you. For pplx (16 heads) in f16:

| seq | one text | 8 texts |
|---|---|---|
| 512 | 8 MB | 64 MB |
| 2048 | 128 MB | 1 GB |
| 8192 | 2 GB | 16 GB |

GLiNER's numbers tell the same story from the other side: a single
15,000-character document (~3,000 tokens) peaked at 3.3 GiB; two of them in a
batch, 5.6 GiB; three, 8.1 GiB. **Batch cost is `count x longest^2`, because
padding makes every member as long as the longest.**

## The knobs, and what each bounds

| Setting | Bounds | Default | Jetson guidance |
|---|---|---|---|
| `DTYPE` | weights + activations | f16 | f16. bf16 if a model overflows f16 |
| `MAX_SEQ_LEN` | worst single input | model's | 2048 for pplx unless you need 32k |
| `MAX_BATCH_TOKENS` | `count x longest` per pass | 16384 | 16384 at seq 2048 = 8 texts; lower if `free` says so |
| `MAX_INPUTS` | request size (413) | 256 | leave; batching splits it internally |

There is no "GPU memory fraction" setting, deliberately. candle allocates
what a forward pass needs and frees it when the tensors drop -- there is no
caching allocator holding a high-water mark, which was the entire GLiNER
problem. What you configure is the *input*, and the input determines the
peak.

## Measuring instead of guessing

```bash
# on the Jetson, in another shell, while sending a worst-case request:
watch -n 0.5 'free -m | head -2; cat /sys/fs/cgroup/system.slice/docker-<id>.scope/memory.current'
```

Record the peak for your largest realistic batch and write it into
`expected-services.conf` and the box's memory budget in the home-lab-ops
runbook. Every service on that box has a number; this one should too.

## Building for CUDA on the Jetson

candle's CUDA backend needs `nvcc` and the compute capability:

```bash
export CUDA_COMPUTE_CAP=87        # Orin
cargo build --release --features cuda
```

JetPack 7.2 hosts do not ship the toolkit by default; `Dockerfile.jetson`
builds inside `nvcr.io/nvidia/l4t-jetpack:r36.4.0` (CUDA 12.6), which runs
fine on the JetPack 7 host -- the same base the GLiNER image uses. Do not
reach for the `sbsa`/`cu130` wheels or images; those target Grace/Blackwell
(sm_90/100), not Orin's sm_87, and fail at the first kernel launch.
