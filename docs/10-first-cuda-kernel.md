# 10 -- Your first CUDA kernel: fused mean pooling

`mean_pool` in `pooling.rs` is three tensor operations:

```rust
let summed = (hidden * mask3)?.sum(1)?;     // multiply, then reduce
let counts = mask.sum(1)?...                // reduce
summed / counts                             // divide
```

Each is a separate kernel launch, and the first materialises a full
`(batch, seq, hidden)` product before summing it. A single hand-written
kernel does the multiply-and-add in one pass with no intermediate. It is a
small, self-contained, verifiable target -- which is what a first kernel
should be.

## The kernel

One thread block per (batch row, hidden dimension chunk); each thread walks
the sequence for its dimension, accumulating only where the mask is 1.

```cuda
// mean_pool.cu
extern "C" __global__ void mean_pool_f16(
    const __half* __restrict__ hidden,   // (b, l, d)
    const unsigned int* __restrict__ mask,  // (b, l)
    __half* __restrict__ out,            // (b, d)
    int l, int d)
{
    int b = blockIdx.x;
    int j = blockIdx.y * blockDim.x + threadIdx.x;   // hidden dimension index
    if (j >= d) return;

    float acc = 0.f;
    int count = 0;
    for (int t = 0; t < l; ++t) {
        unsigned int m = mask[b * l + t];
        if (m) {
            acc += __half2float(hidden[(b * l + t) * d + j]);
            ++count;
        }
    }
    out[b * d + j] = __float2half(acc / (count > 0 ? count : 1));
}
```

Accumulate in f32 even though the data is f16: summing 2,000 half-precision
numbers in half precision loses digits, and `verify` will tell you.

## Hooking it into candle

candle lets you define a custom op that supplies its own CUDA implementation
(`candle_core::CustomOp2` for a two-input op). The shape of the work:

1. Compile the `.cu` to PTX at build time (`build.rs` calling `nvcc -ptx`,
   or the `bindgen_cuda` crate candle itself uses) and `include_str!` it.
2. Implement `CustomOp2` with a `cpu_fwd` (reuse the existing tensor version,
   so CPU builds and `verify` still work) and a `cuda_fwd` that loads the
   PTX with `cudarc`, computes the grid (`b` x `ceil(d / 256)` blocks of 256
   threads), and launches.
3. Swap `mean_pool` to `hidden.apply_op2(mask, MeanPoolOp)` behind a feature
   flag so you can A/B it.

## Verifying and measuring

* `verify` must still pass at 0.999 -- the kernel is only correct if the
  numbers match.
* Time `pool()` alone before and after on `(8, 2048, 1024)` f16. Expect the
  fused kernel to win by the cost of the eliminated intermediate; expect the
  *end-to-end* request time to barely move, because pooling is a rounding
  error next to 28 attention layers. That is the real lesson: profile before
  optimising, and the profile says attention.

## Where to go from here

* **Fused RMSNorm** (`x * rsqrt(mean(x^2) + eps) * w`): one reduce and one
  multiply per token, used 57 times per forward pass in pplx. Bigger win.
* **The mask-add-softmax** in attention (`scores + bias -> softmax`): fusing
  these avoids materialising `(b, h, l, l)` twice. This is the on-ramp to
  understanding what flash attention does, which is the same idea taken to
  its conclusion: never materialise the scores at all.
* candle's own `candle-kernels` crate is a readable collection of exactly
  these kinds of kernels; read `reduce.cu` after writing yours.
