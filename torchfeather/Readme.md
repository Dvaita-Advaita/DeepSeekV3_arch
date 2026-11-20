# TorchFeather: DeepSeek-V3 training from scratch (work in progress)

I'm re-implementing the DeepSeek-V3 architecture and a distributed trainer from scratch, to learn
large-model training (MLA, MoE, 4D/5D parallelism) by building it. The trainer's structure is modelled
on [TorchTitan](https://github.com/pytorch/torchtitan). The model code is written from the DeepSeek-V2/V3 papers.

## Status

| Part | File | Status |
|---|---|---|
| Multi-head Latent Attention, incl. weight absorption for inference | `model/model.py` | ✅ written |
| RoPE with YaRN context extension | `model/rope.py` | ✅ written |
| Depth-scaled weight initialisation | `model/model.py` | ✅ written |
| Pipeline parallelism (GPipe, 1F1B, Interleaved 1F1B, Zero Bubble) | `distributed/pipeline_parallel.py` | ✅ written |
| Config, tokenizer, HF dataloader | `config/`, `components/`, `datasets/` | ✅ written |
| Mixture-of-Experts + load balancing | `model/moe.py` | 🚧 in progress |
| Attention wrapper, FSDP/TP parallelize | `model/attention.py`, `model/parallelize.py` | 🚧 in progress |
| Checkpointing, loss, optimizer, LR schedule, metrics | `components/*` | ⏳ next |
| First end-to-end training run | — | ⏳ next |

`train.py` imports some modules that aren't pushed yet, so **it does not run end-to-end today**.

## What I've learned so far

**RoPE (Rotary Position Embedding).** Each pair of dimensions in a query or key is rotated by an angle
proportional to the token's position, with a different frequency per pair. The dot product between a query
and a key then depends only on their *relative* distance, not absolute positions. Nothing is added to the
embeddings, so position information can't get washed out.

**YaRN (context extension).** If a model trained on 4K tokens sees 128K, its low-frequency RoPE dimensions
hit angles they never saw in training. YaRN splits the dimensions by wavelength (`beta_fast`, `beta_slow`).
High-frequency dimensions are left alone, low-frequency ones are interpolated, and a linear ramp blends the
range in between. It also scales the attention temperature (`mscale`) so the softmax doesn't get too flat at
long context. DeepSeek-V3 also applies `mscale` to the softmax scale in MLA.

**Weight initialisation.** Every residual block adds its output to the stream, so with naive init the
activation variance grows with depth. Here each layer's std is `0.02 / sqrt(2 · (layer_id + 1))`. That
shrinks later layers' contributions, which differs from GPT-2's single global `1/sqrt(2·n_layers)` scale.
I also learned why PyTorch's default `nn.Linear` init gets overwritten with an explicit `init_weights()`.

**Pipeline parallelism.** Split the model's layers into stages on different GPUs and feed micro-batches
through them. The cost is the *bubble*, the time a GPU sits idle waiting for its neighbours:
- **GPipe:** all forwards, then all backwards. Simple, but the bubble is large and every micro-batch's activations must be held.
- **1F1B:** after a warm-up, each stage alternates one forward and one backward. Same bubble as GPipe, but activation memory is bounded by the number of stages, not micro-batches.
- **Interleaved 1F1B:** each GPU holds several non-adjacent stages (rank 0 holds stages 0 and 4, say). The bubble shrinks by the number of virtual stages, at the cost of more communication.
- **Zero Bubble:** splits backward into *input-gradient* (B) and *weight-gradient* (W). B sits on the critical path but W doesn't, so W fills the idle slots and the bubble almost disappears. The V-shaped variant (`ScheduleZBVZeroBubble`) places stages in a "V" across ranks.
- **DualPipe (DeepSeek-V3):** feeds micro-batches from *both ends* of the pipeline at once and overlaps compute with communication, especially the MoE all-to-all, inside each forward/backward pair.

I also learned how to balance stages when the embedding and output layers cost more or less than a
transformer block (the `first_stage_less_layers` / `last_stage_less_layers` weights).

**Device mesh.** `DeviceMesh` arranges GPUs as an N-dimensional grid, for example `(pp, dp, tp)`, so each
parallelism uses its own sub-group. Tensor parallelism talks along the `tp` axis (fast, intra-node),
data parallelism along `dp`, and pipeline along `pp`. Placing the chattiest parallelism on the fastest
links is most of the battle.

## References
- DeepSeek-AI, *DeepSeek-V2* (MLA) and *DeepSeek-V3* technical reports (DualPipe)
- Su et al., *RoFormer* (RoPE); Peng et al., *YaRN*
- Huang et al., *GPipe*; Narayanan et al., *Megatron-LM* (1F1B, interleaved); Qi et al., *Zero Bubble Pipeline Parallelism*
- [TorchTitan](https://github.com/pytorch/torchtitan)
