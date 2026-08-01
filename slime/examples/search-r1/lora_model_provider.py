"""Custom model provider that injects LoRA adapters into a Qwen3-style GPTModel.

Usage:
  --custom-model-provider-path examples/search-r1/lora_model_provider.custom_model_provider
  --only-train-params-name-list 'lora_A|lora_B'

Design:
  - Builds the stock GPTModel (via the existing bridge/raw provider path).
  - Injects rank-r LoRA adapters (lora_A, lora_B) onto target linear modules
    (TELayerNormColumnParallelLinear, TERowParallelLinear for QKV/proj/MLP).
  - Monkey-patches each target module's forward to add the adapter contribution.
  - Adapters are HIDDEN from the weight-sync loop by a name-filter applied in
    common.py's named_params_and_buffers (see Task 4). They're also hidden from
    TensorBackuper via the same filter (backup/restore only base weights).
  - Before SGLang weight-sync (and before backup("actor")), a MERGE hook adds
    (α/r)·B@A into weight.data in-place. After sync, UNMERGE subtracts it.
  - For ref-model KL: a global flag `_lora_enabled` on the model is set False
    before switching to ref, ensuring the forward addon is skipped.

TP sharding (TP>1):
  - ColumnParallel (linear_qkv, linear_fc1): weight sharded dim=0 (output dim).
    lora_A = (r, input_size) REPLICATED. lora_B = (output_size_per_partition, r) SHARDED.
    Merge: weight_local += (α/r) * B_local @ A  (correct: output-dim-sliced result).
  - RowParallel (linear_proj, linear_fc2): weight sharded dim=1 (input dim).
    lora_A = (r, input_size_per_partition) SHARDED. lora_B = (output_size, r) REPLICATED.
    Merge: weight_local += (α/r) * B @ A_local  (correct: input-dim-sliced of B@A).

Environment variables (override defaults):
  LORA_RANK          default 16
  LORA_ALPHA         default 32
  LORA_DROPOUT       default 0.0
  LORA_TARGETS       default "linear_qkv,linear_proj,linear_fc1,linear_fc2"
"""

import logging
import os
import re
import weakref
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Install global LoRA lifecycle hooks (merge-around-backup, merge-around-sync)
# This MUST happen before actor.__init__ creates TensorBackuper instances.
# ---------------------------------------------------------------------------
from lora_hooks import install_global_lora_hooks
install_global_lora_hooks()

# ---------------------------------------------------------------------------
# Configuration from env
# ---------------------------------------------------------------------------
LORA_RANK = int(os.environ.get("LORA_RANK", "16"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "32"))
LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", "0.0"))
LORA_TARGETS = os.environ.get("LORA_TARGETS", "linear_qkv,linear_proj,linear_fc1,linear_fc2").split(",")
LORA_SCALING = LORA_ALPHA / LORA_RANK

# ---------------------------------------------------------------------------
# LoRA injection
# ---------------------------------------------------------------------------

def _tp_world_size() -> int:
    try:
        from megatron.core import parallel_state as mpu

        return mpu.get_tensor_model_parallel_world_size()
    except Exception:
        return 1


def _tp_group():
    from megatron.core import parallel_state as mpu

    return mpu.get_tensor_model_parallel_group()


def _make_replicated_grad_allreduce_hook(pname: str):
    """Grad hook for a REPLICATED LoRA factor under TP>1.

    A replicated factor (lora_A of a ColumnParallel linear, lora_B of a RowParallel one)
    is held identically on every TP rank, but each rank only computes a PARTIAL gradient
    (it owns a slice of the other factor). Megatron only all-reduces grads across TP for
    sequence-parallel LayerNorm params, so without this hook the replicas drift apart ->
    after merge each TP rank writes a DIFFERENT delta into its base-weight shard.
    SUM is the correct reduction (the partials are additive).
    """

    def _hook(grad):
        import torch.distributed as dist

        if grad is None:
            return grad
        dist.all_reduce(grad, group=_tp_group())
        return grad

    return _hook


_GRAD_DEBUG_BUDGET = int(os.environ.get("LORA_GRAD_DEBUG", "24"))
_grad_debug_seen = {"n": 0}


def _make_grad_debug_hook(pname: str):
    """One-shot-ish gradient tracer.

    THE question when train/grad_norm==0 is whether autograd gradients reach the adapter
    params at all. If these lines never appear, the LoRA params are not in the backward
    graph (e.g. activation recompute dropped them because no checkpoint INPUT required
    grad). If they appear with a non-zero norm but grad_norm stays 0, the gradient is
    lost between autograd and the Megatron grad buffer / optimizer instead.
    """

    def _hook(grad):
        if _grad_debug_seen["n"] < _GRAD_DEBUG_BUDGET:
            _grad_debug_seen["n"] += 1
            try:
                gn = float(grad.detach().float().norm())
            except Exception:
                gn = float("nan")
            print(f"[LORA_GRAD] {pname} grad_norm={gn:.6e} shape={tuple(grad.shape)}", flush=True)
        return grad

    return _hook


def _inject_lora(module: nn.Module, name: str, rank: int, scaling: float, dropout: float):
    """Inject lora_A, lora_B parameters into an existing TE linear module and
    monkey-patch its forward to add the adapter contribution.

    For Column-parallel (weight sharded dim=0):
      weight shape = (out_per_partition, in_size)
      lora_A: (rank, in_size)       — replicated across TP
      lora_B: (out_per_partition, rank) — partitioned like weight dim=0

    For Row-parallel (weight sharded dim=1):
      weight shape = (out_size, in_per_partition)
      lora_A: (rank, in_per_partition) — partitioned like weight dim=1
      lora_B: (out_size, rank)         — replicated
    """
    weight = module.weight
    out_features, in_features = weight.shape  # already the local (per-partition) shape
    dtype = weight.dtype

    # Determine partition_dim from the TP attribute on weight
    pdim = getattr(weight, "partition_dim", -1)
    tp = _tp_world_size()

    # Init: A with kaiming_uniform (for nonzero gradient at start), B with zeros (LoRA convention)
    lora_A = nn.Parameter(torch.zeros(rank, in_features, dtype=dtype, device=weight.device))
    lora_B = nn.Parameter(torch.zeros(out_features, rank, dtype=dtype, device=weight.device))
    nn.init.kaiming_uniform_(lora_A, a=5**0.5)
    # B is zero-init -> adapter starts as identity

    # Register as parameters with recognizable names
    module.register_parameter("lora_A", lora_A)
    module.register_parameter("lora_B", lora_B)

    # Set TP attributes so the sync filter knows how to handle them
    # (but they'll be FILTERED OUT of sync entirely)
    lora_A.tensor_model_parallel = (pdim == 1)  # A is sharded only for RowParallel
    lora_A.partition_dim = 1 if pdim == 1 else -1
    lora_A.partition_stride = 1
    lora_B.tensor_model_parallel = (pdim == 0)  # B is sharded only for ColumnParallel
    lora_B.partition_dim = 0 if pdim == 0 else -1
    lora_B.partition_stride = 1

    # RowParallel base output is ALREADY all-reduced across TP by the TE module, while the
    # adapter computes B @ (A_local @ x_local) — only this rank's partial. Adding it after
    # the reduction gives every rank a different (1/TP-of-the-sum) result, so the training
    # forward silently disagrees with the merged weights used for rollout. Reduce the
    # adapter output too (autograd-aware: all-reduce fwd / identity bwd).
    module._lora_needs_tp_reduce = (pdim == 1 and tp > 1)

    if tp > 1:
        # Keep the replicated factor's gradient identical on every TP rank.
        if pdim == 0:
            lora_A.register_hook(_make_replicated_grad_allreduce_hook(f"{name}.lora_A"))
        elif pdim == 1:
            lora_B.register_hook(_make_replicated_grad_allreduce_hook(f"{name}.lora_B"))

    if _GRAD_DEBUG_BUDGET > 0:
        lora_B.register_hook(_make_grad_debug_hook(f"{name}.lora_B"))
        lora_A.register_hook(_make_grad_debug_hook(f"{name}.lora_A"))

    # Dropout
    if dropout > 0:
        module._lora_dropout = nn.Dropout(p=dropout)
    else:
        module._lora_dropout = None

    module._lora_scaling = scaling

    # Store original forward
    module._original_forward = module.forward

    # Patch forward
    def _lora_forward(self, x, *args, **kwargs):
        # Original TE forward (includes fused LayerNorm for Column types)
        result = self._original_forward(x, *args, **kwargs)

        # Check global enable flag (disabled during ref/teacher forward)
        model_root_ref = getattr(self, "_lora_model_root_ref", None)
        if model_root_ref is not None:
            model_root = model_root_ref()
            if model_root is not None and not getattr(model_root, "_lora_enabled", True):
                return result

        # TE modules return (output, bias) or just output
        result_was_tuple = isinstance(result, tuple)
        if result_was_tuple:
            output, bias = result
        else:
            output = result
            bias = None

        # Adapter contribution: x may have been layernorm'd inside the TE module.
        # For TELayerNormColumnParallelLinear, the LN is fused and x we receive is
        # pre-LN. We need post-LN x for the adapter.  But accessing it is non-trivial
        # without breaking into TE internals.
        #
        # SIMPLIFICATION: Instead of intercepting the internal LN'd activations, we
        # apply the adapter on the RAW input x.  This means the adapter sees pre-LN
        # features.  For LoRA this is acceptable and commonly done in practice (many
        # LoRA implementations apply to the pre-LN input since W already includes the
        # LN transform conceptually).
        #
        # Actually for merge-back correctness, the adapter MUST match what merge does:
        #   merge: weight += (α/r)·B@A
        #   so the effective output = weight @ LN(x).T
        # If we apply adapter on raw x (not LN'd), the forward won't match the merged
        # version.  BUT during training we DON'T merge (merge only for sync).  So the
        # training forward is: TE_forward(x) + adapter(x_raw).
        #
        # For the adapter to be mathematically consistent with merge, we'd need LN(x).
        # However, the LayerNorm weights are INSIDE the TE module.  Getting them:
        #   - self.layer_norm_weight exists as a parameter on the module!
        #
        # Let's do it properly: apply RMSNorm manually for the adapter path.
        lora_input = x
        if hasattr(self, "layer_norm_weight"):
            # This is a TELayerNormColumnParallelLinear — x is pre-LN
            ln_w = self.layer_norm_weight
            # RMSNorm: x_norm = x / rms(x) * w
            variance = lora_input.float().pow(2).mean(-1, keepdim=True)
            eps = getattr(self, "layer_norm_epsilon", 1e-6)
            lora_input = (lora_input.float() * torch.rsqrt(variance + eps)).to(dtype=x.dtype)
            lora_input = lora_input * ln_w
        # else: TERowParallelLinear — x is already the direct input to the linear

        if self._lora_dropout is not None:
            lora_input = self._lora_dropout(lora_input)

        # adapter: (x @ A.T) @ B.T  =>  (..., in) -> (..., r) -> (..., out)
        adapter_out = F.linear(F.linear(lora_input, self.lora_A), self.lora_B) * self._lora_scaling

        # RowParallel: base output is post-all-reduce, adapter_out is this rank's partial.
        # Sum the partials so base+adapter matches the merged-weight semantics used for
        # rollout. Autograd-aware (fwd all-reduce, bwd identity).
        if getattr(self, "_lora_needs_tp_reduce", False):
            from megatron.core.tensor_parallel.mappings import reduce_from_tensor_model_parallel_region

            adapter_out = reduce_from_tensor_model_parallel_region(adapter_out)

        # PRESERVE the original return SHAPE, not just "is bias present": TE linears
        # under --disable-bias-linear return (output, None) — a 2-tuple whose bias is
        # None. Megatron's get_query_key_value_tensors does `mixed_qkv, _ = apply_module
        # (self.linear_qkv)(hidden_states)`, i.e. it ALWAYS unpacks a 2-tuple. If we
        # collapse the None-bias case to a bare tensor, that unpack raises
        # "too many values to unpack (expected 2)". So mirror _original_forward exactly.
        if result_was_tuple:
            return (output + adapter_out, bias)
        else:
            return output + adapter_out

    import types
    module.forward = types.MethodType(_lora_forward, module)

    logger.info(f"  LoRA injected: {name} | rank={rank} weight={tuple(weight.shape)} pdim={pdim}")


def _get_layer_norm_epsilon(args):
    """Get the RMSNorm epsilon from parsed args."""
    return getattr(args, "norm_epsilon", 1e-6)


# ---------------------------------------------------------------------------
# Merge / Unmerge for weight sync
# ---------------------------------------------------------------------------

def merge_lora_weights(model: nn.Module):
    """In-place merge: weight.data += (α/r) * B @ A for all LoRA modules."""
    for name, mod in model.named_modules():
        if hasattr(mod, "lora_A") and hasattr(mod, "lora_B"):
            A = mod.lora_A.data  # (r, in)
            B = mod.lora_B.data  # (out, r)
            mod.weight.data.add_(B @ A, alpha=mod._lora_scaling)


def unmerge_lora_weights(model: nn.Module):
    """In-place unmerge: weight.data -= (α/r) * B @ A for all LoRA modules."""
    for name, mod in model.named_modules():
        if hasattr(mod, "lora_A") and hasattr(mod, "lora_B"):
            A = mod.lora_A.data
            B = mod.lora_B.data
            mod.weight.data.sub_(B @ A, alpha=mod._lora_scaling)


# ---------------------------------------------------------------------------
# Enable / disable LoRA (for ref/teacher forward)
# ---------------------------------------------------------------------------

def set_lora_enabled(model: nn.Module, enabled: bool):
    """Set global flag to enable/disable LoRA adapter contribution in forward."""
    model._lora_enabled = enabled


def _enable_input_require_grads(model: nn.Module, args) -> bool:
    """Make the embedding output require grad so activation recompute keeps the LoRA graph.

    ROOT CAUSE this fixes (train/grad_norm == 0 while pg_loss != 0):
    With --recompute-granularity full, every decoder layer's forward runs inside a custom
    autograd Function (Megatron's CheckpointFunction / TE checkpoint). Such a Function's
    output only gets a grad_fn if at least one of its TENSOR INPUTS requires grad —
    module PARAMETERS used inside are invisible to that rule. In full fine-tuning the
    embedding is trainable, so hidden_states already requires grad and every checkpointed
    layer is differentiated normally. Under LoRA the embedding is FROZEN, so hidden_states
    arrives with requires_grad=False and the recompute segment is never backwarded ->
    lora_A/lora_B receive no gradient at all, forever (lora_B stays exactly 0, so
    grad wrt lora_A is 0 too: the adapter is a dead branch).

    This is the same fix HF PEFT applies via `model.enable_input_require_grads()` /
    `prepare_model_for_kbit_training` — flip requires_grad on the embedding OUTPUT (a leaf
    tensor, so the flip is legal) and the whole recompute chain becomes differentiable.
    """
    emb = getattr(model, "embedding", None)
    if emb is None:
        logger.warning("LoRA: no .embedding on model; cannot enable input grads "
                       "(recompute may drop adapter gradients)")
        return False

    def _require_grad_hook(_module, _inputs, output):
        if isinstance(output, torch.Tensor):
            if not output.requires_grad and output.is_floating_point() and torch.is_grad_enabled():
                output.requires_grad_(True)
            return output
        if isinstance(output, tuple):
            out = []
            for t in output:
                if isinstance(t, torch.Tensor) and not t.requires_grad and t.is_floating_point() \
                        and torch.is_grad_enabled():
                    t.requires_grad_(True)
                out.append(t)
            return tuple(out)
        return output

    emb.register_forward_hook(_require_grad_hook)
    logger.info("LoRA: enabled input-require-grads on embedding (recompute=%s) — required for "
                "gradients to reach adapters under activation checkpointing",
                getattr(args, "recompute_granularity", None))
    return True


# ---------------------------------------------------------------------------
# Custom model provider entry point
# ---------------------------------------------------------------------------

def custom_model_provider(pre_process: bool = True, post_process: bool = True,
                          vp_stage: Optional[int] = None):
    """Build stock GPTModel then inject LoRA adapters onto target layers.

    This function is loaded by slime via --custom-model-provider-path.
    It accesses `args` from Megatron global state.
    """
    from megatron.training.global_vars import get_args
    # slime versions differ: newer exposes the INNER builder as _get_model_provider_func
    # (get_model_provider_func wraps it with freeze); older (ACLArena) has only
    # get_model_provider_func, which IS the inner builder (freeze is applied separately
    # via wrap_model_provider_with_freeze). Import whichever exists and use it as the
    # stock builder — we do NOT want the freeze wrapper here (freeze is handled below +
    # via --only-train-params-name-list).
    import slime.backends.megatron_utils.model_provider as _mp
    _stock_builder_factory = getattr(_mp, "_get_model_provider_func", None) or _mp.get_model_provider_func

    args = get_args()

    # Build the stock model using the default (non-custom) provider.
    # Temporarily disable custom_model_provider_path to avoid recursion.
    saved_path = args.custom_model_provider_path
    args.custom_model_provider_path = None
    stock_provider = _stock_builder_factory(args, role="actor")
    if vp_stage is not None:
        model = stock_provider(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
    else:
        model = stock_provider(pre_process=pre_process, post_process=post_process)
    args.custom_model_provider_path = saved_path

    # Get norm epsilon for proper LN replication in adapter forward
    eps = _get_layer_norm_epsilon(args)

    # Inject LoRA onto target modules
    target_pattern = "|".join(re.escape(t) for t in LORA_TARGETS)
    injected = 0
    for name, mod in model.named_modules():
        # Match module name suffix against targets
        # e.g. "decoder.layers.0.self_attention.linear_qkv"
        mod_leaf = name.rsplit(".", 1)[-1] if "." in name else name
        if re.fullmatch(target_pattern, mod_leaf):
            if not hasattr(mod, "weight") or mod.weight is None:
                logger.warning(f"  Skipping {name}: no weight attribute")
                continue
            _inject_lora(mod, name, LORA_RANK, LORA_SCALING, LORA_DROPOUT)
            mod.layer_norm_epsilon = eps
            # Store weakref to model root (NOT as nn.Module attribute to avoid recursion)
            mod._lora_model_root_ref = weakref.ref(model)
            injected += 1

    logger.info(f"LoRA injection complete: {injected} modules, rank={LORA_RANK}, "
                f"alpha={LORA_ALPHA}, targets={LORA_TARGETS}")

    # Set initial state
    model._lora_enabled = True

    # CRITICAL for --recompute-granularity full: without this the checkpointed decoder
    # layers get no grad_fn (frozen embedding -> no input requires grad) and the adapters
    # never receive a gradient (train/grad_norm == 0).
    _enable_input_require_grads(model, args)

    # Freeze all base params, unfreeze LoRA
    # (This is also handled by --only-train-params-name-list 'lora_A|lora_B' in the
    # slime freeze wrapper, but doing it here too for safety)
    for pname, param in model.named_parameters():
        if "lora_A" in pname or "lora_B" in pname:
            param.requires_grad = True
        else:
            param.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parameters: total={total_params:,}  trainable(LoRA)={trainable_params:,}  "
                f"ratio={trainable_params/total_params*100:.2f}%")

    return model
