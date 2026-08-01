"""LoRA lifecycle hooks for slime integration.

This module provides:
  1. merge_lora_weights / unmerge_lora_weights — in-place merge for sync
  2. is_lora_param(name) — filter predicate for named_params_and_buffers
  3. patch_actor_for_lora(actor) — monkey-patches actor's backup + update_weights
     with merge/unmerge around the critical sections
  4. install_global_lora_hooks() — patches TensorBackuper class so that any future
     instance automatically merges/unmerges around backup("actor")
  5. set_lora_enabled(model, flag) — toggle adapter contribution in forward

These are imported by lora_model_provider.py and by the RL run script's
--custom-megatron-init-path if needed.
"""

import logging
from functools import wraps

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Predicate: is this a LoRA adapter parameter?
# ---------------------------------------------------------------------------

LORA_PARAM_NAMES = ("lora_A", "lora_B")


def is_lora_param(name: str) -> bool:
    """Return True if `name` (from named_parameters) is a LoRA adapter param."""
    # Fast check: any segment of the dotted name is lora_A or lora_B
    return any(seg in LORA_PARAM_NAMES for seg in name.split("."))


# ---------------------------------------------------------------------------
# Merge / Unmerge
# ---------------------------------------------------------------------------

@torch.no_grad()
def merge_lora_weights(model: nn.Module) -> int:
    """In-place merge: weight.data += (α/r) * B @ A for all LoRA-injected modules.
    Stores the exact delta on the module for perfect cancellation in unmerge.
    Returns number of modules merged.
    """
    count = 0
    for _name, mod in model.named_modules():
        if hasattr(mod, "lora_A") and hasattr(mod, "lora_B"):
            A = mod.lora_A.data  # (r, in)
            B = mod.lora_B.data  # (out, r)
            scaling = getattr(mod, "_lora_scaling", 1.0)
            # Compute in fp32 for precision, cast to weight dtype, store for exact unmerge
            delta = (B.float() @ A.float() * scaling).to(mod.weight.dtype)
            mod.weight.data.add_(delta)
            mod._lora_merged_delta = delta  # store for exact cancellation
            count += 1
    return count


@torch.no_grad()
def unmerge_lora_weights(model: nn.Module) -> int:
    """In-place unmerge: subtract the EXACT delta stored during merge.
    Falls back to recomputing if no stored delta (e.g. called without prior merge).
    """
    count = 0
    for _name, mod in model.named_modules():
        if hasattr(mod, "lora_A") and hasattr(mod, "lora_B"):
            delta = getattr(mod, "_lora_merged_delta", None)
            if delta is None:
                # Fallback: recompute (may have small rounding error in bf16)
                A = mod.lora_A.data
                B = mod.lora_B.data
                scaling = getattr(mod, "_lora_scaling", 1.0)
                delta = (B.float() @ A.float() * scaling).to(mod.weight.dtype)
            mod.weight.data.sub_(delta)
            mod._lora_merged_delta = None  # clear stored delta
            count += 1
    return count


# ---------------------------------------------------------------------------
# Enable/Disable for ref model forward
# ---------------------------------------------------------------------------

def set_lora_enabled(model: nn.Module, enabled: bool):
    """Toggle LoRA adapter forward contribution globally on the model.

    CRITICAL: the adapter's `_lora_forward` reads the flag off the object its
    `_lora_model_root_ref` weakref points to — which is the BARE GPTModel returned
    by custom_model_provider, i.e. the INNERMOST module. The training model list
    holds it wrapped as DDP(Float16Module(GPTModel)) (or deeper). Setting the flag
    only on the outer wrapper(s) leaves GPTModel._lora_enabled untouched, so the
    adapter never actually disables for the ref/teacher forward (KL anchor silently
    ~0). Walk the ENTIRE `.module` chain and set the flag on every level so the flag
    the adapter reads is always correct regardless of wrapping depth.
    """
    obj = model
    seen = 0
    while obj is not None:
        obj._lora_enabled = enabled
        seen += 1
        obj = getattr(obj, "module", None)
        if seen > 8:  # guard against pathological cycles
            break


# ---------------------------------------------------------------------------
# Out-of-band adapter checkpoint save
# ---------------------------------------------------------------------------
# WHY: The monkey-patched lora_A/lora_B params are registered on TE module
# instances, so they DO appear in module.state_dict() and get picked up by
# Megatron's torch_dist save. BUT TE's sharded_state_dict only declares TP
# sharding axes for {"weight","bias"} via make_sharded_tensors_for_checkpoint;
# every other key (i.e. lora_A/lora_B) is treated as REPLICATED. Under TP>1 the
# 4 ranks each hold a DIFFERENT shard of the adapter yet all claim to be the same
# replicated tensor -> the dist-checkpoint dedups to ONE rank's slice (the
# observed (36,1536,16) 1/4 shape, only in TP-rank0 files) AND the distributed
# optimizer state for those mis-declared params fails to round-trip
# (ShardedTensor.flattened_range unsupported) -> reloaded lora_B == 0,
# exp_avg_sq == 0. Training itself is fine (verified: live lora_B trains, opt
# state accumulates); ONLY the torch_dist serialization of the injected params is
# broken. So we bypass it: gather the LIVE trained adapter across TP into full
# unsharded shapes and write a plain safetensors sidecar next to each iter dir.

def _tp_shard_dim(param) -> int:
    """Return the TP partition dim of a LoRA factor, or -1 if replicated.
    Set in lora_model_provider._inject_lora:
      ColumnParallel (linear_qkv, linear_fc1): lora_B sharded dim0, lora_A replicated.
      RowParallel   (linear_proj, linear_fc2): lora_A sharded dim1, lora_B replicated.
    """
    return int(getattr(param, "partition_dim", -1))


@torch.no_grad()
def _tp_gather_full(t: torch.Tensor, partition_dim: int, tp_group) -> torch.Tensor:
    """All-gather a TP-sharded tensor along its partition dim -> full tensor (cpu).
    Replicated tensors (partition_dim<0) are returned as-is (local copy)."""
    import torch.distributed as dist
    tp = dist.get_world_size(group=tp_group)
    if tp == 1 or partition_dim < 0:
        return t.detach().contiguous().to("cpu")
    parts = [torch.empty_like(t) for _ in range(tp)]
    dist.all_gather(parts, t.detach().contiguous(), group=tp_group)
    return torch.cat(parts, dim=partition_dim).detach().contiguous().to("cpu")


def _normalize_lora_module_name(name: str) -> str:
    """Strip DDP/Float16 wrapper prefixes so the key starts at 'decoder.'.
    e.g. 'module.module.decoder.layers.0.self_attention.linear_qkv'
      -> 'decoder.layers.0.self_attention.linear_qkv'.
    Matches the un-stacked base weight naming used by convert_torch_dist_to_hf."""
    idx = name.find("decoder.")
    return name[idx:] if idx >= 0 else name


@torch.no_grad()
def save_lora_adapter(model_list, save_dir: str, iteration: int) -> None:
    """Gather all LoRA adapters across TP into full shapes and write
    {save_dir}/iter_{iteration:07d}/lora_adapter.safetensors (+ lora_meta.json).

    Collective: EVERY rank must call this (all_gather is collective); only the
    single tp0/dp0/cp0/pp0 writer rank writes the file. Assumes PP=1 (the LoRA
    search run forces PP=1) so the writer rank holds every layer's modules.
    """
    import os
    import json
    import torch.distributed as dist

    try:
        from megatron.core import parallel_state as mpu
    except Exception:  # pragma: no cover
        from megatron.core import mpu  # type: ignore

    from safetensors.torch import save_file

    logger.info("save_lora_adapter: ENTER save_dir=%s iter=%s n_model_chunks=%d",
                save_dir, iteration, len(model_list))
    tp_group = mpu.get_tensor_model_parallel_group()

    adapter: dict[str, torch.Tensor] = {}
    scaling: dict[str, float] = {}
    for model in model_list:
        for name, mod in model.named_modules():
            if hasattr(mod, "lora_A") and hasattr(mod, "lora_B"):
                key = _normalize_lora_module_name(name)
                A_full = _tp_gather_full(mod.lora_A.data, _tp_shard_dim(mod.lora_A), tp_group)
                B_full = _tp_gather_full(mod.lora_B.data, _tp_shard_dim(mod.lora_B), tp_group)
                adapter[key + ".lora_A"] = A_full
                adapter[key + ".lora_B"] = B_full
                scaling[key] = float(getattr(mod, "_lora_scaling", 1.0))

    pp_rank = mpu.get_pipeline_model_parallel_rank()
    pp_size = mpu.get_pipeline_model_parallel_world_size()
    tp_rank = mpu.get_tensor_model_parallel_rank()
    try:
        dp_rank = mpu.get_data_parallel_rank(with_context_parallel=True)
    except TypeError:
        dp_rank = mpu.get_data_parallel_rank()

    is_writer = (tp_rank == 0 and dp_rank == 0 and pp_rank == 0)
    if is_writer:
        if pp_size > 1:
            logger.warning(
                "save_lora_adapter: PP=%d>1 not supported by single-file writer; "
                "adapter for non-first pipeline stages will be MISSING.", pp_size
            )
        outdir = os.path.join(save_dir, f"iter_{iteration:07d}")
        os.makedirs(outdir, exist_ok=True)
        fpath = os.path.join(outdir, "lora_adapter.safetensors")
        save_file(adapter, fpath)
        with open(os.path.join(outdir, "lora_meta.json"), "w") as f:
            json.dump({"iteration": int(iteration), "scaling": scaling,
                       "num_adapter_tensors": len(adapter)}, f, indent=2)
        logger.info("save_lora_adapter: wrote %d tensors -> %s", len(adapter), fpath)

    if dist.is_initialized():
        dist.barrier()


# ---------------------------------------------------------------------------
# Patch actor for merge-around-sync
# ---------------------------------------------------------------------------

def patch_actor_for_lora(actor):
    """Monkey-patch actor's weights_backuper.backup and weight_updater.update_weights
    to call merge/unmerge around the critical sync sections.

    Also patches _switch_model to disable LoRA when switching to ref/teacher.

    Call this AFTER actor.__init__ completes (from the run script or a custom init hook).
    """
    model_list = actor.model  # list of model modules (vp stages)

    def _merge_all():
        for m in model_list:
            merge_lora_weights(m)

    def _unmerge_all():
        for m in model_list:
            unmerge_lora_weights(m)

    # --- Patch backup ---
    original_backup = actor.weights_backuper.backup

    @wraps(original_backup)
    def patched_backup(tag: str):
        if tag == "actor":
            _merge_all()
            original_backup(tag)
            _unmerge_all()
        else:
            # ref/teacher/old_actor backups: just snapshot as-is (unmerged base)
            original_backup(tag)

    actor.weights_backuper.backup = patched_backup

    # --- Patch update_weights (for disagg: reads live model) ---
    original_update_weights = actor.weight_updater.update_weights

    @wraps(original_update_weights)
    def patched_update_weights():
        # For colocate: reads CPU backup (already merged by patched_backup)
        # For disagg: reads live model -> merge first
        if not getattr(actor.args, "colocate", False):
            _merge_all()
        original_update_weights()
        if not getattr(actor.args, "colocate", False):
            _unmerge_all()

    actor.weight_updater.update_weights = patched_update_weights

    # --- Patch _switch_model to disable LoRA during ref/teacher ---
    original_switch = actor._switch_model

    @wraps(original_switch)
    def patched_switch(target_tag: str):
        original_switch(target_tag)
        # When switched to ref/teacher, disable LoRA forward addon
        if target_tag in ("ref", "teacher"):
            for m in model_list:
                set_lora_enabled(m, False)
        else:
            for m in model_list:
                set_lora_enabled(m, True)

    actor._switch_model = patched_switch

    logger.info("LoRA hooks patched onto actor (backup, update_weights, _switch_model)")


# ---------------------------------------------------------------------------
# Global class-level patch (alternative to per-actor patching)
# ---------------------------------------------------------------------------

_GLOBAL_HOOKS_INSTALLED = False


def install_global_lora_hooks():
    """Monkey-patch TensorBackuper._TensorBackuperNormal.backup at CLASS level
    so that ANY future instance merges/unmerges around backup("actor").

    Also patches UpdateWeightFromDistributed.update_weights for disagg.

    Call this once at module import (from lora_model_provider) — it runs before
    actor.__init__ creates the backuper.
    """
    global _GLOBAL_HOOKS_INSTALLED
    if _GLOBAL_HOOKS_INSTALLED:
        return
    _GLOBAL_HOOKS_INSTALLED = True

    from slime.utils.tensor_backper import _TensorBackuperNormal

    _orig_backup = _TensorBackuperNormal.backup

    @wraps(_orig_backup)
    def _hooked_backup(self, tag: str):
        if tag == "actor":
            # Merge all LoRA modules reachable from the source_getter
            # The source_getter returns named_params_and_buffers(args, model)
            # We need the model list. Get it from the source closure.
            model_list = _get_model_from_source_getter(self._source_getter)
            if model_list:
                for m in model_list:
                    merge_lora_weights(m)
                _orig_backup(self, tag)
                for m in model_list:
                    unmerge_lora_weights(m)
            else:
                _orig_backup(self, tag)
        else:
            _orig_backup(self, tag)

    _TensorBackuperNormal.backup = _hooked_backup

    # Also patch the distributed weight updater for disagg mode
    from slime.backends.megatron_utils.update_weight.update_weight_from_distributed import (
        UpdateWeightFromDistributed,
    )

    _orig_uw = UpdateWeightFromDistributed.update_weights

    @wraps(_orig_uw)
    def _hooked_update_weights(self):
        # self.model is the model list
        for m in self.model:
            merge_lora_weights(m)
        _orig_uw(self)
        for m in self.model:
            unmerge_lora_weights(m)

    UpdateWeightFromDistributed.update_weights = _hooked_update_weights

    # Patch actor._switch_model so LoRA adapters are DISABLED during ref/teacher
    # forward. Critical: base weights are frozen, so actor-base == ref-base always.
    # If the adapter stayed enabled while the ref snapshot is restored, ref logprobs
    # would equal actor logprobs -> KL-to-ref == 0, silently killing the KL anchor.
    from slime.backends.megatron_utils.actor import MegatronTrainRayActor

    _orig_switch = MegatronTrainRayActor._switch_model

    @wraps(_orig_switch)
    def _hooked_switch_model(self, target_tag: str):
        _orig_switch(self, target_tag)
        enabled = target_tag not in ("ref", "teacher")
        for m in self.model:
            set_lora_enabled(m, enabled)

    MegatronTrainRayActor._switch_model = _hooked_switch_model

    # Alongside the (broken-for-LoRA) torch_dist checkpoint, ALSO write a
    # full-shape out-of-band lora_adapter.safetensors into the same iter_XXXX dir.
    # This is the file we merge+publish from — the torch_dist copy of the adapter
    # is unusable (see save_lora_adapter docstring).
    #
    # We patch the module-level `save` NAME that the actor actually calls. The
    # actor does `from .model import ... save` (a bound name in the actor module's
    # namespace), so we must rebind `actor_mod.save`, NOT `model.save` (the latter
    # is already dereferenced). save() is called IN-PROCESS from save_model, so no
    # Ray-dispatch ambiguity; it receives (iteration, model, ...) directly and we
    # read save_dir from get_args() (no dependence on self.role/self.args).
    import slime.backends.megatron_utils.actor as _actor_mod

    _orig_save = _actor_mod.save

    @wraps(_orig_save)
    def _hooked_save(iteration, model, optimizer, opt_param_scheduler):
        _orig_save(iteration, model, optimizer, opt_param_scheduler)
        try:
            from megatron.training.global_vars import get_args
            save_dir = getattr(get_args(), "save", None)
            logger.info("LORA_ADAPTER_HOOK: save() iteration=%s save_dir=%s", iteration, save_dir)
            if save_dir:
                # slime save() already uses `iteration` as the iter_XXXX number.
                save_lora_adapter(model, save_dir, iteration)
        except Exception as e:  # never let adapter-save kill the training job
            import traceback
            logger.warning("save_lora_adapter failed (non-fatal): %s: %s\n%s",
                           type(e).__name__, e, traceback.format_exc())

    _actor_mod.save = _hooked_save

    logger.info("Global LoRA hooks installed "
                "(TensorBackuper.backup, UpdateWeightFromDistributed.update_weights, "
                "actor._switch_model, actor.save[+adapter])")


def _get_model_from_source_getter(source_getter):
    """Extract the model list from the source_getter closure.
    The source_getter is typically:
      lambda: named_params_and_buffers(self.args, self.model, ...)
    where 'self' is the actor instance captured in the closure.
    """
    try:
        closure = source_getter.__closure__
        if closure is None:
            return None
        for cell in closure:
            try:
                obj = cell.cell_contents
            except ValueError:
                continue
            # Direct model list
            if isinstance(obj, (list, tuple)) and len(obj) > 0:
                if isinstance(obj[0], nn.Module):
                    return obj
            # Actor instance (has .model attribute which is the model list)
            if hasattr(obj, "model") and hasattr(obj, "weights_backuper"):
                model = getattr(obj, "model", None)
                if isinstance(model, (list, tuple)):
                    return model
    except Exception:
        pass
    return None
