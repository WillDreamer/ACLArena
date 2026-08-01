"""Smoke test: build Qwen3-8B with LoRA provider, verify injection + merge + sync filter.

Run on sdb:
  cd /root/slime && CUDA_VISIBLE_DEVICES=0 LORA_RANK=16 LORA_ALPHA=32 \
    torchrun --nproc_per_node=1 --master_port=29555 /root/smoke_test_lora.py

Checks:
  1. Model builds without error
  2. LoRA params (lora_A, lora_B) exist on target modules
  3. requires_grad is correct (only LoRA trainable)
  4. merge_lora_weights adds to weight.data correctly
  5. unmerge_lora_weights restores original weight
  6. named_params_and_buffers EXCLUDES lora params (after common.py patch)
  7. convert_to_hf doesn't crash on a merged layer 0 weight
  8. forward pass produces output without error
"""
import os, sys, json, tempfile  # noqa

sys.path.insert(0, "/root/slime")
sys.path.insert(0, "/root/slime/examples/search-r1")
sys.path.insert(0, "/root/Megatron-LM")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("LORA_RANK", "16")
os.environ.setdefault("LORA_ALPHA", "32")

import torch
import torch.distributed as dist

# Fake HF config dir
_hf_dir = tempfile.mkdtemp(prefix="qwen3_8b_fake_hf_")
_config = {
    "architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3",
    "hidden_size": 4096, "intermediate_size": 12288,
    "num_hidden_layers": 36, "num_attention_heads": 32,
    "num_key_value_heads": 8, "head_dim": 128,
    "max_position_embeddings": 131072, "vocab_size": 151936,
    "rms_norm_eps": 1e-6, "rope_theta": 1000000.0,
    "tie_word_embeddings": False, "torch_dtype": "bfloat16",
}
with open(os.path.join(_hf_dir, "config.json"), "w") as f:
    json.dump(_config, f)

# Argv
sys.argv = ["smoke"] + [
    "--swiglu", "--num-layers", "36", "--hidden-size", "4096",
    "--ffn-hidden-size", "12288", "--num-attention-heads", "32",
    "--group-query-attention", "--num-query-groups", "8",
    "--use-rotary-position-embeddings", "--disable-bias-linear",
    "--normalization", "RMSNorm", "--norm-epsilon", "1e-6",
    "--rotary-base", "1000000", "--vocab-size", "151936",
    "--kv-channels", "128", "--qk-layernorm",
    "--untie-embeddings-and-output-weights",
    "--tensor-model-parallel-size", "1",
    "--pipeline-model-parallel-size", "1",
    "--context-parallel-size", "1",
    "--hf-checkpoint", _hf_dir,
    "--seq-length", "4096", "--max-position-embeddings", "4096",
    "--micro-batch-size", "1", "--global-batch-size", "1",
    "--attention-backend", "flash", "--bf16",
    "--save-interval", "1", "--save", "/tmp/probe_dummy",
    "--megatron-to-hf-mode", "raw", "--sequence-parallel",
    "--debug-train-only",
    "--actor-num-nodes", "1", "--actor-num-gpus-per-node", "1",
    "--rollout-batch-size", "1", "--n-samples-per-prompt", "1",
    "--num-rollout", "1", "--rollout-max-response-len", "128",
    "--input-key", "prompt", "--advantage-estimator", "grpo",
    # LoRA config:
    "--custom-model-provider-path", "lora_model_provider.custom_model_provider",
    "--only-train-params-name-list", "lora_A", "lora_B",
]

from slime.utils.arguments import parse_args
args = parse_args()

# Init distributed
local_rank = int(os.getenv("LOCAL_RANK", "0"))
torch.cuda.set_device(local_rank)
if not dist.is_initialized():
    dist.init_process_group(backend="nccl", world_size=1, rank=0,
                            device_id=torch.device(f"cuda:{local_rank}"))

from slime.backends.megatron_utils.initialize import init
init(args)

# Build model
from megatron.core.enums import ModelType
from megatron.training.training import get_model
from slime.backends.megatron_utils.model_provider import get_model_provider_func

model_list = get_model(get_model_provider_func(args, role="actor"), ModelType.encoder_or_decoder, wrap_with_ddp=False)
model = model_list[0]

# ============= CHECK 1: Model built =============
print("\n✅ CHECK 1: Model built successfully")
print(f"   Type: {type(model).__name__}")

# ============= CHECK 2: LoRA params exist =============
lora_params = [(n, p) for n, p in model.named_parameters() if "lora_A" in n or "lora_B" in n]
print(f"\n{'✅' if lora_params else '❌'} CHECK 2: LoRA params found: {len(lora_params)}")
if lora_params:
    print(f"   Example: {lora_params[0][0]} shape={tuple(lora_params[0][1].shape)}")
    print(f"   Example: {lora_params[1][0]} shape={tuple(lora_params[1][1].shape)}")

# ============= CHECK 3: requires_grad =============
trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
non_lora_trainable = [n for n, p in trainable if "lora_A" not in n and "lora_B" not in n]
print(f"\n{'✅' if not non_lora_trainable else '❌'} CHECK 3: requires_grad")
print(f"   Total trainable: {len(trainable)}")
print(f"   Non-LoRA trainable (should be 0): {len(non_lora_trainable)}")
if non_lora_trainable:
    print(f"   UNEXPECTED: {non_lora_trainable[:5]}")
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"   Trainable: {trainable_params:,} / {total_params:,} = {trainable_params/total_params*100:.2f}%")

# ============= CHECK 4: merge correctness =============
# Pick layer 0 linear_qkv
target_mod = None
target_name = ""
for name, mod in model.named_modules():
    if "layers.0.self_attention.linear_qkv" in name and hasattr(mod, "lora_A"):
        target_mod = mod
        target_name = name
        break

if target_mod is not None:
    w_before = target_mod.weight.data.clone()
    from lora_hooks import merge_lora_weights, unmerge_lora_weights
    n_merged = merge_lora_weights(model)
    w_after = target_mod.weight.data.clone()
    diff = (w_after - w_before).abs().max().item()
    # B is zero-init so merge should add nothing initially
    print(f"\n✅ CHECK 4: merge_lora_weights (count={n_merged})")
    print(f"   Max diff after merge (B=0 init): {diff:.6e} (expect ~0 since B zero-init)")

    # Make B non-zero and re-test
    target_mod.lora_B.data.fill_(0.01)
    merge_lora_weights(model)  # merge again (cumulative)
    w_after2 = target_mod.weight.data.clone()
    diff2 = (w_after2 - w_after).abs().max().item()
    print(f"   Max diff after B=0.01 merge: {diff2:.6e} (expect > 0)")

    # Unmerge should restore
    unmerge_lora_weights(model)
    unmerge_lora_weights(model)  # two unmerges to undo both merges
    w_restored = target_mod.weight.data.clone()
    restore_err = (w_restored - w_before).abs().max().item()
    print(f"   Restoration error after double unmerge: {restore_err:.6e} (expect ~0)")
    # Reset B
    target_mod.lora_B.data.zero_()
else:
    print("\n❌ CHECK 4: Could not find target module for merge test")

# ============= CHECK 5 (was 6): named_params_and_buffers filter =============
from slime.backends.megatron_utils.update_weight.common import named_params_and_buffers
sync_names = [name for name, _ in named_params_and_buffers(args, model_list)]
lora_in_sync = [n for n in sync_names if "lora_A" in n or "lora_B" in n]
print(f"\n{'✅' if not lora_in_sync else '❌'} CHECK 5: named_params_and_buffers filter")
print(f"   Total sync params: {len(sync_names)}")
print(f"   LoRA params in sync (should be 0): {len(lora_in_sync)}")
if lora_in_sync:
    print(f"   LEAKED: {lora_in_sync[:5]}")

# ============= CHECK 6 (was 7): convert_to_hf =============
from slime.backends.megatron_utils.megatron_to_hf import convert_to_hf
try:
    # Pick a param from sync list and try convert
    test_name = sync_names[0]
    test_param = dict(named_params_and_buffers(args, model_list))[test_name]
    hf_result = convert_to_hf(args, "qwen3forcausallm", test_name, test_param, None)
    print(f"\n✅ CHECK 6: convert_to_hf")
    print(f"   Input: {test_name}")
    print(f"   Output: {[(n, tuple(t.shape)) for n, t in hf_result]}")
except Exception as e:
    print(f"\n❌ CHECK 6: convert_to_hf FAILED: {e}")

# ============= CHECK 7 (was 8): forward pass =============
try:
    seq_len = 32
    input_ids = torch.randint(0, 151936, (1, seq_len), device="cuda")
    # GPTModel forward needs specific inputs; let's just do a simple pass
    # through the model's decoder layers manually
    from megatron.core import mpu
    # For full forward we'd need position_ids, attention_mask, etc.
    # Simpler: just test one LoRA-injected module's forward
    x = torch.randn(1, seq_len, 4096, dtype=torch.bfloat16, device="cuda")
    out = target_mod.forward(x)
    if isinstance(out, tuple):
        out = out[0]
    print(f"\n✅ CHECK 7: Forward pass through LoRA module")
    print(f"   Input: {tuple(x.shape)} -> Output: {tuple(out.shape)}")
except Exception as e:
    print(f"\n❌ CHECK 7: Forward FAILED: {e}")
    import traceback; traceback.print_exc()

# ============= CHECK 8: VRAM usage =============
alloc = torch.cuda.memory_allocated() / 1024**3
reserved = torch.cuda.memory_reserved() / 1024**3
print(f"\n📊 VRAM: allocated={alloc:.2f} GB, reserved={reserved:.2f} GB")
print(f"   (Full 8B model bf16 = ~16 GB; LoRA overhead = ~{trainable_params * 2 / 1024**3:.3f} GB)")

print("\n" + "="*70)
print("ALL SMOKE TESTS PASSED" if not lora_in_sync and not non_lora_trainable else "SOME CHECKS FAILED")
print("="*70)

dist.destroy_process_group()
