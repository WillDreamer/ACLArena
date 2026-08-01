"""Build Qwen3-8B GPTModel on 1 GPU via slime's full parse_args + provider path
and dump: transformer_impl, module tree, linear classes, param names.

Run:
  cd /root/slime && CUDA_VISIBLE_DEVICES=0 \
    torchrun --nproc_per_node=1 --master_port=29555 /root/probe_model_tree.py
"""
import gc, os, sys, json, tempfile  # noqa

sys.path.insert(0, "/root/slime")
sys.path.insert(0, "/root/slime/examples/search-r1")
sys.path.insert(0, "/root/Megatron-LM")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
import torch.distributed as dist

# ----- Create a fake HF dir with Qwen3 config.json (AutoConfig.from_pretrained) -----
_hf_dir = tempfile.mkdtemp(prefix="qwen3_8b_fake_hf_")
_config = {
    "architectures": ["Qwen3ForCausalLM"],
    "model_type": "qwen3",
    "hidden_size": 4096,
    "intermediate_size": 12288,
    "num_hidden_layers": 36,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "max_position_embeddings": 131072,
    "vocab_size": 151936,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000.0,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "sliding_window": None,
    "attention_dropout": 0.0,
    "use_sliding_window": False,
}
with open(os.path.join(_hf_dir, "config.json"), "w") as f:
    json.dump(_config, f)

# ----- Build argv that slime's parse_args will accept -----
MODEL_ARGS = [
    "--swiglu",
    "--num-layers", "36",
    "--hidden-size", "4096",
    "--ffn-hidden-size", "12288",
    "--num-attention-heads", "32",
    "--group-query-attention",
    "--num-query-groups", "8",
    "--use-rotary-position-embeddings",
    "--disable-bias-linear",
    "--normalization", "RMSNorm",
    "--norm-epsilon", "1e-6",
    "--rotary-base", "1000000",
    "--vocab-size", "151936",
    "--kv-channels", "128",
    "--qk-layernorm",
    "--untie-embeddings-and-output-weights",
    "--tensor-model-parallel-size", "1",
    "--pipeline-model-parallel-size", "1",
    "--context-parallel-size", "1",
    "--hf-checkpoint", _hf_dir,
    "--seq-length", "4096",
    "--max-position-embeddings", "4096",
    "--micro-batch-size", "1",
    "--global-batch-size", "1",
    "--attention-backend", "flash",
    "--bf16",
    "--save-interval", "1",
    "--save", "/tmp/probe_dummy",
    "--megatron-to-hf-mode", "raw",
    "--sequence-parallel",
    # Force slime to skip sglang parse:
    "--debug-train-only",
    # Need actor-num-nodes for world_size calc:
    "--actor-num-nodes", "1",
    "--actor-num-gpus-per-node", "1",
    # slime required args:
    "--rollout-batch-size", "1",
    "--n-samples-per-prompt", "1",
    "--num-rollout", "1",
    "--rollout-max-response-len", "128",
    "--input-key", "prompt",
    "--advantage-estimator", "grpo",
]
sys.argv = ["probe"] + MODEL_ARGS

from slime.utils.arguments import parse_args  # noqa

args = parse_args()

print("=" * 70)
print("transformer_impl        =", getattr(args, "transformer_impl", "?"))
print("spec                    =", getattr(args, "spec", None))
print("padded_vocab_size       =", args.padded_vocab_size)
print("num_experts             =", getattr(args, "num_experts", None))
print("qk_layernorm            =", getattr(args, "qk_layernorm", None))
print("fp8_param_gather        =", getattr(args, "fp8_param_gather", None))
print("megatron_to_hf_mode     =", getattr(args, "megatron_to_hf_mode", None))
print("=" * 70)

# ----- Initialize distributed + megatron -----
local_rank = int(os.getenv("LOCAL_RANK", "0"))
torch.cuda.set_device(local_rank)
if not dist.is_initialized():
    dist.init_process_group(backend="nccl", world_size=1, rank=0,
                            device_id=torch.device(f"cuda:{local_rank}"))

from slime.backends.megatron_utils.initialize import init  # noqa

init(args)

# ----- Build model (same as slime does in actor.py) -----
from megatron.core.enums import ModelType  # noqa
from megatron.training.training import get_model  # noqa
from slime.backends.megatron_utils.model_provider import get_model_provider_func  # noqa

model_list = get_model(get_model_provider_func(args, role="actor"), ModelType.encoder_or_decoder, wrap_with_ddp=False)
model = model_list[0]

print("\n" + "=" * 70)
print("MODEL TYPE:", type(model).__module__ + "." + type(model).__name__)
print("=" * 70)

# ----- Dump linear-like modules -----
from collections import Counter, OrderedDict  # noqa
import inspect  # noqa

cls_counter = Counter()
linear_examples = OrderedDict()
for name, mod in model.named_modules():
    cls = type(mod).__name__
    if any(k in cls.lower() for k in ("linear", "columnparallel", "rowparallel")):
        cls_counter[cls] += 1
        if cls not in linear_examples:
            linear_examples[cls] = (name, mod)

print("\n=== LINEAR CLASS COUNTS ===")
for cls, cnt in cls_counter.most_common():
    print(f"  {cnt:4d}  {cls}  ({type(linear_examples[cls][1]).__module__})")

print("\n=== EXAMPLE OF EACH CLASS ===")
for cls, (path, mod) in linear_examples.items():
    print(f"\n--- {cls}  @ {path[:80]}")
    for attr in ("input_size", "output_size", "input_size_per_partition",
                 "output_size_per_partition", "skip_bias_add",
                 "sequence_parallel", "te_return_bias", "gather_output"):
        if hasattr(mod, attr):
            print(f"      .{attr:30s} = {getattr(mod, attr)}")
    w = getattr(mod, "weight", None)
    if w is not None:
        print(f"      .weight shape={tuple(w.shape)} dtype={w.dtype} "
              f"tp={getattr(w,'tensor_model_parallel',None)} pdim={getattr(w,'partition_dim',None)}")
    else:
        print("      .weight = None")
    b = getattr(mod, "bias", None)
    if b is not None and hasattr(b, "shape"):
        print(f"      .bias shape={tuple(b.shape)}")
    # children (are they submodules? e.g. TEColumnParallelLinear has a Linear inside)
    children = list(mod.named_children())
    if children:
        print(f"      children: {[n for n,_ in children]}")
    # forward sig
    try:
        sig = inspect.signature(mod.forward)
        params = [str(p) for p in sig.parameters.values()]
        print(f"      forward({', '.join(params[:6])}{'...' if len(params)>6 else ''})")
    except Exception:
        pass

# ----- Layer 0 named_parameters -----
print("\n=== NAMED_PARAMETERS (layer 0 only) ===")
for name, p in model.named_parameters():
    if ".layers.0." in name:
        print(f"  {name:80s} shape={str(tuple(p.shape)):>20s}  req_grad={p.requires_grad}"
              f"  tp={getattr(p,'tensor_model_parallel',None)} pdim={getattr(p,'partition_dim',None)}")

print(f"\n=== TOTAL PARAMS: {sum(p.numel() for p in model.parameters()):,} ===")
print("=== DONE ===")
dist.destroy_process_group()
