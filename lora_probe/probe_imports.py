import importlib, inspect
for m in ["torch","megatron","megatron.core","sglang","slime","peft"]:
    try:
        mod = importlib.import_module(m)
        print(f"OK   {m:16s} -> {getattr(mod,'__file__','?')}")
    except Exception as e:
        print(f"FAIL {m:16s}: {type(e).__name__}: {str(e)[:80]}")

# Locate the Megatron parallel Linear classes
print("\n=== Megatron tensor_parallel layers ===")
try:
    from megatron.core.tensor_parallel import layers as tp_layers
    print("layers file:", tp_layers.__file__)
    for name in ["ColumnParallelLinear","RowParallelLinear","VocabParallelEmbedding","LinearWithGradAccumulationAndAsyncCommunication"]:
        cls = getattr(tp_layers, name, None)
        print(f"  {name}: {'FOUND' if cls else 'missing'}")
except Exception as e:
    print("tp layers import FAIL:", repr(e))
