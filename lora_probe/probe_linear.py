import inspect
from megatron.core.tensor_parallel import layers as L

for name in ["ColumnParallelLinear", "RowParallelLinear"]:
    cls = getattr(L, name)
    print("="*70)
    print(name)
    print("  __init__ signature:")
    try:
        print("   ", inspect.signature(cls.__init__))
    except Exception as e:
        print("    <sig fail>", e)
    print("  forward signature:")
    try:
        print("   ", inspect.signature(cls.forward))
    except Exception as e:
        print("    <sig fail>", e)
    print("  MRO:", [c.__name__ for c in cls.__mro__])
