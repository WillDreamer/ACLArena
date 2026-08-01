"""Apply the LoRA adapter filter to slime's named_params_and_buffers.

Run on sdb:  python3 /root/patch_common.py

This patches /root/slime/slime/backends/megatron_utils/update_weight/common.py
to filter out parameters whose name contains 'lora_A' or 'lora_B' from the
weight-sync iteration. The filter is a 3-line addition to both the _global and
_vanilla iterators.
"""
import re

COMMON_PY = "/root/slime/slime/backends/megatron_utils/update_weight/common.py"

with open(COMMON_PY, "r") as f:
    content = f.read()

# Check if already patched
if "lora_A" in content:
    print("Already patched (lora_A found in common.py)")
    exit(0)

# Patch 1: _named_params_and_buffers_vanilla — filter in the for loop
# Original:  for name, param in model_module.named_parameters():
#                yield _compute_fqn(name), param
# Add filter before yield
old_vanilla = '''        for name, param in model_module.named_parameters():
            yield _compute_fqn(name), param'''
new_vanilla = '''        for name, param in model_module.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                continue
            yield _compute_fqn(name), param'''

if old_vanilla not in content:
    print("ERROR: cannot find vanilla pattern to patch")
    exit(1)
content = content.replace(old_vanilla, new_vanilla, 1)

# Patch 2: _named_params_and_buffers_global — filter in the for loop
# Original:  for name, param in model_module.named_parameters():
#                # for model without ddp wrap
#                if not name.startswith("module.module."):
old_global = '''        for name, param in model_module.named_parameters():
            # for model without ddp wrap
            if not name.startswith("module.module."):'''
new_global = '''        for name, param in model_module.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                continue
            # for model without ddp wrap
            if not name.startswith("module.module."):'''

if old_global not in content:
    print("ERROR: cannot find global pattern to patch")
    exit(1)
content = content.replace(old_global, new_global, 1)

with open(COMMON_PY, "w") as f:
    f.write(content)

print(f"Patched {COMMON_PY} — added lora_A/lora_B filter to both iterators")
