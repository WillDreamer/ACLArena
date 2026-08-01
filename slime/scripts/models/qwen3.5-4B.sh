MODEL_ARGS=(
   --spec "slime_plugins.models.qwen3_5" "get_qwen3_5_spec"

   --disable-bias-linear
   --qk-layernorm
   --group-query-attention
   --num-attention-heads 16
   --num-query-groups 4
   --kv-channels 256
   --num-layers 32
   --hidden-size 2560
   --ffn-hidden-size 9216
   --use-gated-attention

   --normalization RMSNorm
   --apply-layernorm-1p
   --position-embedding-type rope
   --norm-epsilon 1e-6
   --rotary-percent 0.25
   --swiglu
   # Qwen3.5-4B-Base ties input/output embeddings (tie_word_embeddings: true),
   # so do NOT pass --untie-embeddings-and-output-weights (unlike the 9B/27B).
   --vocab-size 248320

   --rotary-base 10000000

   # qwen3.5 specific
   --attention-output-gate
)
