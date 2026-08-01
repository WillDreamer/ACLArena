#!/bin/bash

set -ex

ROOT_DIR=/data/user

# cd $ROOT_DIR
# git clone https://github.com/PeterGriffinJin/Search-R1.git
# cd Search-R1

# conda create -n searchr1 python=3.9 -y
# conda activate searchr1
# pip install -e .

# # Set your working directory
# WORK_DIR=$ROOT_DIR/Search-R1
# LOCAL_DIR=$WORK_DIR/data/nq_hotpotqa_train

# # Process multiple dataset search format train file
# DATA=nq,hotpotqa
# python $WORK_DIR/scripts/data_process/qa_search_train_merge.py \
#     --local_dir $LOCAL_DIR \
#     --data_sources $DATA

# # (Optional) Process multiple dataset search format test file
# # Note: the final file is not shuffled
# DATA=nq,triviaqa,popqa,hotpotqa,2wikimultihopqa,musique,bamboogle
# python $WORK_DIR/scripts/data_process/qa_search_test_merge.py \
#     --local_dir $LOCAL_DIR \
#     --data_sources $DATA

## 2

# micromamba activate slime
# hf download Qwen/Qwen3-4B --local-dir /data/user/Qwen3-4B
# cd $ROOT_DIR/slime
source scripts/models/qwen3-30B-A3B.sh
export PYTHONPATH="$ROOT_DIR/slime:$ROOT_DIR/Megatron-LM:${PYTHONPATH}"
# python3 tools/convert_hf_to_torch_dist.py \
#   ${MODEL_ARGS[@]} \
#   --hf-checkpoint /data/user/Qwen3-4B \
#   --save /data/user/Qwen3-4B_torch_dist
echo "${MODEL_ARGS[@]}"
torchrun --nproc-per-node 4 \
   tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint /data/user/Qwen/Qwen3-30B-A3B-Base \
   --save /data/user/Qwen3-30B-A3B_base_torch_dist/


## 3.

# conda create -n retriever python=3.10 -y
# conda activate retriever
# # we recommend installing torch with conda for faiss-gpu
# conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
# pip install transformers datasets pyserini
# ## install the gpu version faiss to guarantee efficient RL rollout
# conda install -c pytorch -c nvidia faiss-gpu=1.8.0 -y
# pip install uvicorn fastapi

# 4. Set your save path
# save_path=$ROOT_DIR/search_data/Index
# # Download the index and corpus files
# python $ROOT_DIR/slime/examples/search-r1/local_dense_retriever/download.py --save_path $save_path
# # Combine split index files
# cat $save_path/part_* > $save_path/e5_Flat.index
# # Decompress the corpus
# gzip -d $save_path/wiki-18.jsonl.gz


# 5.
# conda activate retriever

# Set paths
# index_file=$save_path/e5_Flat.index
# corpus_file=$save_path/wiki-18.jsonl
# retriever_name=e5
# retriever_path=intfloat/e5-base-v2

# # Start the retrieval server
# python slime/examples/search-r1/local_dense_retriever/retrieval_server.py \
#     --index_path $index_file \
#     --corpus_path $corpus_file \
#     --topk 3 \
#     --retriever_name $retriever_name \
#     --retriever_model $retriever_path \
#     --faiss_gpu