#!/bin/bash
#SBATCH -A coreai_prod_infbench
#SBATCH -p batch 
#SBATCH -N 13                     # number of nodes
#SBATCH -t 02:00:00              # wall time  (8 for backfill, 4 for Luna)
#SBATCH -J "coreai_prod_infbench-sglang:test-a2a"
#SBATCH --exclusive             # exclusive node access
#SBATCH --mem=0                 # all mem avail
#SBATCH --ntasks-per-node=1     # n tasks per machine (one task per gpu)
#SBATCH --overcommit
#SBATCH --dependency=singleton  # only run one instance at a time
#SBATCH -o logs/slurm.log
set -x

# CONTAINER="ishandhanani/sglang:nightly-dev-20251024-5983e5bd-dyn" #"lmsysorg/sglang:dev"
# CONTAINER="/lustre/fsw/coreai_prod_infbench/faradawny/docker/cutedsl-sglang:nightly-dev-20251024-5983e5bd.sqsh"
CONTAINER="/lustre/fsw/coreai_prod_infbench/faradawny/docker/sglang:nightly-dev-20251121-c56fc424.sqsh"

# if [[ ! -f ${CONTAINER} ]]; then
#     srun -A coreai_prod_infbench -N1 --partition=batch --exclusive --time=00:30:00 bash -c "enroot import --output ${CONTAINER} docker://lmsysorg/sglang:v0.5.5.post2"
# fi

BASE_DIR="$(pwd)"
BASE_DIR_C="/base_dir"

HF_DIR="/lustre/fsw/coreai_comparch_infbench/common/cache"
HF_DIR_C="/root/.cache/huggingface"

CONFIGS="/lustre/share/coreai_comparch_infbench/kylliang/sglang_wideep_configs"
CONFIGS_C="/configs"

model_path="nvidia/DeepSeek-R1-0528-FP4"

CONTAINER_NAME="sglang"

MOUNTS="--container-mounts=$BASE_DIR:$BASE_DIR_C,$HF_DIR:$HF_DIR_C,$CONFIGS:$CONFIGS_C,/lustre:/lustre"
EXPORTS="--export=ALL,HF_TOKEN=${HF_TOKEN}"

DIR="${BASE_DIR}/logs"
mkdir -p $DIR
OUTFILE="${DIR}/output-%t.txt"
NUM_PREFILL=1
NUM_DECODE=12

nodes=( $(scontrol show hostnames $SLURM_NODELIST) )
PREFILL_NODE0=${nodes[0]}
DECODE_NODE0=${nodes[$NUM_PREFILL]}
PREFILL_NODE0=$(nslookup ${PREFILL_NODE0} | grep 'Address:' | awk 'NR>1 {print $2}')
DECODE_NODE0=$(nslookup ${DECODE_NODE0} | grep 'Address:' | awk 'NR>1 {print $2}')

deepep_config="${CONFIGS_C}/deepep_configs/deepep_config.json"
prefill_expert_path="${CONFIGS_C}/expert_distributions/prefill_dsr1-0528_sglang_bench_serving_in1024out1024_num25000.json"
decode_expert_path="${CONFIGS_C}/expert_distributions/decode_dsr1-0528_loadgen_in1024out1024_num2000_2p12d.json"

read -r -d '' cmd <<EOF
set -x

wait_until_ready() {
  local SERVER_URL="\$1"
  local SERVER_PID="\$2"
  while true; do
    status_code=\$(curl -s -o /dev/null -w "%{http_code}" "\${SERVER_URL}/health" || echo "000")
    if [ "\$status_code" -eq 200 ]; then
      echo "Server \${SERVER_URL} is ready"
      break
    fi
    if [ -n "\$SERVER_PID" ] && ! kill -0 \$SERVER_PID 2>/dev/null; then
      echo "Error: Server process (PID \$SERVER_PID) died unexpectedly while starting."
      exit 1
    fi
    sleep 30
  done
}

echo "=== 11-30 low precision prefill SLURM_NODEID \$SLURM_NODEID"


export SGLANG_HEALTH_CHECK_TIMEOUT=3600

if [[ "\$SLURM_NODEID" -lt $NUM_PREFILL ]]; then

# PREFILL
export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=1800

# Create cache directories
export SGLANG_TORCH_PROFILER_DIR=${BASE_DIR_C}/../prefill_torch_profiler
mkdir -p \${SGLANG_TORCH_PROFILER_DIR}
export SGLANG_DUMPER_DIR=${BASE_DIR_C}/../prefill_sglang_dump
mkdir -p \${SGLANG_DUMPER_DIR}
export SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR=${BASE_DIR_C}/../prefill_sglang_expert_distribution_recorder
mkdir -p \${SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR}
export FLASHINFER_WORKSPACE_BASE=${BASE_DIR_C}/../prefill_flashinfer_workspace_base
mkdir -p \${FLASHINFER_WORKSPACE_BASE}
export SGL_DG_CACHE_DIR=${BASE_DIR_C}/../prefill_deepgemm_cache
mkdir -p \${SGL_DG_CACHE_DIR}

DYN_SKIP_SGLANG_LOG_FORMATTING=1 \
SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024 \
SGLANG_NVFP4_CKPT_FP8_GEMM_IN_ATTN=1 \
SGLANG_PER_TOKEN_GROUP_QUANT_8BIT_V2=1 \
SGL_JIT_DEEPGEMM_PRECOMPILE=0 \
MC_TE_METRIC=true \
SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE=100000 \
SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=100000 \
SGLANG_DISAGGREGATION_WAITING_TIMEOUT=100000 \
SGLANG_HACK_SEQ_BOOTSTRAP_ROOM=1 \
SGLANG_MOONCAKE_CUSTOM_MEM_POOL=True \
MC_FORCE_MNNVL=1 \
NCCL_MNNVL_ENABLE=1 \
NCCL_CUMEM_ENABLE=1 \
SGLANG_USE_MESSAGE_QUEUE_BROADCASTER=0 \
SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=1 \
PYTHONUNBUFFERED=1 \
python3 -m sglang.launch_server \
    `#--served-model-name deepseek-ai/DeepSeek-R1` \
    --disaggregation-mode prefill \
    --host 0.0.0.0 \
    --port 30000 \
    --max-running-requests 768 \
    --context-length 4224 \
    --disable-radix-cache \
    --disable-shared-experts-fusion \
    --watchdog-timeout 1000000 \
    --disable-chunked-prefix-cache \
    --attention-backend trtllm_mla \
    --kv-cache-dtype fp8_e4m3 \
    --enable-single-batch-overlap \
    --tp-size $((4 * ${NUM_PREFILL})) \
    --dp-size $((4 * ${NUM_PREFILL})) \
    --ep-size $((4 * ${NUM_PREFILL})) \
    --enable-dp-attention \
    --moe-dense-tp-size 1 \
    --enable-dp-lm-head \
    --chunked-prefill-size 131072 \
    --eplb-algorithm deepseek \
    --model-path ${model_path} \
    --trust-remote-code \
    --disable-cuda-graph \
    --mem-fraction-static 0.84 \
    --max-total-tokens 131072 \
    --max-prefill-tokens 32768 \
    --load-balance-method round_robin \
    --quantization modelopt_fp4 \
    --moe-runner-backend flashinfer_cutlass \
    --disaggregation-bootstrap-port 30001 \
    --dist-init-addr ${PREFILL_NODE0}:20000 \
    --dist-timeout 3600 \
    --nnodes ${NUM_PREFILL} \
    --node-rank \$SLURM_NODEID \
&
SERVER_PID=\$!

else

# DECODE
export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=1800

node_rank=\$((\$SLURM_NODEID - $NUM_PREFILL))

# Create cache directories
export SGLANG_TORCH_PROFILER_DIR=${BASE_DIR_C}/../decode_torch_profiler
mkdir -p \${SGLANG_TORCH_PROFILER_DIR}
export SGLANG_DUMPER_DIR=${BASE_DIR_C}/../decode_sglang_dump
mkdir -p \${SGLANG_DUMPER_DIR}
export SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR=${BASE_DIR_C}/../decode_sglang_expert_distribution_recorder
mkdir -p \${SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR}
export FLASHINFER_WORKSPACE_BASE=${BASE_DIR_C}/../decode_flashinfer_workspace_base
mkdir -p \${FLASHINFER_WORKSPACE_BASE}
export SGL_DG_CACHE_DIR=${BASE_DIR_C}/../decode_deepgemm_cache
mkdir -p \${SGL_DG_CACHE_DIR}

DYN_SKIP_SGLANG_LOG_FORMATTING=1 \
SGLANG_NVFP4_CKPT_FP8_GEMM_IN_ATTN=1 \
SGLANG_PER_TOKEN_GROUP_QUANT_8BIT_V2=1 \
SGL_JIT_DEEPGEMM_PRECOMPILE=0 \
SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024 \
SGLANG_CUTEDSL_MOE_NVFP4_DISPATCH=1 \
SGLANG_FP4_GEMM_BACKEND=cutlass \
MC_TE_METRIC=true \
SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE=100000 \
SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=100000 \
SGLANG_DISAGGREGATION_WAITING_TIMEOUT=100000 \
SGLANG_HACK_SEQ_BOOTSTRAP_ROOM=1 \
SGLANG_MOONCAKE_CUSTOM_MEM_POOL=True \
MC_FORCE_MNNVL=1 \
NCCL_MNNVL_ENABLE=1 \
NCCL_CUMEM_ENABLE=1 \
SGLANG_USE_MESSAGE_QUEUE_BROADCASTER=0 \
SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=1 \
PYTHONUNBUFFERED=1 \
python3 -m sglang.launch_server \
    `#--served-model-name deepseek-ai/DeepSeek-R1` \
    --model-path ${model_path} \
    --trust-remote-code \
    --disaggregation-mode decode \
    --dist-init-addr ${DECODE_NODE0}:20000 \
    --dist-timeout 3600 \
    --disaggregation-bootstrap-port 30001 \
    --nnodes ${NUM_DECODE} \
    --node-rank \$node_rank \
    --tp-size $((4 * ${NUM_DECODE})) \
    --dp-size $((4 * ${NUM_DECODE})) \
    --ep-size $((4 * ${NUM_DECODE})) \
    --enable-dp-attention \
    --host 0.0.0.0 \
    --port 30000 \
    --decode-log-interval 1 \
    --max-running-requests 49152 \
    --context-length 4224 \
    --disable-radix-cache \
    --disable-shared-experts-fusion \
    --watchdog-timeout 1000000 \
    --disable-chunked-prefix-cache \
    --kv-cache-dtype fp8_e4m3 \
    --enable-single-batch-overlap \
    --mem-fraction-static 0.83 \
    --moe-a2a-backend deepep \
    --deepep-mode low_latency \
    --ep-dispatch-algorithm static \
    --cuda-graph-bs 1024 \
    --num-reserved-decode-tokens 112 \
    --ep-num-redundant-experts 32 \
    --eplb-algorithm deepseek \
    --moe-dense-tp-size 1 \
    --enable-dp-lm-head \
    --prefill-round-robin-balance \
    --max-total-tokens 3122380 \
    --max-prefill-tokens 16384 \
    --quantization modelopt_fp4 \
    --moe-runner-backend flashinfer_cutedsl \
&
SERVER_PID=\$!

fi



if [[ "\$SLURM_NODEID" -eq 0 ]]; then
  # Define dataset path
  DATASET_DIR="${BASE_DIR_C}/datasets"
  DATASET_FILE="\${DATASET_DIR}/ShareGPT_V3_unfiltered_cleaned_split.json"
  
  # Check if file exists and download if needed
  if [[ ! -f "\$DATASET_FILE" ]]; then
    echo "Dataset file not found at \$DATASET_FILE, downloading..."
    mkdir -p "\$DATASET_DIR"
    wget -O "\$DATASET_FILE" \
      "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
    echo "Dataset downloaded successfully to \$DATASET_FILE"
  else
    echo "Using existing dataset file at \$DATASET_FILE"
  fi

  
  wait_until_ready http://${PREFILL_NODE0}:30000 \$SERVER_PID
  
  echo "Launching router"
  python3 -m sglang_router.launch_router --pd-disaggregation --mini-lb --prefill http://${PREFILL_NODE0}:30000 30001 --decode http://${DECODE_NODE0}:30000 --host 0.0.0.0 --port 8000 &

  sleep 10
  echo "Launch bench_one_batch_server 1k 1k 1k"
  python3 -m sglang.bench_one_batch_server \
    --dataset-path "\$DATASET_FILE" \
    --model-path ${model_path} \
    --base-url http://${PREFILL_NODE0}:8000 \
    --batch-size 1024 \
    --input-len 1024 \
    --output-len 1 \
    --skip-warmup \
    --result-filename /lustre/fsw/coreai_prod_infbench/faradawny/tme-slurm-commands-guide/trevor/logs/result.jsonl

  wait \$SERVER_PID

else
  wait \$SERVER_PID
fi

EOF

srun --mpi=pmix --ntasks-per-node=1 -o $OUTFILE -e $OUTFILE --container-image="$CONTAINER" $MOUNTS $EXPORTS --container-name=$CONTAINER_NAME bash -c "${cmd}"

# remove from decode:     --init-expert-location ${decode_expert_path} \ --kill-on-bad-exit 