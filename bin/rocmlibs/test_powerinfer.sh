#!/bin/bash
# 
#  test_powerinfer.sh:  Script to test powerinfer installation with various models
#

# --- Start standard header to set AOMP environment variables ----
realpath=`realpath $0`
thisdir=`dirname $realpath`
. $thisdir/../aomp_common_vars
# --- end standard header ----

# Set POWERINFER_INSTALL_DIR if it is not already set via env variable
POWERINFER_INSTALL_DIR=${POWERINFER_INSTALL_DIR:-$AOMP_INSTALL_DIR/PowerInfer}

# Set MODEL_DIR if it is not already set via env variable
MODEL_DIR=${MODEL_DIR:-$AOMP_REPOS/powerinfer_models}

# Create MODEL_DIR if it does not exist
if [ ! -d $MODEL_DIR ]; then
    mkdir -p $MODEL_DIR
fi

# Install huggingface-cli to download the PowerInfer GGUF models
pip install -U "huggingface_hub[cli]"

# Use huggingface-cli to download the PowerInfer GGUF version of LLaMA(ReLU)-2-7B model
huggingface-cli download --resume-download --local-dir ReluLLaMA-7B --local-dir-use-symlinks False PowerInfer/ReluLLaMA-7B-PowerInfer-GGUF
MODEL_NAME=ReluLLaMA-7B/llama-7b-relu.powerinfer.gguf

# # Use huggingface-cli to download the PowerInfer GGUF version of LLaMA(ReLU)-2-13B model
# huggingface-cli download --resume-download --local-dir ReluLLaMA-13B --local-dir-use-symlinks False PowerInfer/ReluLLaMA-13B-PowerInfer-GGUF
# MODEL_NAME=ReluLLaMA-13B/llama-13b-relu.powerinfer.gguf

# # Use huggingface-cli to download the PowerInfer GGUF version of Falcon(ReLU)-40B model
# huggingface-cli download --resume-download --local-dir ReluFalcon-40B --local-dir-use-symlinks False PowerInfer/ReluFalcon-40B-PowerInfer-GGUF
# MODEL_NAME=ReluFalcon-40B/falcon-40b-relu.powerinfer.gguf

# # Use huggingface-cli to download the PowerInfer GGUF version of LLaMA(ReLU)-2-70B model
# huggingface-cli download --resume-download --local-dir ReluLLaMA-70B --local-dir-use-symlinks False PowerInfer/ReluLLaMA-70B-PowerInfer-GGUF
# MODEL_NAME=ReluLLaMA-70B/llama-70b-relu.powerinfer.gguf

# # Use huggingface-cli to download the PowerInfer GGUF version of ProSparse-LLaMA-2-7B model
# huggingface-cli download --resume-download --local-dir ProSparse-LLaMA-2-7B --local-dir-use-symlinks False PowerInfer/ProSparse-LLaMA-2-7B-GGUF
# MODEL_NAME=ProSparse-LLaMA-2-7B/prosparse-llama-2-7b-gguf.powerinfer.gguf

# # Use huggingface-cli to download the PowerInfer GGUF version of ProSparse-LLaMA-2-13B model
# huggingface-cli download --resume-download --local-dir ProSparse-LLaMA-2-13B --local-dir-use-symlinks False PowerInfer/ProSparse-LLaMA-2-13B-GGUF
# MODEL_NAME=ProSparse-LLaMA-2-13B/prosparse-llama-2-13b-gguf.powerinfer.gguf

# # Use huggingface-cli to download the PowerInfer GGUF version of Bamboo-base-7B model
# huggingface-cli download --resume-download --local-dir Bamboo-base-7B --local-dir-use-symlinks False PowerInfer/Bamboo-base-v0.1-gguf
# MODEL_NAME=Bamboo-base-7B/bamboo-base-7b-gguf.powerinfer.gguf

# # Use huggingface-cli to download the PowerInfer GGUF version of Bamboo-DPO-7B model
# huggingface-cli download --resume-download --local-dir Bamboo-DPO-7B --local-dir-use-symlinks False PowerInfer/Bamboo-DPO-v0.1-gguf
# MODEL_NAME=Bamboo-DPO-7B/bamboo-dpo-7b-gguf.powerinfer.gguf

PREDICTION_LENGTH=128
THREAD_COUNT=8
PROMPT="Once upon a time"
VRAM_BUDGET=4

MODEL_NAME_BASE=$(basename $MODEL_NAME)

# Set the log file name
LOG_FILE_NAME=${MODEL_NAME_BASE}_n${PREDICTION_LENGTH}_t${THREAD_COUNT}_vram${VRAM_BUDGET}GB

# Automatically increment the log file name
if [ -f $LOG_FILE_NAME ]; then
    i=1
    while [ -f $LOG_FILE_NAME.$i ]; do
        let i++
    done
    LOG_FILE_NAME=$LOG_FILE_NAME.$i
fi

$POWERINFER_INSTALL_DIR/bin/main -m $MODEL_DIR/$MODEL_NAME -n $PREDICTION_LENGTH -t $THREAD_COUNT -p $PROMPT --vram-budget $VRAM_BUDGET 2>&1 | tee $LOG_FILE_NAME.log

# Print the log file name and parameters with which the model was run
echo "Inference via PowerInfer completed for the following:"
echo "Model: $MODEL_NAME"
echo "Prediction length: $PREDICTION_LENGTH"
echo "Thread count: $THREAD_COUNT"
echo "Prompt: $PROMPT"
echo "VRAM budget: $VRAM_BUDGET GB"
echo "Log file: $LOG_FILE_NAME.log"
