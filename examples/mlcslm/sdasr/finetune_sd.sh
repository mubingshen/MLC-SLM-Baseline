#!/bin/bash

. ./path.sh || exit 1;
unset LD_LIBRARY_PATH

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
echo "CUDA_VISIBLE_DEVICES is ${CUDA_VISIBLE_DEVICES}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

stage=2
stop_stage=2

train_data_dir=/MCLSLM/train_data
dev_data_dir=/MCLSLM/MLC-SLM_Workshop-Development_Set/data

examples=finetune_pyannote
train_wav_list=$examples/rttm/train_wav.lst
dev_wav_list=$examples/rttm/dev_wav.lst

finetune_dataset_conf=conf/finetune_pyannote_speaker_diarization.yaml
finetune_save_path=$examples/exp
checkpoint='pyannote/segmentation-3.0'
hf_access_token=your_hf_access_token


if [ -z "$hf_access_token" ]; then
  log "[ERROR]: The hf_access_token is empty. If \"include_overlap\" is set to true,\
    the \"hf_access_token\" for \"pyannote/segmentation-3.0\" should be provided." && exit 1
fi

if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
  if [ ! -f "$train_wav_list" ] || [ ! -f "$dev_wav_list" ]; then
    log " Stage 1: Prepare finetune files..."
    mkdir -p $examples/rttm/
    python local/prepare_finetune_files.py --dataset_path $train_data_dir --output_path $examples/rttm/ \
                                        --dataset_part 'train'
    python local/prepare_finetune_files.py --dataset_path $dev_data_dir --output_path $examples/rttm/ \
                                        --dataset_part 'dev'
    log "save finetune files to $train_wav_list $dev_wav_list"
  else
    log " Stage 1: $train_wav_list $dev_wav_list exists. Skip this stage."
  fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log " Stage2: finetune segmentation model..."
    python local/finetune_segmentation.py --dataset $finetune_dataset_conf --output_path $finetune_save_path  \
                                    --max_epoch 20 --batch_size 32 --devices 4\
                                    --hf_access_token $hf_access_token --checkpoint $checkpoint 
fi

