#!/bin/bash
# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
unset LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES="7"
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
nj=4

stage=0
stop_stage=3

test_data_dir=your_test_data_dir

. local/parse_options.sh || exit 1

rttm_dir=$exp/rttm

spilt_dir=$test_data_dir/sd_split
data_type=raw
num_utts_per_shard=1000
decode_mode=whisper_llm_decode
asr_dir=../asr/exp/step2
asr_checkpoint=$asr_dir/avg_3.pt
asr_decode_dir=exp/sdasr


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  log " Stage0: Split test wavs using predicted rttms..."
  python local/split_test_wavs_using_rttms.py --rttm_dir $rttm_dir --test_data_dir $test_data_dir --output_dir $spilt_dir
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  log " Stage1: prepare required data format"
  if [ $data_type == "shard" ]; then
    tools/make_shard_list.py --num_utts_per_shard $num_utts_per_shard \
      --num_threads $nj $spilt_dir/wav.scp $spilt_dir/text \
      $(realpath $test_data_dir/sd_shards) $test_data_dir/sd_${data_type}_data.list
  else
    tools/make_raw_list.py $spilt_dir/wav.scp $spilt_dir/text $test_data_dir/sd_${data_type}_data.list
  fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  log " Stage2: Decode with ASR-LLM models trained on the task I"
  python wenet/bin/recognize_mlcslm_baseline.py --gpu 7 \
    --modes $decode_mode \
    --config $asr_dir/train.yaml \
    --data_type $data_type \
    --test_data $test_data_dir/sd_${data_type}_data.list \
    --checkpoint $asr_checkpoint \
    --batch_size 10 \
    --dtype bf16 \
    --result_dir $asr_decode_dir 
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  log " Stage3: Generate hyp.stm using hyp rttms and hyp transcripts "
  python local/generate_hyp_stm.py --rttm_dir $rttm_dir --text $asr_decode_dir/$decode_mode/text --out_file $asr_decode_dir/$decode_mode/hyp.stm

  log "Add spaces between Japanese/Korean/Thai characters"
  python local/add_space_between_chars_sdasr.py --input $asr_decode_dir/$decode_mode/hyp.stm --output $asr_decode_dir/$decode_mode/hyp_space.stm

  log "Test done, you need to submit $asr_decode_dir/$decode_mode/hyp_space.stm to the leaderboard!"
fi