#!/bin/bash
. ./path.sh || exit 1;
unset LD_LIBRARY_PATH

export CUDA_VISIBLE_DEVICES="6"
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

stage=0
stop_stage=0

nj=30

# data_type can be `raw` or `shard`. Typically, raw is used for small dataset,
# `shard` is used for large dataset which is over 1k hours, and `shard` is
# faster on reading data and training.
data_type=raw
num_utts_per_shard=1000

test_data_dir=test_data
test_segments_path=test_data/segments
test_split_dir=test_data/split
test_split_log_dir=test_data/split_log
cmd=local/run.pl

model_dir=your_model_dir

# use average_checkpoint will get better result
decode_checkpoint=$model_dir/avg_3.pt
decode_mode=whisper_llm_decode

. tools/parse_options.sh || exit 1;


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "Stage 0: Generate segments"

    ln -s your_test_data_path ./test_data
    python local/prepare_test_segments.py --data_dir $test_data_dir --segments_path $test_segments_path

    log "Segments file of test dataset is saved into $test_segments_path"
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  log "Stage 1: Split wavs using segments"
  log "Split test dataset wavs"

  nutt=$(<${test_segments_path} wc -l)
  nj=$((nj<nutt?nj:nutt))
  mkdir -p $test_split_log_dir
  split_segments=""
  for n in $(seq ${nj}); do
      split_segments="${split_segments} ${test_split_log_dir}/segments.${n}"
  done
  local/split_scp.pl "${test_segments_path}" ${split_segments}

  # shellcheck disable=SC2046
  ${cmd} "JOB=1:${nj}" "${test_split_log_dir}/split_wavs.JOB.log" \
      python local/split_wav.py \
          "--segments_path=${test_split_log_dir}/segments.JOB" \
          "--output_dir=${test_split_dir}/split.JOB"

  cat ${test_split_dir}/split.*/wav.scp | shuf > $test_data_dir/wav.scp
  cat ${test_split_dir}/split.*/text | shuf > $test_data_dir/text # The text file is not used in the test stage, but it is required for the data preparation.
  
  log "Split wavs done"
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  log "Stage 2: Prepare data, prepare required format"
  
  if [ $data_type == "shard" ]; then
    tools/make_shard_list.py --num_utts_per_shard $num_utts_per_shard \
      --num_threads $nj $test_data_dir/wav.scp $test_data_dir/text \
      $(realpath $test_data_dir/shards) $test_data_dir/${data_type}_data.list
  else
    tools/make_raw_list.py $test_data_dir/wav.scp $test_data_dir/text \
      $test_data_dir/${data_type}_data.list
  fi

  log "Data preparation done"
fi



if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  # Test model, please specify the model you want to test by --checkpoint
  python wenet/bin/recognize_mlcslm_baseline.py --gpu 6 \
    --modes $decode_mode \
    --config $model_dir/train.yaml \
    --data_type $data_type \
    --test_data $test_data_dir/${data_type}_data.list \
    --checkpoint $decode_checkpoint \
    --batch_size 10 \
    --dtype bf16 \
    --result_dir $model_dir/test

  python local/add_space_between_chars_asr.py --input $model_dir/test/$decode_mode/text --output $model_dir/test/$decode_mode/text_space

  log "Test done, you need to submit $model_dir/test/$decode_mode/text_space to the leaderboard!"
fi
