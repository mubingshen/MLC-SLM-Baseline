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

stage=11
stop_stage=11

conf_file=conf/diar.yaml
include_overlap=false
checkpoint='pyannote/segmentation-3.0'
hf_access_token=your_hf_access_token

train_data_dir=your_train_data_dir
dev_data_dir=your_dev_data_dir

examples=reference_3dspeaker
references=$examples/reference_files
exp=$examples/reference_without_overlap

. local/parse_options.sh || exit 1

wav_list=$references/dev_wav.list
ref_rttm_list=$references/dev_rttm.list
json_dir=$exp/json
embs_dir=$exp/embs
rttm_dir=$exp/rttm

spilt_dir=$dev_data_dir/sd_split
data_type=raw
num_utts_per_shard=1000
decode_mode=whisper_llm_decode
asr_dir=../asr/exp/step2
asr_checkpoint=$asr_dir/avg_3.pt
asr_decode_dir=exp/sdasr


if [ "$include_overlap" = true ] && [ -z "$hf_access_token" ]; then
  log "[ERROR]: The hf_access_token is empty. If \"include_overlap\" is set to true,\
    the \"hf_access_token\" for \"pyannote/segmentation-3.0\" should be provided." && exit 1
fi

if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
  if [ ! -f "$wav_list" ]; then
    log " Stage 1: Prepare ref files..."
    mkdir -p $references
    python local/prepare_reference_files.py --dataset_path $dev_data_dir --output_path $references --dataset_part 'dev'          
    log "generate ref files in $references"
  else
    log " Stage 1: $wav_list exists. Skip this stage."
  fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  mkdir -p $exp
  if [ "$include_overlap" = true ]; then
    log " Stage2: Do overlap detection for ref wavs..."
    python local/overlap_detection.py --wavs $wav_list --out_dir $json_dir --hf_access_token $hf_access_token --checkpoint $checkpoint 
  fi
  log " Stage2: Do vad for ref wavs..."
  python local/voice_activity_detection.py --wavs $wav_list --out_file $json_dir/vad.json
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  log " Stage3: Prepare subsegments info..."
  python local/prepare_subseg_json.py --vad $json_dir/vad.json --out_file $json_dir/subseg.json
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  log " Stage4: Extract speaker embeddings..."
  # Set speaker_model_id to damo/speech_eres2net_sv_zh-cn_16k-common when using eres2net 
  speaker_model_id=iic/speech_campplus_sv_zh_en_16k-common_advanced
  torchrun --nproc_per_node=$nj local/extract_diar_embeddings.py \
                                  --model_id $speaker_model_id \
                                  --conf $conf_file \
                                  --subseg_json $json_dir/subseg.json \
                                  --embs_out $embs_dir \
                                  --gpu 7 \
                                  --use_gpu
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  log " Stage5: Perform clustering and postprocessing, and output sys rttms..."
  if [ "$include_overlap" = true ]; then
    cluster_rttm_dir=$rttm_dir/intermediate
  else
    cluster_rttm_dir=$rttm_dir
  fi
  torchrun --nproc_per_node=$nj local/cluster_and_postprocess.py \
                                  --conf $conf_file \
                                  --wavs $wav_list \
                                  --audio_embs_dir $embs_dir \
                                  --rttm_dir $cluster_rttm_dir
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  if [ "$include_overlap" = true ]; then
    log " Stage6: Do overlap detection postprocess..."
    python local/refine_with_OD.py \
      --init_rttm_dir $rttm_dir/intermediate \
      --rttm_dir $rttm_dir \
      --segmentation_dir $json_dir \
      --hf_access_token $hf_access_token \
      --checkpoint $checkpoint 
  fi
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  log " Stage7: Get the DER metrics..."
  if [ -f $ref_rttm_list ]; then
    cat $ref_rttm_list | while read line;do cat $line;done > $exp/concat_ref_rttm
    log "Computing DER overall..."
    python local/compute_der.py --exp_dir $exp --ref_rttm $exp/concat_ref_rttm
    log "Computing DER perlang..."
    python local/compute_der_perlang.py --exp_dir $exp --ref_rttm $ref_rttm_list
  else
    log "Refrttm.list is not detected. Can't calculate the result"
  fi
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
  log " Stage8: Split dev wavs using predicted rttms..."
  python local/split_wavs_using_rttms.py --rttm_dir $rttm_dir --dev_data_dir $dev_data_dir --output_dir $spilt_dir
fi

if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
  log " Stage8: prepare required data format"
  if [ $data_type == "shard" ]; then
    tools/make_shard_list.py --num_utts_per_shard $num_utts_per_shard \
      --num_threads $nj $spilt_dir/wav.scp $spilt_dir/text \
      $(realpath $dev_data_dir/sd_shards) $dev_data_dir/sd_${data_type}_data.list
  else
    tools/make_raw_list.py $spilt_dir/wav.scp $spilt_dir/text $dev_data_dir/sd_${data_type}_data.list
  fi
fi

if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
  log " Stage10: Decode with ASR-LLM models trained on the task I"
  python wenet/bin/recognize_mlcslm_baseline.py --gpu 6 \
    --modes $decode_mode \
    --config $asr_dir/train.yaml \
    --data_type $data_type \
    --test_data $dev_data_dir/sd_${data_type}_data.list \
    --checkpoint $asr_checkpoint \
    --batch_size 10 \
    --dtype bf16 \
    --result_dir $asr_decode_dir 
fi

if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ]; then
  log " Stage11: Compute the Time-Constrained minimum-Permutation Word Error Rate (tcpWER) using MeetEval"
  
  log " Generate ref.stm using ref rttms and ref transcripts "
  python local/generate_ref_stm.py --rttm $ref_rttm_list --text $dev_data_dir/text_tn --out_file $asr_decode_dir/$decode_mode/ref.stm
  
  log " Generate hyp.stm using hyp rttms and hyp transcripts "
  python local/generate_hyp_stm.py --rttm_dir $rttm_dir --text $asr_decode_dir/$decode_mode/text --out_file $asr_decode_dir/$decode_mode/hyp.stm

  log "Add spaces between Japanese/Korean/Thai characters"
  python local/add_space_between_chars_sdasr.py --input $asr_decode_dir/$decode_mode/ref.stm --output $asr_decode_dir/$decode_mode/ref_space.stm
  python local/add_space_between_chars_sdasr.py --input $asr_decode_dir/$decode_mode/hyp.stm --output $asr_decode_dir/$decode_mode/hyp_space.stm

  meeteval-wer tcpwer -r $asr_decode_dir/$decode_mode/ref_space.stm -h $asr_decode_dir/$decode_mode/hyp_space.stm --collar 5
  for lang in American Australian British Filipino Indian French German Italian Japanese Korean Portuguese Russian Spanish Thai Vietnamese; do
    grep $lang $asr_decode_dir/$decode_mode/ref_space.stm > $asr_decode_dir/$decode_mode/ref_space_$lang.stm
    grep $lang $asr_decode_dir/$decode_mode/hyp_space.stm > $asr_decode_dir/$decode_mode/hyp_space_$lang.stm
    meeteval-wer tcpwer -r $asr_decode_dir/$decode_mode/ref_space_$lang.stm -h $asr_decode_dir/$decode_mode/hyp_space_$lang.stm --collar 5
  done
fi