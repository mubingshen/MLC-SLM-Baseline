# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Xiaoyu Chen, Di Wu)
#               2025 ASLP@NPU for MLC-SLM Baseline. (authors: Bingshen Mu)
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import argparse
import copy
import logging
import os

import torch
import yaml
from torch.utils.data import DataLoader

from wenet.dataset.dataset import Dataset
from wenet.utils.config import override_config
from wenet.utils.init_model import init_model
from wenet.utils.init_tokenizer import init_tokenizer


def get_args():
    parser = argparse.ArgumentParser(description='recognize with your model')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--test_data', required=True, help='test data file')
    parser.add_argument('--data_type',
                        default='raw',
                        choices=['raw', 'shard'],
                        help='train and cv data type')
    parser.add_argument('--gpu',
                        type=int,
                        default=-1,
                        help='gpu id for this rank, -1 for cpu')
    parser.add_argument('--device',
                        type=str,
                        default="cpu",
                        choices=["cpu", "npu", "cuda"],
                        help='accelerator to use')
    parser.add_argument('--dtype',
                        type=str,
                        default='fp32',
                        choices=['fp16', 'fp32', 'bf16'],
                        help='model\'s dtype')
    parser.add_argument('--num_workers',
                        default=0,
                        type=int,
                        help='num of subprocess workers for reading')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    parser.add_argument('--beam_size',
                        type=int,
                        default=10,
                        help='beam size for search')
    parser.add_argument('--length_penalty',
                        type=float,
                        default=0.0,
                        help='length penalty')
    parser.add_argument('--blank_penalty',
                        type=float,
                        default=0.0,
                        help='blank penalty')
    parser.add_argument('--result_dir', required=True, help='asr result file')
    parser.add_argument('--batch_size',
                        type=int,
                        default=16,
                        help='asr result file')
    parser.add_argument('--modes',
                        nargs='+',
                        help="""decoding mode, support the following:
                             attention
                             ctc_greedy_search
                             ctc_prefix_beam_search
                             attention_rescoring
                             rnnt_greedy_search
                             rnnt_beam_search
                             rnnt_beam_attn_rescoring
                             ctc_beam_td_attn_rescoring
                             hlg_onebest
                             hlg_rescore
                             paraformer_greedy_search
                             paraformer_beam_search""")
    parser.add_argument('--search_ctc_weight',
                        type=float,
                        default=1.0,
                        help='ctc weight for nbest generation')
    parser.add_argument('--search_transducer_weight',
                        type=float,
                        default=0.0,
                        help='transducer weight for nbest generation')
    parser.add_argument('--ctc_weight',
                        type=float,
                        default=0.0,
                        help='ctc weight for rescoring weight in \
                                  attention rescoring decode mode \
                              ctc weight for rescoring weight in \
                                  transducer attention rescore decode mode')

    parser.add_argument('--transducer_weight',
                        type=float,
                        default=0.0,
                        help='transducer weight for rescoring weight in '
                        'transducer attention rescore mode')
    parser.add_argument('--attn_weight',
                        type=float,
                        default=0.0,
                        help='attention weight for rescoring weight in '
                        'transducer attention rescore mode')
    parser.add_argument('--decoding_chunk_size',
                        type=int,
                        default=-1,
                        help='''decoding chunk size,
                                <0: for decoding, use full chunk.
                                >0: for decoding, use fixed chunk size as set.
                                0: used for training, it's prohibited here''')
    parser.add_argument('--num_decoding_left_chunks',
                        type=int,
                        default=-1,
                        help='number of left chunks for decoding')
    parser.add_argument('--simulate_streaming',
                        action='store_true',
                        help='simulate streaming inference')
    parser.add_argument('--reverse_weight',
                        type=float,
                        default=0.0,
                        help='''right to left weight for attention rescoring
                                decode mode''')
    parser.add_argument('--override_config',
                        action='append',
                        default=[],
                        help="override yaml config")

    parser.add_argument('--word',
                        default='',
                        type=str,
                        help='word file, only used for hlg decode')
    parser.add_argument('--hlg',
                        default='',
                        type=str,
                        help='hlg file, only used for hlg decode')
    parser.add_argument('--lm_scale',
                        type=float,
                        default=0.0,
                        help='lm scale for hlg attention rescore decode')
    parser.add_argument('--decoder_scale',
                        type=float,
                        default=0.0,
                        help='lm scale for hlg attention rescore decode')
    parser.add_argument('--r_decoder_scale',
                        type=float,
                        default=0.0,
                        help='lm scale for hlg attention rescore decode')

    parser.add_argument(
        '--context_bias_mode',
        type=str,
        default='',
        help='''Context bias mode, selectable from the following
                                option: decoding-graph, deep-biasing''')
    parser.add_argument('--context_list_path',
                        type=str,
                        default='',
                        help='Context list path')
    parser.add_argument('--context_graph_score',
                        type=float,
                        default=0.0,
                        help='''The higher the score, the greater the degree of
                                bias using decoding-graph for biasing''')

    parser.add_argument('--use_lora',
                        type=bool,
                        default=False,
                        help='''Whether to use lora for biasing''')
    parser.add_argument("--lora_ckpt_path",
                        default=None,
                        type=str,
                        help="lora checkpoint path.")
    args = parser.parse_args()
    print(args)
    return args


def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    if args.gpu != -1:
        # remain the original usage of gpu
        args.device = "cuda"
    if "cuda" in args.device:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    if len(args.override_config) > 0:
        configs = override_config(configs, args.override_config)

    test_conf = copy.deepcopy(configs['dataset_conf'])

    test_conf['filter_conf']['max_length'] = 1500
    test_conf['filter_conf']['min_length'] = 0
    test_conf['filter_conf']['token_max_length'] = 102400
    test_conf['filter_conf']['token_min_length'] = 0
    test_conf['filter_conf']['max_output_input_ratio'] = 102400
    test_conf['filter_conf']['min_output_input_ratio'] = 0
    test_conf['speed_perturb'] = False
    test_conf['spec_aug'] = False
    test_conf['spec_sub'] = False
    test_conf['spec_trim'] = False
    test_conf['shuffle'] = False
    test_conf['sort'] = False
    test_conf['cycle'] = 1
    test_conf['list_shuffle'] = False
    if 'fbank_conf' in test_conf:
        test_conf['fbank_conf']['dither'] = 0.0
    elif 'mfcc_conf' in test_conf:
        test_conf['mfcc_conf']['dither'] = 0.0
    test_conf['batch_conf']['batch_type'] = "static"
    test_conf['batch_conf']['batch_size'] = args.batch_size

    tokenizer = init_tokenizer(configs)
    test_dataset = Dataset(args.data_type,
                           args.test_data,
                           tokenizer,
                           test_conf,
                           partition=False)

    test_data_loader = DataLoader(test_dataset,
                                  batch_size=None,
                                  num_workers=args.num_workers)
    
    # Init asr model from configs
    args.jit = False
    model, configs = init_model(args, configs, tokenizer, inference_mode=True)
    print(model)

    device = torch.device(args.device)
    model = model.to(device)
    model.eval()
    # TODO different dtype for model decoding
    dtype = torch.float32
    if args.dtype == 'fp16':
        dtype = torch.float16
    elif args.dtype == 'bf16':
        dtype = torch.bfloat16
    logging.info("compute dtype is {}".format(dtype))
    
    files = {}
    for mode in args.modes:
        dir_name = os.path.join(args.result_dir, mode)
        os.makedirs(dir_name, exist_ok=True)
        file_name = os.path.join(dir_name, 'text')
        files[mode] = open(file_name, 'w')
    max_format_len = max([len(mode) for mode in args.modes])

    with torch.cuda.amp.autocast(enabled=True,
                                 dtype=dtype,
                                 cache_enabled=False):
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_data_loader):
                keys = batch["keys"]
                feats = batch["feats"].to(device)
                feats_lengths = batch["feats_lengths"].to(device)
                results = model.decode(
                    feats,
                    feats_lengths,
                    configs['llm_path']
                    )
                # import pdb;pdb.set_trace()
                assert len(keys) == len(results)
                for i in range(len(keys)):
                    logging.info(f"{keys[i]} {results[i]}")
                    files[mode].write(f"{keys[i]} {results[i]}\n")
        for mode, f in files.items():
            f.close()


if __name__ == '__main__':
    main()
