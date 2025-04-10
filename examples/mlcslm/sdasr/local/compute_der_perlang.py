# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import os
import sys
import json
import argparse
import shutil
import numpy as np
from speakerlab.utils.utils import parse_config, get_logger
from DER import DER


def main(args): 
    logger = get_logger()
    sys_rttm_dir = os.path.join(args.exp_dir, 'rttm')
    result_dir = os.path.join(args.exp_dir, 'result')
    os.makedirs(result_dir, exist_ok=True)
    sys_rttm_dict={}
    sys_rttm_lines=open(args.ref_rttm).readlines()
    for sys_rttm_file in sys_rttm_lines:
        if sys_rttm_file.endswith('.rttm') or 'sys' in sys_rttm_file:
            continue
        language_id = os.path.basename(sys_rttm_file).split('_')[0]
        if language_id not in sys_rttm_dict:
            sys_rttm_dict[language_id] = []
        sys_rttm_dict[language_id].append(sys_rttm_file.strip())

    try:
        msg_list=[]
        for language_id, sys_rttms in sys_rttm_dict.items():
            with open(os.path.join(args.exp_dir, language_id+'_ref_rttm'), 'w') as outfile:
                for fname in sys_rttms:
                    with open(fname, 'r') as infile:
                        outfile.write(infile.read())

            concate_rttm_file = sys_rttm_dir + f"/{language_id}_output_rttm"
            if os.path.exists(concate_rttm_file):
                os.remove(concate_rttm_file)

            meta_file = os.path.join(args.exp_dir, 'json/subseg.json')
            with open(meta_file, "r") as f:
                full_meta = json.load(f)

            all_keys = full_meta.keys()
            A = ['_'.join(word.rstrip().split("_")[:-2]) for word in all_keys if language_id in word]
            all_rec_ids = list(set(A))
            all_rec_ids.sort()
            if len(all_rec_ids) <= 0:
                msg = "No recording IDs found! Please check if meta_data json file is properly generated."
                raise ValueError(msg)

            out_rttm_files = []
            for rec_id in all_rec_ids:
                out_rttm_files.append(os.path.join(sys_rttm_dir, rec_id+'.rttm'))

            with open(concate_rttm_file, "w") as cat_file:
                for f in out_rttm_files:
                    with open(f, "r") as indi_rttm_file:
                        shutil.copyfileobj(indi_rttm_file, cat_file)
            
            if args.ref_rttm != '':
                [MS, FA, SER, DER_] = DER(
                    os.path.join(args.exp_dir, language_id+'_ref_rttm'),
                    concate_rttm_file,
                )
                msg = ', '.join(['MS: %f' % MS,
                        'FA: %f' % FA,
                        'SER: %f' % SER,
                        'DER: %f' % DER_])
                logger.info(language_id+': '+msg)
                msg_list.append(language_id+': '+msg)
                
            else: 
                msg = 'There is no ref rttm file provided. Computing DER is Failed.'

        with open('{}/der_perlang.txt'.format(result_dir),'w') as f:
            for msg in msg_list:
                f.write(msg+'\n')
    except Exception as e:
        import traceback
        traceback.print_exc()
        import pdb; pdb.set_trace()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir',
                        type=str,
                        default="",
                        help="exp dir")
    parser.add_argument('--ref_rttm',
                        type=str,
                        default="",
                        help="ref rttm file")
    args = parser.parse_args()
    main(args)