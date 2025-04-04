import glob
import os
import argparse
import numpy

parser=argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, default='./MLC-SLM_Workshop-Development_Set/data')
parser.add_argument('--output_path', type=str, default='./examples')
parser.add_argument('--dataset_part', type=str, default='dev')

def generate_file_lists(args):
    dataset_path=args.dataset_path
    '''
    -data
    --MLC-SLM_Workshop-Development_Set
    ----English
    -----American
    ------...
    '''
    ref_file_dict={}
    audio_file_dict={}
    uem_file_dict={}
    ref_file_lists=glob.glob(f'{dataset_path}/**/*.txt', recursive=True)
    audio_file_lists=glob.glob(f'{dataset_path}/**/*.wav', recursive=True)
    for ref_file in ref_file_lists:
        ref_data=numpy.genfromtxt(ref_file, dtype=str, delimiter='\t', encoding='utf-8')

        ref_file=ref_file.replace('.txt', '.wav')
        language_id=ref_file.split('/')[-2] if args.dataset_part=='dev' else ref_file.split('/')[-3]
        if language_id not in ref_file_dict:
            ref_file_dict[language_id]={}
        if language_id not in uem_file_dict:
            uem_file_dict[language_id]={}
        if ref_file not in ref_file_dict[language_id]:
            ref_file_dict[language_id][ref_file]=[]
        if ref_file not in uem_file_dict[language_id]:
            uem_file_dict[language_id][ref_file]=[]

        line_idx=0
        for line in ref_data:
            if len(line)!=0:
                '''
                target SPEAKER 2speakers_example 0 40.932 7.801 <NA> <NA> 2 <NA> <NA>
                origin 1.680273886484085       2.3401385301640696      O1      Hi Will
                '''
                ref_file_dict[language_id][ref_file].append(\
                    'SPEAKER {} 1 {} {} <NA> <NA> {} <NA> <NA>'.format(\
                    ref_file, round(float(line[0]), 4), round(float(line[1])-float(line[0]), 4), line[2]))
            if line_idx==len(ref_data)-1:
                # import pdb; pdb.set_trace()
                uem_file_dict[language_id][ref_file].append(\
                    '{} NA {} {}'.format(\
                    ref_file, 0.0, round(float(line[1]), 4)))
            line_idx+=1
            
    for audio_file in audio_file_lists:
        audio_file_dict[audio_file]=audio_file

    # import pdb; pdb.set_trace()
    # write audio lists
    with open(os.path.join(args.output_path, f'{args.dataset_part}_wav.lst'), 'w') as f:
        for line in audio_file_dict.values():
            f.write(f'{line}\n')
    # write ref files
    with open(os.path.join(args.output_path, f'{args.dataset_part}_rttm.rttm'), 'w') as f:
        for language, language_dict in ref_file_dict.items():
            for session, session_list in language_dict.items():
                for line in session_list:
                    f.write(f'{line}\n')
    # write ref lists
    with open(os.path.join(args.output_path, f'{args.dataset_part}_uem.uem'), 'w') as f:
        for language, language_dict in uem_file_dict.items():
            for session, session_list in language_dict.items():
                for line in session_list:
                    f.write(f'{line}\n')


if __name__=='__main__':
    args=parser.parse_args()
    generate_file_lists(args)
