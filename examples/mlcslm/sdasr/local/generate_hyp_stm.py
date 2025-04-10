import os
import argparse

def generate_hyp_stm(rttm_dir, text, out_file):
    text_dict = {}
    with open(text, 'r') as fin:
        lines = fin.readlines()
        for line in lines:
            uttid = line.strip().split()[0]
            text = ' '.join(line.strip().split()[1:])
            text_dict[uttid] = text
    fin.close()
    
    rttm_files = {}
    for filename in os.listdir(rttm_dir):
        if filename.endswith('.rttm'):
            file_path = os.path.join(rttm_dir, filename)
            rttm_files[filename.split('.')[0]] = file_path
    
    with open(out_file, 'w') as fout:
        for rttm, rttm_path in rttm_files.items():
            file_name = rttm
            channel = 1
            with open(rttm_path, 'r') as fin:
                lines = fin.readlines()
                for line in lines:
                    start_time = float(line.strip().split()[3])
                    dur_time = float(line.strip().split()[4])
                    end_time = start_time + dur_time
                    speaker_id = line.strip().split()[7]
                    query_key = file_name + '-' + speaker_id + '-' + str(int(round(float(start_time), 2) * 100)).zfill(6)
                    for key in text_dict.keys():
                        if query_key in key:
                            transcript = text_dict[key]
                            fout.write(f"{file_name} {channel} {speaker_id} {start_time:.4f} {end_time:.4f} {transcript}\n")
                            break         
    fout.close()                
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate hyp stm')
    parser.add_argument('--rttm_dir', type=str, help='The directory of rttms')
    parser.add_argument('--text', type=str, help='The transcription file after text normalization')
    parser.add_argument('--out_file', type=str, help='The path of ref.stm')
    args = parser.parse_args()
    # STM :== <filename> <channel> <speaker_id> <begin_time> <end_time> <transcript>
    
    rttm_dir = args.rttm_dir
    text = args.text
    out_file = args.out_file
    
    generate_hyp_stm(rttm_dir, text, out_file)