import argparse

def generate_ref_stm(rttm, text, out_file):
    text_dict = {}
    with open(text, 'r') as fin:
        lines = fin.readlines()
        for line in lines:
            uttid = line.strip().split()[0]
            text = ' '.join(line.strip().split()[1:])
            text_dict[uttid] = text
    fin.close()
    
    rttm_list = []
    with open(rttm, 'r') as fin:
        lines = fin.readlines()
        for line in lines:
            rttm_list.append(line.strip())
    fin.close()
    
    with open(out_file, 'w') as fout:
        for rttm in rttm_list:
            
            file_name = rttm.split('/')[-1].split('.')[0]
            channel = 1
            with open(rttm, 'r') as fin:
                lines = fin.readlines()
                for line in lines:
                    start_time = float(line.strip().split()[3])
                    dur_time = float(line.strip().split()[4])
                    end_time = start_time + dur_time
                    speaker_id = line.strip().split()[7]
                    query_key = file_name.split('_')[0] + '-' + '_'.join(file_name.split('_')[1:]) + '-' + speaker_id + '-' + str(int(round(float(start_time), 2) * 100)).zfill(6)
                    for key in text_dict.keys():
                        if query_key in key:
                            transcript = text_dict[key]
                            fout.write(f"{file_name} {channel} {speaker_id} {start_time:.4f} {end_time:.4f} {transcript}\n")
                            break
    fout.close()                
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate reference stm')
    parser.add_argument('--rttm', type=str, help='The rttm file list')
    parser.add_argument('--text', type=str, help='The transcription file after text normalization')
    parser.add_argument('--out_file', type=str, help='The path of ref.stm')
    args = parser.parse_args()
    # STM :== <filename> <channel> <speaker_id> <begin_time> <end_time> <transcript>
    
    rttm = args.rttm
    text = args.text
    out_file = args.out_file
    
    generate_ref_stm(rttm, text, out_file)