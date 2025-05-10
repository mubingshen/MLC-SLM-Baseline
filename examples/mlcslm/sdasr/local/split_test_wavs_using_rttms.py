import soundfile as sf
import os
from tqdm import tqdm
import argparse


def split_wavs_using_rttms(rttm_dir, test_data_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    rttm_files = {}
    for filename in os.listdir(rttm_dir):
        if filename.endswith('.rttm'):
            file_path = os.path.join(rttm_dir, filename)
            rttm_files[filename.split('.')[0]] = file_path
            
    languages = ['English', 'French', 'German', 'Italian', 'Japanese', 'Korean', 'Portuguese', 'Russian', 'Spanish', 'Thai', 'Vietnamese']
    wav_files = {}
    for lang in languages:
        lang_dir = os.path.join(test_data_dir, lang)
        for root, _, files in os.walk(lang_dir):
            for filename in files:
                if filename.endswith('.wav'):
                    file_path = os.path.join(root, filename)
                    wavname = (root.split('/')[-1] + '_' + filename).split('.')[0]
                    wav_files[wavname] = file_path
    
    assert len(rttm_files) == len(wav_files), f"Number of rttm files {len(rttm_files)} does not match number of wav files {len(wav_files)}"
    
    wav_scp_path = os.path.join(output_dir, 'wav.scp')
    text_path = os.path.join(output_dir, 'text')
    with open(wav_scp_path, 'w') as fscp, open(text_path, 'w') as ftext:
        for key in tqdm(wav_files.keys(), desc="Processing", unit="wavs"):
            wav_file = wav_files[key]
            rttm_file = rttm_files[key]
            wav_data, sample_rate = sf.read(wav_file)
            with open(rttm_file, 'r') as fin:
                lines = fin.readlines()
                for line in lines:
                    start_time = float(line.strip().split()[3])
                    start_sample = int(start_time * sample_rate)
                    start_time_rounded = round(start_time, 2)
                    start_time_str = str(int(start_time_rounded * 100)).zfill(6)
                    
                    dur_time = float(line.strip().split()[4])
                    
                    end_time = start_time + dur_time
                    end_sample = int(end_time * sample_rate)
                    end_time_rounded = round(end_time, 2)
                    end_time_str = str(int(end_time_rounded * 100)).zfill(6)
                    
                    spkid = line.strip().split()[7]
                    record_id = f"{key}-{spkid}-{start_time_str}-{end_time_str}"
                    output_path = os.path.join(output_dir, f"{record_id}.wav")
                    segment = wav_data[start_sample:end_sample]
                    if len(segment) == 0:
                        print(f"Warning: Segment {record_id} has zero length.")
                        continue
                    sf.write(output_path, segment, sample_rate)
                    fscp.write(f"{record_id} {output_path}\n")
                    ftext.write(f"{record_id} None\n")
            fin.close()
    fscp.close()
    ftext.close()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split wavs using predicted rttms')
    parser.add_argument('--rttm_dir', type=str, help='The directory of rttms')
    parser.add_argument('--test_data_dir', type=str, help='The directory of test sets')
    parser.add_argument('--output_dir', type=str, help='The directory of splited wavs')
    args = parser.parse_args()
    
    rttm_dir = args.rttm_dir
    test_data_dir = args.test_data_dir
    output_dir = args.output_dir
    
    split_wavs_using_rttms(rttm_dir, test_data_dir, output_dir)
    