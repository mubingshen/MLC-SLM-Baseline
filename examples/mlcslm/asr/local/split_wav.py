import soundfile as sf
import os
from tqdm import tqdm
import argparse


def split_audio(segments_file, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(segments_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    wav_scp_path = os.path.join(output_dir, 'wav.scp')
    text_path = os.path.join(output_dir, 'text')
    with open(wav_scp_path, 'w', encoding='utf-8') as wav_scp_file, open(text_path, 'w', encoding='utf-8') as text_file:
        for line in tqdm(lines, desc="Processing", unit="wavs"):
            parts = line.strip().split()
            record_id = parts[0]
            wav_path = parts[1]
            start_time = float(parts[2])
            end_time = float(parts[3])
            gt_text = ' '.join(parts[4:])

            try:
                audio_data, sample_rate = sf.read(wav_path)
                start_sample = int(start_time * sample_rate)
                end_sample = int(end_time * sample_rate)
                segment = audio_data[start_sample:end_sample]
                output_path = os.path.join(output_dir, f"{record_id}.wav")
                sf.write(output_path, segment, sample_rate)
                print(f"Saving {record_id}.wav")
                wav_scp_file.write(f"{record_id} {output_path}\n")
                text_file.write(f"{record_id} {gt_text}\n")
            except Exception as e:
                print(f"Error in saving {record_id}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split wavs using segments')
    parser.add_argument('--segments_path', type=str, help='The path of segment file')
    parser.add_argument('--output_dir', type=str, help='The directory of splited wavs')
    args = parser.parse_args()
    
    segments_path = args.segments_path
    output_dir = args.output_dir
    split_audio(segments_path, output_dir)
    