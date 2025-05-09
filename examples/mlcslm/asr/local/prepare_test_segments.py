import os
import argparse

def process_txt_files(root_dir):
    results = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                wav_path = file_path.rsplit('.', 1)[0] + '.wav'
                relative_path = os.path.relpath(file_path, root_dir)
                parts = relative_path.split(os.sep)
                if len(parts) >= 2:
                    record_id = "-".join(parts[:-1] + [os.path.splitext(parts[-1])[0]])
                else:
                    record_id = os.path.splitext(relative_path)[0]

                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 3:
                            start_time = float(parts[0])
                            end_time = float(parts[1])
                            spkid = parts[2]
                            gt_text = "This is a test text"  # Placeholder for actual text
                            start_time_rounded = round(start_time, 2)
                            end_time_rounded = round(end_time, 2)
                            start_time_str = str(int(start_time_rounded * 100)).zfill(6)
                            end_time_str = str(int(end_time_rounded * 100)).zfill(6)
                            combined_value = f"{record_id}-{spkid}-{start_time_str}-{end_time_str}"
                            results.append((combined_value, wav_path, start_time_rounded, end_time_rounded, gt_text))
    return results

def write_to_segments(segments_path, segment):
    with open(segments_path, 'w', encoding='utf-8') as f:
        for seg_id, wav_path, start, end, gt_text in segment:
            f.write(f"{seg_id} {wav_path} {start} {end} {gt_text}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate original wav.scp')
    parser.add_argument('--data_dir', type=str, help='Data directory')
    parser.add_argument('--segments_path', type=str, help='The path of segment file')
    args = parser.parse_args()

    data_dir = args.data_dir
    segments_path = args.segments_path
    
    segment = process_txt_files(data_dir)
    write_to_segments(segments_path, segment)
    