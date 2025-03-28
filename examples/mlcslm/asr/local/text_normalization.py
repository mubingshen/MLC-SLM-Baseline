import re
import argparse

def text_normalization(input_text, output_text):
    custom_punctuations = r'!"#$%&()*+,./:;<=>?@[\\]^_`{|}~。、？！・¿¡，'
    punctuation_pattern = re.compile(f'[{re.escape(custom_punctuations)}]')
    with open(input_text, 'r', encoding='utf-8') as fin, open(output_text, 'w') as fout:
        lines = fin.readlines()
        for line in lines:
            parts = line.strip().split()
            uttid = parts[0]
            ori_text = ' '.join(parts[1:])
            ori_text = ori_text.lower()
            text_tn = punctuation_pattern.sub('', ori_text)
            text_tn = re.sub(r' +', ' ', text_tn)
            fout.write(f"{uttid} {text_tn}\n")
    fin.close()
    fout.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Text normalization for training and calculating error rate.')
    parser.add_argument('--input', type=str, help='The path of segment file')
    parser.add_argument('--output', type=str, help='The directory of splited wavs')
    args = parser.parse_args()
    
    input_text = args.input
    output_text = args.output
    
    text_normalization(input_text, output_text)