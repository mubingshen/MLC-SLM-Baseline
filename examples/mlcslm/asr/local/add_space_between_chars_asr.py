import re
import argparse
from tqdm.auto import tqdm
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        help='input text file',
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        help="output text file"
    )

    return parser.parse_args()

def add_space_between_chars(text):
    pattern = re.compile(
        r"([\u1100-\u11ff\u2e80-\ua4cf\ua840-\uD7AF\uF900-\uFAFF\uFE30-\uFE4F\uFF65-\uFFDC\U00020000-\U0002FFFF\u3000-\u303F\uff01-\uff60\u0E00-\u0E7F])"
    ) # CJKT chars
    chars = pattern.split(text)
    chars = [ch for ch in chars if ch.strip()]
    text = " ".join(w for w in chars)
    text = re.sub(r"\s+", " ", text)
    return text

if __name__ == '__main__':
    args = get_args()
    with open(args.output, "w") as f:
        for line in tqdm(open(args.input, "r").readlines()):
            utt = line.strip().split()[0]
            if 'Japanese' in utt or 'Korean' in utt or 'Thai' in utt:
                text = ' '.join(line.strip().split()[1:])
                text_tn = add_space_between_chars(text)
            else:
                text_tn = ' '.join(line.strip().split()[1:])
            f.write(f"{utt} {text_tn}\n")
