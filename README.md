# MLC-SLM Baseline
Large Language Models (LLMs) have demonstrated remarkable capabilities across various downstream tasks, serving as powerful foundation models for language understanding and generation. Recently, there has been significant interest in applying LLMs to speech and audio processing tasks, including Automatic Speech Recognition (ASR), Audio Captioning, and emerging areas such as Spoken Dialogue Models.

However, the development of robust LLM-based Spoken Dialogue Models relies heavily on real-world conversational speech data, which encapsulates the complexity of human communication, including natural pauses, interruptions, speaker overlaps, and diverse conversational styles. The scarcity of such data, especially in multilingual contexts, poses a significant challenge to advancing the field.

The importance of real-world conversational speech extends beyond technological advancement—it is essential for building AI systems that can understand and respond naturally in multilingual, dynamic, and context-rich environments. This is especially crucial for next-generation human-AI interaction systems, where spoken dialogue is a primary communication mode.

Thus, this workshop aims to bridge the gap by hosting the challenge of building multilingual conversational speech language models (MLC-SLM) together with the release of a real-world multilingual conversational speech dataset.

The challenge consists of two tasks, both of which require participants to explore the development of speech language models (SLMs):

***Task I: Multilingual Conversational Speech Recognition***

***Task II: Multilingual Conversational Speech Diarization and Recognition***

This project provides baseline systems for the two tasks mentioned above!

## Update
* We calculate CER/tcpCER for Japanese, Korean and Thai, and WER/tcpWER for other languages in two tasks.
* In Task I, we use the meeteval toolkit to calculate the error rate, avoiding additional errors caused by the different toolkits for calculating the error rate in the two tasks.
* The code can refer to the last stage in `./examples/mlcslm/asr/run.sh` and `./examples/mlcslm/sdasr/infer_sd.sh`.

## Setup
* Clone the repo
```shell
git clone https://github.com/mubingshen/MLC-SLM-Baseline.git
```
* Install requirements
```shell
pip install -r requirements.txt
```
## Introduction

* [Task I](./examples/mlcslm/asr): Follow the detailed steps in `./examples/mlcslm/asr`
* [Task II](./examples/mlcslm/sdasr): Follow the detailed steps in `./examples/mlcslm/sdasr`

## To-do list
- [x] Task I ASR baseline with vanilla whisper-large-v3 encoder & Qwen2.5-7B
- [x] Task I ASR baseline with vanilla whisper-large-v3 encoder & Llama3.1-8B will be coming soon
- [x] Task II speaker diarization baseline will be coming soon

## Baseline Results on the Dev set
## Task I: Multilingual Conversational Speech Recognition

![task1_fig](./figs/task1.png)

**Baseline-Qwen**: Vanilla Whisper-large-v3 Encoder + Qwen2.5-7B

**Baseline-Llama**: Vanilla Whisper-large-v3 Encoder + Llama3.1-8B

**Training steps**:
* Step 1: Train the projector between the encoder and LLM
* Step 2: Load the projector trained in the first step, and then train the projector and LLM LoRA simultaneously

**Evaluation matrix**: Word Error Rate (WER) or Character Error Rate (CER)

| LID                | Vanilla Whisper-large-v3    | Baseline-Qwen | Baseline-Llama |
|--------------------|-----------------------------|---------------|----------------|
| English-American   | 14.14                       | 13.83         | 16.87          |
| English-Australian | 11.72                       | 11.19         | 13.32          |
| English-British    | 10.08                       | 11.00         | 10.97          |
| English-Filipino   | 9.20                        | 8.06          | 8.26           |
| English-Indian     | 13.96                       | 16.87         | 15.67          |
| French             | 26.72                       | 25.69         | 26.43          |
| German             | 20.53                       | 33.95         | 32.37          |
| Italian            | 17.94                       | 23.47         | 24.15          |
| Japanese           | 21.64                       | 34.74         | 33.82          |
| Korean             | 13.80                       | 20.77         | 22.56          |
| Portuguese         | 20.82                       | 34.02         | 33.91          |
| Russian            | 7.36                        | 18.25         | 19.07          |
| Spanish            | 12.24                       | 14.31         | 16.41          |
| Thai               | 14.49                       | 21.67         | 19.62          |
| Vietnamese         | 23.02                       | 21.50         | 22.92          |
| Avg.               | 15.36                       | 21.49         | 21.56          |

## Task II: Multilingual Conversational Speech Diarization and Recognition

![task1_fig](./figs/task2.png)

**Baseline**: 3D-Speaker Diarization + Task I pre-trained SLM model

**Training steps**:
* Step 1: Finetune the pyannote-segmetation module with `./examples/mlcslm/sdasr/finetune_sd.sh`
* Step 2: Load the segmentation module in the first step, and infer dev-set with `./examples/mlcslm/sdasr/infer_sd.sh`

**Evaluation matrix**: Diarization Error Rate (DER)

| LID                | w/o overlap 3D-Speaker | w/ overlap 3D-Speaker |
|--------------------|------------------------|-----------------------|
| English-American   | 20.18                  | 22.37                 |
| English-Australian | 13.76                  | 14.00                 |
| English-British    | 18.85                  | 19.52                 |
| English-Filipino   | 13.19                  | 12.67                 |
| English-Indian     | 8.19                   | 8.03                  |
| French             | 22.62                  | 23.50                 |
| German             | 22.33                  | 24.17                 |
| Italian            | 10.64                  | 11.55                 |
| Japanese           | 26.46                  | 26.32                 |
| Korean             | 23.25                  | 25.45                 |
| Portuguese         | 17.60                  | 17.99                 |
| Russian            | 11.37                  | 12.15                 |
| Spanish            | 12.92                  | 13.44                 |
| Thai               | 10.90                  | 11.32                 |
| Vietnamese         | 14.64                  | 15.30                 |
| Avg.               | 16.44                  | 17.16                 |

Time-Constrained minimum-Permutation Word Error Rate (tcpWER) or Character Error Rate (tcpCER) with collar = 5

| LID                | w/o overlap 3D-Speaker + Baseline-Llama | w/ overlap 3D-Speaker + Baseline-Llama |
|--------------------|-----------------------------------------|----------------------------------------|
| English-American   | 53.73                                   | 70.33                                  |
| English-Australian | 52.63                                   | 60.77                                  |
| English-British    | 71.92                                   | 77.17                                  |
| English-Filipino   | 50.37                                   | 58.87                                  |
| English-Indian     | 70.72                                   | 69.19                                  |
| French             | 96.04                                   | 104.54                                 |
| German             | 86.74                                   | 95.17                                  |
| Italian            | 83.31                                   | 84.01                                  |
| Japanese           | 71.30                                   | 88.63                                  |
| Korean             | 59.55                                   | 78.68                                  |
| Portuguese         | 118.84                                  | 106.89                                 |
| Russian            | 69.21                                   | 84.53                                  |
| Spanish            | 75.61                                   | 82.25                                  |
| Thai               | 83.56                                   | 78.02                                  |
| Vietnamese         | 82.80                                   | 91.96                                  |
| Avg.               | 76.12                                   | 81.85                                  |


## Contact US
* Bingshen Mu: `bsmu@mail.nwpu.edu.cn`.
* Zhaokai Sun: `zksun@mail.nwpu.edu.cn`.