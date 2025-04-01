# MLC-SLM Baseline
Large Language Models (LLMs) have demonstrated remarkable capabilities across various downstream tasks, serving as powerful foundation models for language understanding and generation. Recently, there has been significant interest in applying LLMs to speech and audio processing tasks, including Automatic Speech Recognition (ASR), Audio Captioning, and emerging areas such as Spoken Dialogue Models.

However, the development of robust LLM-based Spoken Dialogue Models relies heavily on real-world conversational speech data, which encapsulates the complexity of human communication, including natural pauses, interruptions, speaker overlaps, and diverse conversational styles. The scarcity of such data, especially in multilingual contexts, poses a significant challenge to advancing the field.

The importance of real-world conversational speech extends beyond technological advancement—it is essential for building AI systems that can understand and respond naturally in multilingual, dynamic, and context-rich environments. This is especially crucial for next-generation human-AI interaction systems, where spoken dialogue is a primary communication mode.

Thus, this workshop aims to bridge the gap by hosting the challenge of building multilingual conversational speech language models (MLC-SLM) together with the release of a real-world multilingual conversational speech dataset.

The challenge consists of two tasks, both of which require participants to explore the development of speech language models (SLMs):

***Task I: Multilingual Conversational Speech Recognition***

***Task II: Multilingual Conversational Speech Diarization and Recognition***

This project provides baseline systems for the two tasks mentioned above!

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

* [Task I](./examples/mlcslm/asr): Follow the detailed steps in `./examples/mlcslm/asr`. 
* [Task II](./examples/mlcslm/sdasr): Follow the detailed steps in `./examples/mlcslm/sdasr`.

## To do list
- [x] Task I baseline with whisper-large-v3 encoder & Qwen2.5-7B
- [ ] Task I baseline with whisper-large-v3 encoder & llama3.1-8B will be coming soon
- [ ] Task II speaker diarization baseline will be coming soon

## Baseline Results
## Task I: Multilingual Conversational Speech Recognition
Baseline：Vanilla Whisper-large-v3 Encoder + Qwen2.5-7B

Evaluation matrix: Word Error Rate (WER) or Character Error Rate (CER)

| LID                | Vanilla Whisper-large-v3    | Baseline |
|--------------------|-----------------------------|----------|
| English-American   | 14.14                       | 14.04    |
| English-Australian | 11.72                       | 11.60    |
| English-British    | 10.08                       | 11.37    |
| English-Filipino   | 9.20                        | 8.15     |
| English-Indian     | 13.96                       | 17.73    |
| French             | 26.72                       | 25.33    |
| German             | 20.53                       | 36.64    |
| Italian            | 17.94                       | 24.22    |
| Japanese           | 21.64                       | 34.88    |
| Korean             | 13.80                       | 20.60    |
| Portuguese         | 20.82                       | 36.09    |
| Russian            | 7.36                        | 7.51     |
| Spanish            | 12.24                       | 15.00    |
| Thai               | 14.49                       | 23.10    |
| Vietnamese         | 23.02                       | 18.22    |
| Avg.               | 15.36                       | 19.82    |

