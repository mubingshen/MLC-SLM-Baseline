

import argparse
from copy import deepcopy
from pyannote.audio import Model
from pyannote.database import registry, FileFinder
from pyannote.audio.tasks import SpeakerDiarization
import torch
import pytorch_lightning as pl
torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser(description='finetune pyannoate segmentation')
parser.add_argument('--dataset', default='./conf/finetune_pyannote_speaker_diarization.yml', type=str, help='pyannote segmentation config')
parser.add_argument('--devices', default=4, type=int, help='used devices')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--max_epoch', default=20, type=int, help='epoch num')
parser.add_argument('--output_path', default='./exp_finetune', type=str, help='save model path')
parser.add_argument('--checkpoint', default='pyannote/segmentation-3.0', type=str, help='load checkpoint or pretrain model')
parser.add_argument('--hf_access_token', default='', type=str, help='hf_access_token for pyannote/segmentation-3.0')
parser.add_argument('--max_speakers_per_chunk', default=4, type=int, help='max speaker num per audio chunk')
parser.add_argument('--max_speakers_per_frame', default=2, type=int, help='max speaker num per audio frame')

def main():
    args=parser.parse_args()

    print(f'load dataset...')
    registry.load_database(args.dataset)
    MCLSLM = registry.get_protocol('MLCSLM.SpeakerDiarization.FinetuneProtocol'\
                                , preprocessors={"audio": FileFinder()})
    print(f'dataset sample:')
    for resource in MCLSLM.train():
        print(resource["uri"])
        break

    segmentation_params = {
        'segmentation':args.checkpoint,
        'segmentation_batch_size':args.batch_size,
        'use_auth_token':args.hf_access_token,
    }

    print(f'loading pretrained model...')

    pretrained_model = Model.from_pretrained(
        segmentation_params['segmentation'],
        use_auth_token=segmentation_params['use_auth_token'], 
        strict=False,
    )

    print(f'defineing task...')
    seg_task = SpeakerDiarization(MCLSLM, duration=10.0, max_speakers_per_chunk=args.max_speakers_per_chunk \
                                  , max_speakers_per_frame=args.max_speakers_per_frame)

    finetuned = deepcopy(pretrained_model)
    finetuned.task = seg_task

    print(f'start training...')
    trainer = pl.Trainer(devices=args.devices, max_epochs=args.max_epoch, accelerator="gpu" \
                        , default_root_dir=args.output_path
                        , enable_progress_bar=True
                        , callbacks=[pl.callbacks.ModelCheckpoint(save_top_k=-1)]
                        )
    trainer.fit(finetuned)

if __name__ == '__main__':
    main()
