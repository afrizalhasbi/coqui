from trainer import Trainer, TrainerArgs
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsAudioConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

from datetime import datetime
import argparse
import random
import os

parser = argparse.ArgumentParser()
parser.add_argument('--ds_name', required=True)
parser.add_argument('--token', required=False, default=None)
parser.add_argument('--num_epochs', required=False, default=1)
parser.add_argument('--batch_size', required=False, default=16)
parser.add_argument('--lr', required=False, default="0.001")
parser.add_argument('--grad_accum_steps', required=False, default=1)
parser.add_argument('--logger', required=False, default=None)
parser.add_argument('--eval', required=False, default=True)

args = parser.parse_args()

ds_name = args.ds_name 
batch_size = args.batch_size
lr = float(args.lr)
grad_accum_steps = args.grad_accum_steps
num_epochs = args.num_epochs
eval = args.eval

run_id = random.randint(10_000, 99_999)
def rename_dirs(root_dir, run_id):
    month = datetime.now().strftime('%B')
    for d in os.listdir(root_dir):
        path = os.path.join(root_dir, d)
        if os.path.isdir(path) and month in d:
            new_name = d.split(f'-{month}')[0]
            os.rename(path, os.path.join(root_dir, f"{new_name}_{run_id}"))
    print("Removing the -Month name from the dir because its annoying as fuck okay, fuck off")

# ------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------ #


if args.token is not None:
    token = args.token
    os.environ['HF_TOKEN'] = token
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

dataset_config = BaseDatasetConfig(
    formatter="huggingface",
    dataset_name=ds_name,
    path=ds_name.split('/')[1] + "_mp3",
    meta_file_train=ds_name,
    language="en",
)
audio_config = VitsAudioConfig(
    sample_rate=22050, win_length=1024, hop_length=256, num_mels=80, mel_fmin=0, mel_fmax=None
)

config = VitsConfig(
    audio=audio_config,
    run_name=f"vits_{run_id}",
    batch_size=batch_size,
    eval_batch_size=batch_size if eval else 0,
    batch_group_size=1,
    num_loader_workers=1,
    num_eval_loader_workers=1 if eval else 0,
    run_eval=eval,
    lr=lr,
    test_delay_epochs=-1,
    epochs=num_epochs,
    text_cleaner="basic_cleaners",
    use_phonemes=False,
    compute_input_seq_cache=True,
    print_step=25,
    print_eval=eval,
    mixed_precision=False,
    output_path="run",
    datasets=[dataset_config],
    cudnn_benchmark=False,
)

# INITIALIZE THE AUDIO PROCESSOR
# Audio processor is used for feature extraction and audio I/O.
# It mainly serves to the dataloader and the training loggers.
ap = AudioProcessor.init_from_config(config)

# INITIALIZE THE TOKENIZER
# Tokenizer is used to convert text to sequences of token IDs.
# config is updated with the default characters if not defined in the config.
tokenizer, config = TTSTokenizer.init_from_config(config)

# LOAD DATA SAMPLES
# Each sample is a list of ```[text, audio_file_path, speaker_name]```
# You can define your custom sample loader returning the list of samples.
# Or define your custom formatter and pass it to the `load_tts_samples`.
# Check `TTS.tts.datasets.load_tts_samples` for more details.
train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=eval,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)

# init model
model = Vits(config, ap, tokenizer, speaker_manager=None)

print(dataset_config)
print(f"Train samples:{len(train_samples)}")
print(f"Eval samples:{len(eval_samples) if eval else 0}")

# ------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------ #


# init the trainer and 🚀
trainer = Trainer(
    TrainerArgs(grad_accum_steps=grad_accum_steps),
    config,
    "run/vits",
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)
trainer.fit()
# rename_dirs('run/vits', run_id)
