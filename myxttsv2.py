from trainer import Trainer, TrainerArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig
from TTS.tts.models.xtts import XttsAudioConfig
from TTS.utils.manage import ModelManager

from datetime import datetime
import argparse
import random
import os

parser = argparse.ArgumentParser()
parser.add_argument('--ds_name', required=True)
parser.add_argument('--token', required=False, default=None)
parser.add_argument('--num_epochs', required=False, default=1)
parser.add_argument('--batch_size', required=False, default=16)
parser.add_argument('--grad_accum_steps', required=False, default=16)
parser.add_argument('--logger', required=False, default=None)
args = parser.parse_args()

ds_name = args.ds_name 
batch_size = args.batch_size
grad_accum_steps = args.grad_accum_steps
eval = args.eval
num_epochs = args.num_epochs
# Note: we recommend that batch_size * grad_accum_steps need to be at least 252 for more efficient training.
# You can increase/decrease batch_size but then set grad_accum_steps accordingly.

if args.token is not None:
    token = args.token
    os.environ['HF_TOKEN'] = token
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def rename_dirs(root_dir):
    month = datetime.now().strftime('%B')
    for d in os.listdir(root_dir):
        path = os.path.join(root_dir, d)
        if os.path.isdir(path) and month in d:
            new_name = d.split(f'-{month}')[0]
            os.rename(path, os.path.join(root_dir, new_name))
    print("Removing the -Month name from the dir because its annoying as fuck okay, fuck off")

# ------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------ #


# Define here the dataset that you want to use for the fine-tuning on.
config_dataset = BaseDatasetConfig(
    formatter="huggingface",
    dataset_name=ds_name,
    path=ds_name.split('/')[1] + "_mp3",
    meta_file_train=ds_name,
    language="en",
)

# Logging parameters
RUN_NAME = f"xtts_{random.randint(10_000, 99_999)}"
PROJECT_NAME = "XTTS_trainer"
DASHBOARD_LOGGER = args.logger if args.logger is not None else "tensorboard"
LOGGER_URI = None

# Set here the path that the checkpoints will be saved.
OUT_PATH = "run/xttsv2"

# Training Parameters
OPTIMIZER_WD_ONLY_ON_WEIGHTS = True  # for multi-gpu training please make it False

# Add here the configs of the datasets
DATASETS_CONFIG_LIST = [config_dataset]

# Define the path where XTTS v2.0.1 files will be downloaded
CHECKPOINTS_OUT_PATH = os.path.join(OUT_PATH, "XTTS_v2.0_original_model_files/")
os.makedirs(CHECKPOINTS_OUT_PATH, exist_ok=True)

# DVAE files
DVAE_CHECKPOINT_LINK = "https://huggingface.co/coqui/XTTS-v2/resolve/main/dvae.pth"
MEL_NORM_LINK = "https://huggingface.co/coqui/XTTS-v2/resolve/main/mel_stats.pth"

# Set the path to the downloaded files
DVAE_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(DVAE_CHECKPOINT_LINK))
MEL_NORM_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(MEL_NORM_LINK))

# download DVAE files if needed
if not os.path.isfile(DVAE_CHECKPOINT) or not os.path.isfile(MEL_NORM_FILE):
    print(" > Downloading DVAE files!")
    ModelManager._download_model_files([MEL_NORM_LINK, DVAE_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True)

# Download XTTS v2.0 checkpoint if needed
TOKENIZER_FILE_LINK = "https://huggingface.co/coqui/XTTS-v2/resolve/main/vocab.json"
XTTS_CHECKPOINT_LINK = "https://huggingface.co/coqui/XTTS-v2/resolve/main/model.pth"

# XTTS transfer learning parameters: You we need to provide the paths of XTTS model checkpoint that you want to do the fine tuning.
TOKENIZER_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(TOKENIZER_FILE_LINK))  # vocab.json file
XTTS_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(XTTS_CHECKPOINT_LINK))  # model.pth file

# download XTTS v2.0 files if needed
if not os.path.isfile(TOKENIZER_FILE) or not os.path.isfile(XTTS_CHECKPOINT):
    print(" > Downloading XTTS v2.0 files!")
    ModelManager._download_model_files(
        [TOKENIZER_FILE_LINK, XTTS_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True
    )

LANGUAGE = config_dataset.language


# ------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------ #


def main():
    # init args and config
    model_args = GPTArgs(
        max_conditioning_length=132300,  # 6 secs
        min_conditioning_length=66150,  # 3 secs
        debug_loading_failures=False,
        max_wav_length=255995*5,  # ~11.6 * 5 seconds
        max_text_length=200,
        mel_norm_file=MEL_NORM_FILE,
        dvae_checkpoint=DVAE_CHECKPOINT,
        xtts_checkpoint=XTTS_CHECKPOINT,  # checkpoint path of the model that you want to fine-tune
        tokenizer_file=TOKENIZER_FILE,
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )
    # define audio config
    audio_config = XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)
    # training parameters config
    config = GPTTrainerConfig(
        output_path=OUT_PATH,
        model_args=model_args,
        run_name=RUN_NAME,
        project_name=PROJECT_NAME,
        run_description="""
            GPT XTTS training
            """,
        dashboard_logger=DASHBOARD_LOGGER,
        logger_uri=LOGGER_URI,
        audio=audio_config,
        batch_size=batch_size,
        batch_group_size=0,
        eval_batch_size=batch_size,
        eval_split_max_size=256,
        num_loader_workers=1,        
        print_step=50,
        plot_step=1,
        log_model_step=1,
        save_step=10000,
        save_n_checkpoints=1,
        save_checkpoints=True,
        # target_loss="loss",
        print_eval=eval,
        epochs=num_epochs,
        training_seed=111,
        # Optimizer values like tortoise, pytorch implementation with modifications to not apply WD to non-weight parameters.
        optimizer="AdamW",
        optimizer_wd_only_on_weights=OPTIMIZER_WD_ONLY_ON_WEIGHTS,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=5e-06,  # learning rate
        lr_scheduler="MultiStepLR",
        # it was adjusted accordly for the new step scheme
        lr_scheduler_params={"milestones": [50000 * 18, 150000 * 18, 300000 * 18], "gamma": 0.5, "last_epoch": -1}
    )
    print(config)

    # init the model from config
    model = GPTTrainer.init_from_config(config)

    # load training samples
    train_samples, eval_samples = load_tts_samples(
        DATASETS_CONFIG_LIST,
        eval_split=eval,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )
    print(f"Train samples:{len(train_samples)}")
    print(f"Eval samples:{len(eval_samples) if eval else 0}")

    # init the trainer and 🚀
    trainer = Trainer(
        TrainerArgs(
            restore_path=None,  # xtts checkpoint is restored via xtts_checkpoint key so no need of restore it using Trainer restore_path parameter
            skip_train_epoch=False,
            start_with_eval=False,
            grad_accum_steps=grad_accum_steps,
        ),
        config,
        output_path=OUT_PATH,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()
    rename_dirs(OUT_PATH)


if __name__ == "__main__":
    main()
