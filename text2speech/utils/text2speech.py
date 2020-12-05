import os
import time

import soundfile as sf
import torch
import yaml
import argparse

from espnet2.tts.fastspeech2_accent import FastSpeech2Accent
from parallel_wavegan.utils import load_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_cofing_dir = os.path.join(BASE_DIR, 'model_config')

text2mel_config_path = os.path.join(model_cofing_dir, 'text2mel', 'fastspeech2.yaml')
text2mel_pretrained_path = os.path.join(model_cofing_dir, 'text2mel', 'fastspeech2.train.loss.ave.pth')
with open(text2mel_config_path) as f:
    text2mel_config = yaml.load(f, Loader=yaml.Loader)

mel2speech_config_path = os.path.join(model_cofing_dir, 'mel2speech', 'pwg_config.yaml')
mel2speech_pretrained_path = os.path.join(model_cofing_dir, 'mel2speech', 'pwg_jsut-400000steps.pkl')
with open(mel2speech_config_path) as f:
    mel2speech_config = yaml.load(f, Loader=yaml.Loader)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def retrieve_idim_size(token_list):

    if isinstance(token_list, str):
        with open(token_list, encoding="utf-8") as f:
            token_list = [line.rstrip() for line in f]
        # "args" is saved as it is in a yaml file by BaseTask.main().
        # Overwriting token_list to keep it as "portable".
        token_list = token_list.copy()
    elif isinstance(token_list, (tuple, list)):
        token_list = token_list.copy()
    else:
        raise RuntimeError("token_list must be str or dict")
    vocab_size = len(token_list)

    return vocab_size


def get_text2mel_model():

    args = argparse.Namespace(**text2mel_config)
    idim = retrieve_idim_size(args.token_list)
    model = FastSpeech2Accent(idim=idim, odim=80, **args.tts_conf).to(device)

    def extract_pretrained_parameters(pretrained_dict):
        new_pretrain_dict = {}
        for k, v in pretrained_dict.items():
            if k.startswith('tts.'):
                new_pretrain_dict[k.replace('tts.', '')] = v
        return new_pretrain_dict

    pretrained_dict = torch.load(text2mel_pretrained_path, map_location=device)
    model.load_state_dict(extract_pretrained_parameters(pretrained_dict))

    model.eval()

    return model


def get_mel2speech_model():

    mel2speech_model = load_model(mel2speech_pretrained_path, mel2speech_config)
    mel2speech_model.remove_weight_norm()

    mel2speech_model = mel2speech_model.eval().to(device)

    return mel2speech_model


text2mel_model = get_text2mel_model()
mel2speech_model = get_mel2speech_model()


def text2speech_with_accent(file_name: str, phoneme: torch.Tensor, accent: torch.Tensor):

    '''

    :param file_name: wav file name created by this function
    :param phoneme: Long Tensor of pytorch
    :param accent: Long Tensor of pytorch

    '''

    mel, _, _ = text2mel_model.inference(phoneme, accent)

    with torch.no_grad():
        start = time.time()
        y = mel2speech_model.inference(mel).view(-1)
        rtf = (time.time() - start) / (len(y) / mel2speech_config["sampling_rate"])
        print(f'{file_name} rtf:{rtf}')
        # save as PCM 16 bit wav file
        sf.write(file_name,
                 y.cpu().numpy(),
                 mel2speech_config["sampling_rate"],
                 "PCM_16")


if __name__ == '__main__':

    phoneme = torch.tensor(
        [13, 4, 26, 5, 3, 13, 2, 10, 6, 6, 15, 4, 2, 7, 2, 10, 2, 7, 2, 18, 2, 8, 2, 7, 5, 9, 6, 18, 2, 8, 2, 10,
         2, 8, 2, 4, 8, 3, 16, 6, 11, 5, 40], dtype=torch.long, device=device)
    accent = torch.tensor(
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.long, device=device)

    text2speech_with_accent('a', phoneme, accent)
