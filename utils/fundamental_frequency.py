import subprocess
import os
import csv
import pickle

import torch
import crepe
from scipy.io import wavfile

class FeatureExtractor():
    def __init__(self, wav, phoneme, accent, duration, text2speech, dumpdir = './dump'):
        self.wav = wav
        self.phoneme = phoneme
        self.accent = accent
        self.duration = duration
        self.dumpdir = dumpdir
        self.text2speech = text2speech

        os.makedirs('./dump', exist_ok=True)

    def get_frequency(self, wav):
        sr, audio = wavfile.read(wav)
        _, frequency, _, _ = crepe.predict(audio, sr, viterbi=True)
        return frequency

    def get_realtime_duration(self):
        with open(self.duration) as f:
            durations_list = f.read().split()
        realtime_duration = [0]
        for i in range(len(durations_list)):
            realtime_duration.append(int(durations_list[i]) * 100 // (24000 // 300))
        return realtime_duration

    def get_frequency_means(self):
        frequency = self.get_frequency(self.wav)
        realtime_duration = self.get_realtime_duration()

        frequency_means = []
        sum_duration = 0
        for i in range(1, len(realtime_duration)):
            sum_duration += realtime_duration[i]
            realtime_duration[i] = sum_duration
            if realtime_duration[i] - realtime_duration[i - 1] == 0:
                frequency_means.append(0)
            else:
                sum_frequency = 0
                for j in range(realtime_duration[i - 1], realtime_duration[i]):
                    sum_frequency += frequency[j]
                frequency_mean = sum_frequency / (realtime_duration[i] - realtime_duration[i - 1])
                frequency_means.append(frequency_mean)

        return frequency_means

    def create_wav_and_get_frequency(self, accent):
        wavfile_path = os.path.join(self.dumpdir, 'temp.wav')
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        phoneme = torch.tensor(self.phoneme, dtype=torch.long, device=device)
        accent = torch.tensor(accent, dtype=torch.long, device=device)
        self.text2speech(wavfile_path, phoneme, accent)

        frequency = self.get_frequency(wavfile_path)
        return frequency

    def check_frequency_loss(self, new_accent):
        accent_frequency = self.get_frequency(self.wav)
        new_accent_frequency = self.create_wav_and_get_frequency(new_accent)
        length = min(len(accent_frequency), len(new_accent_frequency))

        loss = 0
        for i in range(length):
            loss += (accent_frequency[i] - new_accent_frequency[i]) ** 2
        loss = loss / length
        return loss

    def save_accent(self, accent, number):
        with open(os.path.join(self.dumpdir, 'data' + str(number) + '.pkl'), 'wb') as web:
            pickle.dump(accent, web)


class FeatureComparator():
    def __init__(self, wav, phoneme, accent, accent_place, duration, text2speech):
        self.wav = wav
        self.phoneme = phoneme
        self.accent = accent
        self.accent_place = accent_place
        self.feature_extractor = FeatureExtractor(wav, phoneme, accent, duration, text2speech)

    def create_new_accent(self, position, number):
        accent = self.accent
        n = list(str(bin(number)).replace('0b', '').zfill(5)) #example: 1 → [0,0,0,0,1], 6 → [0,0,1,1,0], 25 → [1,1,0,0,1]
        for k in range(5):
            if n[k] == '0' and 0 <= (position - 2 + k) < len(accent):
                accent[position - 2 + k] = 0
            if n[k] == '1' and 0 <= (position - 2 + k) < len(accent):
                accent[position - 2 + k] = 1
        return accent

    def __call__(self):
        revised_accent = self.accent
        for position in self.accent_place:
            score = []
            for number in range(32):
                new_accent = self.create_new_accent(position, number)
                loss = self.feature_extractor.check_frequency_loss(new_accent)
                score.append(loss)
            best_accent_number = score.index(min(score))

            revised_accent = self.create_new_accent(position, best_accent_number)
            self.feature_extractor.save_accent(revised_accent, position)

        return revised_accent

#    def check_updown_loss(self, phoneme, accent, new_accent):