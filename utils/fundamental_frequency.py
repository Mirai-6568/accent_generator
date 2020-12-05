import subprocess
import os
import csv
import pickle

import torch
import crepe
from scipy.io import wavfile

from text2speech.utils.text2speech import text2speech_with_accent

class FeatureExtractor():
    def get_frequency(self, wav):
        sr, audio = wavfile.read(wav)
        _, frequency, _, _ = crepe.predict(audio, sr, viterbi=True)
        return frequency

    def get_realtime_duration(self, duration):
        with open(duration) as f:
            durations_list = f.read().split()
        realtime_duration = [0]
        for i in range(len(durations_list)):
            realtime_duration.append(int(durations_list[i]) * 100 // (24000 // 300))
        return realtime_duration

    def get_frequency_means(self, wav, duration):
        frequency = self.get_frequency(wav)
        realtime_duration = self.get_realtime_duration(duration)

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

    def create_wav_and_get_frequency(self, phoneme, accent, file_name='./dump/temp.wav'):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        phoneme = torch.tensor(phoneme, dtype=torch.long, device=device)
        accent = torch.tensor(accent, dtype=torch.long, device=device)
        text2speech_with_accent(file_name, phoneme, accent)

        frequency = self.get_frequency(file_name)
        return frequency

    def check_frequency_loss(self, phoneme, wav, new_accent):
        accent_frequency = self.get_frequency(wav)
        new_accent_frequency = self.create_wav_and_get_frequency(phoneme, new_accent)
        length = min(len(accent_frequency), len(new_accent_frequency))

        loss = 0
        for i in range(length):
            loss += (accent_frequency[i] - new_accent_frequency[i]) ** 2
        loss = loss / length
        return loss

#    def check_updown_loss(self, phoneme, accent, new_accent):

    def __call__(self, wav, duration):
        return self.get_frequency_means(wav, duration)


class FeatureComparator():
    def create_new_accent(self, accent, i, j):
        n = list(str(bin(j)).replace('0b', '').zfill(5)) #example: 1 → [0,0,0,0,1], 6 → [0,0,1,1,0], 25 → [1,1,0,0,1]
        for k in range(5):
            if n[k] == '0' and 0 <= (i - 2 + k) < len(accent):
                accent[i - 2 + k] = 0
            if n[k] == '1' and 0 <= (i - 2 + k) < len(accent):
                accent[i - 2 + k] = 1
        return accent

    def save_accent(self, accent, i):
        with open('./dump/data' + str(i) + '.pkl', 'wb') as web:
            pickle.dump(accent, web)

    def __call__(self, wav, phoneme, accent, accent_place):
        feature_extractor = FeatureExtractor()
        revised_accent = accent
        for i in accent_place:
            score = []
            for j in range(32):
                new_accent = self.create_new_accent(accent, i, j)
                loss = feature_extractor.check_frequency_loss(phoneme, wav, new_accent)
                score.append(loss)
            best_accent_number = score.index(min(score))

            revised_accent = self.create_new_accent(revised_accent, i, best_accent_number)
            self.save_accent(revised_accent, i)

        return revised_accent


class FeatureComparator2():
    def create_new_accent(self, accent, i, j):
        n = list(str(bin(j)).replace('0b', '').zfill(5)) #example: 1 → [0,0,0,0,1], 6 → [0,0,1,1,0], 25 → [1,1,0,0,1]
        for k in range(5):
            if n[k] == '0' and 0 <= (i - 2 + k) < len(accent):
                accent[i - 2 + k] = 0
            if n[k] == '1' and 0 <= (i - 2 + k) < len(accent):
                accent[i - 2 + k] = 1
        return accent

    def save_accent(self, accent, i):
        with open('./dump/log' + str(i) + '.pkl', 'wb') as web:
            pickle.dump(accent, web)

    def __call__(self, phoneme, accent, accent_place):
        feature_extractor = FeatureExtractor()
        revised_accent = accent
        for i in accent_place:
            score = []
            for j in range(32):
                new_accent = self.create_new_accent(accent, i, j)
                loss = feature_extractor.check_frequency_loss(phoneme, accent, new_accent)
                score.append(loss)
            best_accent_number = score.index(min(score))

            revised_accent = self.create_new_accent(revised_accent, i, best_accent_number)
            self.save_accent(revised_accent, i)

        return revised_accent