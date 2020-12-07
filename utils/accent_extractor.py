import os
import numpy as np

class AccentExtractor():
    def __init__(self, wav, phoneme, duration, feature_extractor, feature_comparator, text2speech):
        self.wav = wav
        self.phoneme = phoneme
        self.duration = duration
        self.feature_extractor = feature_extractor
        self.feature_comparator = feature_comparator
        self.text2speech = text2speech

    def get_initial_accent(self):
        accent = np.zeros(len(self.phoneme))

        feature_extractor = self.feature_extractor(self.wav, self.phoneme, accent, self.duration, self.text2speech)
        frequency_means = feature_extractor.get_frequency_means()

        for i in range(len(accent) - 1):    #accent nucleus does not appear on the last phoneme
            if i == 0:  #for the first phoneme
                if frequency_means[i] > frequency_means[i + 1]:
                    accent[i] = 1
            else:   #for the other phonemes
                if frequency_means[i - 1] < frequency_means[i] > frequency_means[i + 1]:
                    accent[i] = 1

        for i in range(len(accent) - 3):   #each accent nucleus appears separately, at least three phonemes in between
            if accent[i] == 1:
                if accent[i + 2] == 1 or accent[i + 3] == 1:
                    accent[i] = 0

        return accent

    def __call__(self):
        accent = self.get_initial_accent()
        accent_place = [i for i, value in enumerate(accent) if value == 1]  #get the index of accent nucleus

        feature_comparator = self.feature_comparator(self.wav, self.phoneme, accent, accent_place, self.duration, self.text2speech)
        for _ in range(1):
            accent = feature_comparator()

        return accent