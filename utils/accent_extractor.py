import os
import numpy as np

class AccentExtractor():
    def __call__(self, wav, phoneme, durations, feature_extractor, feature_comparator):
        os.makedirs('./dump', exist_ok = True)
        accent = np.zeros(len(phoneme))

        frequency_means = feature_extractor(wav, durations)
        accent_place = []
        for i in range(len(phoneme) - 1):
            if i == 0:
                if frequency_means[i] > frequency_means[i + 1]:
                    accent[i] = 1
                    accent_place.append(i)
            else:
                if frequency_means[i - 1] < frequency_means[i] > frequency_means[i + 1]:
                    accent[i] = 1
                    accent_place.append(i)

        for i in range(len(phoneme) - 3):
            if accent[i] == 1:
                if accent[i + 2] == 1 or accent[i + 3] == 1:
                    accent[i] = 0
                    accent_place.remove(i)

        for _ in range(1):
            accent = feature_comparator(wav, phoneme, accent, accent_place)

        return accent