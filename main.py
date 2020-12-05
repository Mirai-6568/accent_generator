from utils.accent_extractor import AccentExtractor
from utils.fundamental_frequency import FeatureExtractor, FeatureComparator

input_wav = "./target/sample.wav"
input_phonemes = [] #example: [7 ,6 ,6 ,10 ,2 ,12, 11 ,6, 6, 11 ,2 ,12 ,8 ,4 ,3, 7 ,6 ,10 ,5, 14 ,11, 2, 10, 5 ,13 ,3 ,8, 6, 10, 2, 3, 11, 6, 12, 3 ,22 ,3, 3 ,15 ,4 ,11 ,5, 10 ,5 ,7, 3, 9, 3 ,17, 2 ,20, 4 ,24, 5 ,21, 3 ,3 ,16, 6, 11, 5 ,40]
duration = "./target/duration" #example: 4 14 11 2 9 5 7 4 9 6 7 7 0 6 9 5 6 3 12 24 6 7 4 5 5 4 5 4 3 7 5 7 7 8 7 8 4 7 5 5 7 3 3 3 6 4 4 5 3 5 7 2 9 3 4 7 4 4 7 7 3 0 8

wav2acc = AccentExtractor()
wav2f0 = FeatureExtractor()
comparator = FeatureComparator()

accent = wav2acc(input_wav, input_phonemes, duration, feature_extractor = wav2f0, feature_comparator = comparator)
print(accent)