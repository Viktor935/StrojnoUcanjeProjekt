import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import mfcc
import os
import pickle
import random
from collections import defaultdict

def load_data(filename, split_ratio, train_set, test_set):
    with open(filename, 'rb') as file:
        while True:
            try:
                dataset.append(pickle.load(file))
            except EOFError:
                break
    for data in dataset:
        if random.random() < split_ratio:
            train_set.append(data)
        else:
            test_set.append(data)

dataset = []
train_data, test_data = [], []
load_data('features.dat', 0.66, train_data, test_data)

genre_mapping = defaultdict(int)

directory_path = "G:/G Radna/PSU projekt/Data/genres_original"

index = 1
for genre in os.listdir(directory_path):
    genre_mapping[index] = genre
    index += 1

def calculate_accuracy(test_set, predictions):
    correct_count = sum(1 for i in range(len(test_set)) if test_set[i][-1] == predictions[i])
    return correct_count / len(test_set)

with open('knn_classifier.pkl', 'rb') as file:
    trained_knn_classifier = pickle.load(file)

print("Početak predviđanja.")
num_test_instances = len(test_data)
predictions = []

for i, test_instance in enumerate(test_data):
    test_features = np.concatenate((test_instance[0], test_instance[1].ravel())).reshape(1, -1)
    predicted_class = trained_knn_classifier.predict(test_features)
    predictions.append(predicted_class[0])  
    
    print(f"Predviđeno {i+1}/{num_test_instances}: {predicted_class[0]}")

accuracy = calculate_accuracy(test_data, predictions)
print(f"Preciznost: {accuracy}")

specific_file_path = "G:\\G Radna\\PSU projekt\\test\\t1.wav"
rate, signal = wav.read(specific_file_path)
mfcc_features = mfcc(signal, rate, winlen=0.020, appendEnergy=False, nfft=1024)
covariance_matrix = np.cov(mfcc_features.T)
mean_matrix = mfcc_features.mean(axis=0)
feature_tuple = (mean_matrix, covariance_matrix, None)  #

specific_features = np.concatenate((feature_tuple[0], feature_tuple[1].ravel())).reshape(1, -1)
predicted_genre_index = trained_knn_classifier.predict(specific_features)
print(f"Predviđen žanr: {genre_mapping[predicted_genre_index[0]]}")

