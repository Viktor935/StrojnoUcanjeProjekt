import numpy as np #za numeričke operacije
import scipy.io.wavfile as wav #za čitanje .wav datoteka
from python_speech_features import mfcc #za ekstrahiranje MFCC karakteristika
import os #za rukovanje datotekama
import pickle #za serijalizaciju
import random #za nasumične vrijednosti
from collections import defaultdict #za stvaranje riječnika
import operator #za sortiranje
from sklearn.neighbors import KNeighborsClassifier #za KNN klasifikator

def compute_distance(instance1, instance2, k):
    mm1, cm1 = instance1[0], instance1[1]
    mm2, cm2 = instance2[0], instance2[1]
    inv_cm2 = np.linalg.inv(cm2)
    term1 = np.trace(np.dot(inv_cm2, cm1))
    term2 = np.dot(np.dot((mm2 - mm1).T, inv_cm2), mm2 - mm1)
    term3 = np.log(np.linalg.det(cm2)) - np.log(np.linalg.det(cm1))
    return term1 + term2 + term3 - k

def find_neighbors(train_set, test_instance, k):
    distances = []
    for data in train_set:
        dist = compute_distance(data, test_instance, k) + compute_distance(test_instance, data, k)
        distances.append((data[2], dist))
    distances.sort(key=operator.itemgetter(1))
    return [distances[i][0] for i in range(k)]

def get_majority_class(neighbors):
    vote_count = {}
    for neighbor in neighbors:
        if neighbor in vote_count:
            vote_count[neighbor] += 1
        else:
            vote_count[neighbor] = 1
    sorted_votes = sorted(vote_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_votes[0][0]

def calculate_accuracy(test_set, predictions):
    correct_count = sum(1 for i in range(len(test_set)) if test_set[i][-1] == predictions[i])
    return correct_count / len(test_set)

data_directory = 'G:/G Radna/PSU projekt/Data/genres_original'
output_file = open("features.dat", "wb")
genre_index = 0

print("Počni ekstrahiranje MFCC vrijednosti.")

for genre_folder in os.listdir(data_directory):
    genre_index += 1
    if genre_index == 11:
        break
    genre_folder_path = os.path.join(data_directory, genre_folder)
    print(f"Obrada žanra: {genre_folder}")
    for audio_file in os.listdir(genre_folder_path):
        try:
            audio_path = os.path.join(genre_folder_path, audio_file)
            rate, signal = wav.read(audio_path)
            mfcc_features = mfcc(signal, rate, winlen=0.020, appendEnergy=False)
            covariance_matrix = np.cov(mfcc_features.T)
            mean_matrix = mfcc_features.mean(axis=0)
            feature_tuple = (mean_matrix, covariance_matrix, genre_index)
            pickle.dump(feature_tuple, output_file)
        except Exception as e:
            print(f"Exception: {e} in folder: {genre_folder}, file: {audio_file}")
output_file.close()
print("Extrahiranje završeno.")

# Load dataset and split into training and testing sets
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

print(f"Total : {len(dataset)}")
print(f"Training set : {len(train_data)}")
print(f"Test set : {len(test_data)}")


genre_mapping = defaultdict(int)

directory_path = "G:/G Radna/PSU projekt/Data/genres_original"

index = 1
for genre in os.listdir(directory_path):
    genre_mapping[index] = genre
    index += 1


# Treniranje KNN klasifikatora
train_features = np.array([np.concatenate((data[0], data[1].ravel())) for data in train_data])
train_labels = np.array([data[2] for data in train_data])
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(train_features, train_labels)


# Spremanje treniranog klasifikatora
with open('knn_classifier.pkl', 'wb') as file:
    pickle.dump(knn_classifier, file)

# Učitavanje treniranog klasifikatora
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

specific_file_path = 'G:\G Radna\PSU projekt\\test\\t1.wav'
rate, signal = wav.read(specific_file_path)
mfcc_features = mfcc(signal, rate, winlen=0.020, appendEnergy=False)
covariance_matrix = np.cov(mfcc_features.T)
mean_matrix = mfcc_features.mean(axis=0)
feature_tuple = (mean_matrix, covariance_matrix, None)

specific_features = np.concatenate((feature_tuple[0], feature_tuple[1].ravel())).reshape(1, -1)
predicted_genre_index = trained_knn_classifier.predict(specific_features)
print(f"Predviđen žanr: {genre_mapping[predicted_genre_index[0]]}")

