import streamlit as st
import time
import st_audiorec
from st_audiorec import st_audiorec
import numpy as np
import pickle
import tempfile
import os
import librosa
from python_speech_features import mfcc

# inicijaliziranje session state-a za snimljeni zvuk i uploadani file
def init_session_state():
    if 'recorded_audio' not in st.session_state:
        st.session_state['recorded_audio'] = None
    if 'uploaded_files' not in st.session_state:
        st.session_state['uploaded_files'] = []

# spremi uploadani file u session state
def save_uploaded_file(uploaded_file):
    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        st.session_state['uploaded_files'].append((uploaded_file.name, file_bytes))
        st.write(f"Datoteka '{uploaded_file.name}' je prenesena i spremljena.")

# funkcija za ekstrahiranje audio signala
def extract_features(signal, rate):
    mfcc_features = mfcc(signal, rate, winlen=0.020, appendEnergy=False)
    covariance_matrix = np.cov(mfcc_features.T)
    mean_matrix = mfcc_features.mean(axis=0)
    return mean_matrix, covariance_matrix

# ucitavanje iztreniranog klasifikatora
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'knn_classifier.pkl')
    if os.path.exists(model_path):
        with open(model_path, 'rb') as file:
            trained_knn_classifier = pickle.load(file)
        return trained_knn_classifier
    else:
        st.error(f"Model file '{model_path}' not found. Please ensure the file is in the correct location.")
        return None

# pravljenje predikcije za zanr
def predict_genre(mean_matrix, covariance_matrix, classifier, genre_mapping):
    features = np.concatenate((mean_matrix, covariance_matrix.ravel())).reshape(1, -1)
    predicted_genre_index = classifier.predict(features)
    return genre_mapping[predicted_genre_index[0]]

# ovdje krece "main" za streamlit apk
def audiorec_ver1():
    # pokrecem session state
    init_session_state()

    # ucitavam model
    knn_classifier = load_model()

    # provjera je li model ucitan
    if knn_classifier is None:
        return

    # mapiranje zanrova za kasnije
    genre_mapping = {
        1: 'blues', 2: 'classical', 3: 'country', 4: 'disco', 
        5: 'hiphop', 6: 'jazz', 7: 'metal', 8: 'pop', 
        9: 'reggae', 10: 'rock'
    }

    # naslov i ostali HTML stuff
    st.title('Klasifikacija glazbe')
    st.write('## _Projekt iz PSU_')
    st.write('Jednostavna aplikacija za detekciju žanra glazbe')
    st.divider()
    st.write('### Snimi glazbu preko mikrofona:')

    # pozivanje audio recording UI
    wav_audio_data = st_audiorec()

    # spremam u session state snimljen zvuk
    if wav_audio_data is not None:
        st.session_state['recorded_audio'] = wav_audio_data

    # kada se klikne gumb poziva se predikcija za zanr koja je u session stateu
    if st.button('Pronadi zanr'):
        if st.session_state['recorded_audio'] is not None:
            st.write('Sada ću odgonetnuti žanr snimljene glazbe')
            time.sleep(2)
            with tempfile.NamedTemporaryFile(delete=False) as temp_wav_file:
                temp_wav_file.write(st.session_state['recorded_audio'])
                temp_wav_file.close()
                signal, rate = librosa.load(temp_wav_file.name, sr=22050)
                mean_matrix, covariance_matrix = extract_features(signal, rate)
                genre = predict_genre(mean_matrix, covariance_matrix, knn_classifier, genre_mapping)
                st.write(f'Tvoj žanr je: {genre}')
        else:
            st.write('Glazba bi bila poželjna, ovaj zvuk to nije.')

    st.divider()

    # UI za uploadanje fileova, samo radi sa wav datotekama
    with st.form("forma"):
        st.write('### Prenesi datoteku:')
        uploaded_file = st.file_uploader("Odaberi datoteku", type=['wav'], accept_multiple_files=False)
        submitted = st.form_submit_button("Žanriraj")
        if submitted and uploaded_file is not None:
            save_uploaded_file(uploaded_file)

    # uzimanje datoteke iz dijela di smo uploadali file
    if 'uploaded_files' in st.session_state and st.session_state['uploaded_files']:
        st.write('### Odaberi datoteku za klasifikaciju:')
        selected_files = st.multiselect('Odaberi datoteku:', [f[0] for f in st.session_state['uploaded_files']], max_selections=(1))

        #kada se odabere datoteke i klikne na gumb da se pokrene klasifikacija
        #zovem model i radi se predikcija
        if st.button('Pokreni klasifikaciju'):
            if selected_files:
                st.write('Sada ću odgonetnuti žanr odabrane datoteke')
                time.sleep(2)
                for file_name in selected_files:
                    file_bytes = next(f[1] for f in st.session_state['uploaded_files'] if f[0] == file_name)
                    with tempfile.NamedTemporaryFile(delete=False) as temp_wav_file:
                        temp_wav_file.write(file_bytes)
                        temp_wav_file.close()
                        signal, rate = librosa.load(temp_wav_file.name, sr=22050)
                        mean_matrix, covariance_matrix = extract_features(signal, rate)
                        genre = predict_genre(mean_matrix, covariance_matrix, knn_classifier, genre_mapping)
                        st.write(f"Datoteka: {file_name}, Tvoj žanr je: {genre}")

if __name__ == '__main__':
    audiorec_ver1()
