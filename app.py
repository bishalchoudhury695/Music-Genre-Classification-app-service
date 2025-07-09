from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import LabelEncoder
import sounddevice as sd
import numpy as np
import librosa
import soundfile as sf
from tensorflow.keras.models import model_from_json # type: ignore
import json
import logging

app = Flask(__name__)

with open('ma.json', 'r') as json_file:
    model_json = json_file.read()
model = model_from_json(model_json)

#Load the model weights from the HDF5
model.load_weights('MSTCNN.weights.h5')

# Load the label encoder from the JSON
with open('label_encoder.json', 'r') as f:
    encoder_classes = json.load(f)

#Reconstruct the LabelEncoder object
encoder = LabelEncoder()
encoder.classes_ = np.array(encoder_classes)

#Load pre-trained model and label encoder
# model = load_model ('MSTCNN_model.h5*)
# with open ("label encoder-pkI', 'rb') as f:
#encoder = pickle.load(f)

# Define Parameters
SAMPLE_RATE = 22050
DURATION = 30
NUM_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512
EXPECTED_MFCC_LEN = 1320  # Expected number of frames in MFCC for the model

# Variable to store the last recorded audio
last_audio = None

# Record real-time audio
def record_audio(duration=DURATION, sr=SAMPLE_RATE):
    global last_audio
    print("Recording started...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1)
    sd.wait()  # Wait until recording is finished
    print("Recording finished.")
    last_audio = np.squeeze(audio) #Store the recorded audio for playback
    return last_audio

# Preprocess audio to MFCC with padding/truncation
def preprocess_audio(audio, sr=SAMPLE_RATE, num_mfcc=NUM_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH, expected_len=EXPECTED_MFCC_LEN):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
    # Pad or truncate to ensure consistent length
    if mfcc.shape[1] < expected_len:
        mfcc = np.pad(mfcc, ((0, 0), (0, expected_len - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :expected_len]
    return np.expand_dims(mfcc.T, axis=0)  # Add batch dimension for model input

# Classify genre of audio
# Configure lgging
logging.basicConfig(level=logging.INFO)

def classify_genre(audio_mfcc):
    logging.info("Classifying genre...")
    prediction = model.predict(audio_mfcc)
    logging.info(f"Prediction: {prediction}")
    genre_index = np.argmax(prediction, axis=1)
    genre_label = encoder.inverse_transform(genre_index)
    confidence = prediction[0][genre_index[0]] * 100  # Confidence in percentage
    logging.info(f"Predicted genre: {genre_label[0]}, Confidence: {confidence}%")
    return genre_label[0], confidence

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/record', methods=['POST'])
def record():
    audio = record_audio() # Record audio
    audio_mfcc = preprocess_audio(audio) # Process to MFCC
    genre, confidence = classify_genre(audio_mfcc) #Predict genre
    return jsonify({'genre': genre, 'confidence': confidence})

@app.route('/save', methods=['POST'])
def save():
    global last_audio
    if last_audio is not None:
        file_name = f"recording_{np.random.randint(1000, 9999)}.wav"
        sf.write(file_name, last_audio, SAMPLE_RATE) # Save using soundfile
        return jsonify({'message': f'Audio saved as {file_name}'})
    else:
        return jsonify({'error': 'No audio recorded to save.'}), 400

@app.route('/play', methods=['POST'])
def play():
    global last_audio
    if last_audio is not None:
        sd.play(last_audio, samplerate=SAMPLE_RATE) # play back the last recorded audio
        sd.wait()  # Wait until playback finishes
        return jsonify({'message': 'Playback finished.'})
    else:
        return jsonify({'error': 'No audio recorded to play.'}), 400

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
