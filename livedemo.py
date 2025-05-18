# eeg_jump_game.py
# Real-time EEG-based jump action using EEGNet and the Unicorn Hybrid Black Pytorch

import numpy as np
import pyautogui
import time
from scipy.signal import butter, filtfilt
import torch
import torch.nn as nn
import torch.nn.functional as F

from unicorn_bin.udp import UnicornUDP

# === CONFIGURATION ===
SAMPLING_RATE = 250        # Unicorn device sampling rate
CHANNEL_COUNT = 8           # 8 EEG channels
WINDOW_SIZE = 250           # 1-second window at 250 Hz
SLIDE_STEP = 50             # Slide the window by 50 samples (~200ms refresh)
JUMP_THRESHOLD = 0.85       # Probability threshold for jump
MODEL_PATH = "best_model.h5"  # Path to trained EEGNet model

# === Load Trained Model ===
model = load_model('C:\\Users\\simeo\\Desktop\\RETEV2025\\models\\firstattempt91per.h5')
print("Model loaded.")

# === Butterworth Bandpass Filter ===
def bandpass_filter(data, low=2, high=60, fs=250, order=5):
    b, a = butter(order, [low / (fs / 2), high / (fs / 2)], btype='band')
    return filtfilt(b, a, data, axis=-1)

# === Trigger Spacebar Press ===
def trigger_jump():
    pyautogui.press("space")
    print("[Jump Triggered]")

# === Connect to Unicorn ===
unicorn = UnicornUDP()
unicorn.start()
print("Connected to Unicorn EEG.")

# === Real-Time Sliding Buffer ===
eeg_buffer = np.zeros((CHANNEL_COUNT, WINDOW_SIZE))

# === Main Real-Time Loop ===
try:
    while True:
        samples = unicorn.get_data()
        if not samples:
            continue

        # Append samples to buffer (assumes samples is shape [N, 8])
        for s in samples:
            eeg_buffer = np.roll(eeg_buffer, -1, axis=1)
            eeg_buffer[:, -1] = s[:CHANNEL_COUNT]

        # Slide window only every SLIDE_STEP samples
        if eeg_buffer.shape[1] % SLIDE_STEP == 0:
            # Filter
            filtered = bandpass_filter(eeg_buffer)

            # Reshape to model input (1, 8, 250, 1)
            eeg_input = filtered[np.newaxis, ..., np.newaxis]

            # Predict
            prediction = model.predict(eeg_input, verbose=0)
            jump_prob = prediction[0][1]  # assuming [no jump, jump]

            print(f"Jump Probability: {jump_prob:.2f}")

            if jump_prob > JUMP_THRESHOLD:
                trigger_jump()

        time.sleep(0.01)  # prevent CPU overload

except KeyboardInterrupt:
    print("\nInterrupted by user. Stopping...")
    unicorn.stop()
