import numpy as np
import time
print('Imported np and time')
import torch
print('imported torch')
import torch.nn.functional as F
print('imported big F')
from pyKey import pressKey, releaseKey
import threading

from utils.utils import EEGListener  # Make sure this is working
from models.eegnet import EEGNet      # Adjust the import if needed


def trigger_key(key_name, delay=0.05):
    """Simulates a key press."""
    pressKey(key_name)
    time.sleep(delay)
    releaseKey(key_name)


class EEGKeyboard:
    def __init__(self, model, device='cpu', buffer_duration=1.0, pred_interval=0.2):
        self.sfreq = 250
        self.model = model
        self.device = device
        self.duration = buffer_duration
        self.pred_itv = pred_interval
        self.buffer_len = int(self.sfreq * self.duration)
        self.eeg_buffer = np.zeros((self.buffer_len, 8), dtype=np.float32)
        self.running = False
        self.prediction_thread = None

    def update_buffer(self, sample) -> None:
        """
        Update the EEG buffer with the new sample.
        """
        sample = sample[:-1]  # Remove last value if extra timestamp
        assert len(sample) == 8, "Sample must have 8 EEG channels"
        self.eeg_buffer[:-1] = self.eeg_buffer[1:]
        self.eeg_buffer[-1] = np.array(sample)

    def preprocess(self, window):
        """
        Preprocesses the EEG window to match EEGNet input shape.
        Returns: torch.Tensor of shape (1, 1, C, T)
        """
        window = window.T  # (C, T)
        tensor = torch.from_numpy(window).unsqueeze(0)  # (1, 1, C, T)
        return tensor.float().to(self.device)

    def predict(self):
        """
        Performs model prediction and triggers key if event detected.
        """
        eeg_window = self.eeg_buffer.copy()
        input_tensor = self.preprocess(eeg_window)

        with torch.no_grad():
            self.model.eval()
            output = self.model(input_tensor)
            probs = F.softmax(output, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()

        print(f"Prediction: {probs.cpu().numpy()} -> class {pred_class}")

        # Trigger action on specific class
        if pred_class == 1:
            trigger_key('SPACEBAR', 0.2)

    def _prediction_loop(self):
        while self.running:
            self.predict()
            time.sleep(self.pred_itv)

    def start_prediction(self):
        if self.running:
            return
        self.running = True
        self.prediction_thread = threading.Thread(target=self._prediction_loop, daemon=True)
        self.prediction_thread.start()

    def stop_prediction(self):
        self.running = False
        if self.prediction_thread:
            self.prediction_thread.join()


def load_model(model_path, device='cpu'):
    """
    Loads the trained EEGNet model from disk.
    """
    model = EEGNet(n_classes=2, n_channels=8, n_time=125)  # Adjust Samples if buffer window differs
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("✅ Model loaded.")
    return model


def main():
    model_path = "models\\eegnet_best_02_space.pth"  # Adjust path if needed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(model_path, device=device)
    eeg_keyboard = EEGKeyboard(model, device=device)

    stream_name = 'UnicornRecorderLSLStream'  # Confirm this name
    eeg_listener = EEGListener(stream_name=stream_name, callback=eeg_keyboard.update_buffer)

    eeg_listener.start()
    eeg_keyboard.start_prediction()

    try:
        while True:
            time.sleep(3)
    except KeyboardInterrupt:
        eeg_keyboard.stop_prediction()
        eeg_listener.stop()
        print("🛑 Stopped by user.")


if __name__ == '__main__':
    main()
