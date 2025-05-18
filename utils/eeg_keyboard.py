import numpy as np
import time
from pyKey import pressKey, releaseKey, showKeys
import threading

from utils.utils import EEGListener


def trigger_key(key_name, delay=0.05):
    """ This function simulates a key press """
    pressKey(key_name)
    time.sleep(delay)
    releaseKey(key_name)


class EEGKeyboard:

    

    def __init__(self, model, buffer_duration=0.5, pred_interval=0.1):
        self.sfreq = 250
        self.model = model
        self.duration = buffer_duration
        self.pred_itv = pred_interval
        self.buffer_len = int(self.sfreq * self.duration)
        self.eeg_buffer = np.zeros((self.buffer_len, 8), dtype=np.float32)
        self.running = False
        self.prediction_thread = None

    def update_buffer(self, sample) -> None:
        """
        Update the EEG buffer with the new sample.
        
        Parameters:
            sample (list): The incoming EEG data sample.
        """
        sample = sample[:-1]
        assert len(sample) == 8, "Sample must have 8 channels"

        # Shift the buffer to make room for the new sample
        self.eeg_buffer[:-1] = self.eeg_buffer[1:]
        # Add the new sample to the end of the buffer
        self.eeg_buffer[-1] = np.array(sample)
    
    def predict(self):
        
        eeg_window = self.eeg_buffer.copy()
        pred = self.model.predict(eeg_window)
        pred = np.squeeze(pred)

        # most likely class - assumes the output is a probability distribution
        #ml_class = np.argmax(pred, axis=1)[0]
        ml_class = pred
        print(pred)

        # TODO assign the class to the key based on your model

        if ml_class == 1:
            trigger_key('SPACEBAR',0.2)
        
        #else: do nothing - no key press detected
       
    def _prediction_loop(self):
        """
        Internal method to run predictions in a loop.
        """
        while self.running:
            self.predict()
            time.sleep(self.pred_itv)

    def start_prediction(self):
        """
        Start the prediction loop in a daemon thread.
        """
        if self.running:
            return  # Already running

        self.running = True
        self.prediction_thread = threading.Thread(target=self._prediction_loop, daemon=True)
        self.prediction_thread.start()

    def stop_prediction(self):
        """
        Stop the prediction loop.
        """
        self.running = False
        if self.prediction_thread:
            self.prediction_thread.join()
       

class Trashy:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def predict(self, x):
        """
        Predicts the class based on a threshold.

        Parameters:
            x (numpy.ndarray): The input data.

        Returns:
            numpy.ndarray: The predicted class (0 or 1).
        """
        print(x)
        arr = np.array(x)
        first_col = arr[:, 0]
        avg = np.mean(first_col)
        #print(avg)
        return avg > 4.3 and avg < 7

def main():
    # Load the model 
    model = Trashy(2)  # TODO load your model here

    # Create an EEGKeyboard instance
    eeg_keyboard = EEGKeyboard(model)

    # create EEG data stream
    stream_name = 'UnicornRecorderRawDataLSLStream'
    stream_name = 'UnicornRecorderLSLStream'
    eeg_listener = EEGListener(stream_name=stream_name, callback=eeg_keyboard.update_buffer)

    # start the prediction loop
    eeg_listener.start()
    eeg_keyboard.start_prediction()

    # keep programm running
    try:
        while True:
            time.sleep(3)
    except KeyboardInterrupt:
        eeg_keyboard.stop_prediction()
        eeg_listener.stop()



if __name__ == '__main__':
    main()


