# EEG Hackathon

Hey hackers!

You have come to the right place to get started with EEG data analysis... in a fun way.
In this repository, we will provide you with the necessary tools to collect and analyze EEG data while playing a game by focussing on the game "Canabalt".
The goal is to process the EEG data to detect when the player is pressing a key and to classify the key-stroke based on a short signal window.

## Getting Started & Dependencies

General system requirements:

- Windows 10 or higher
- Bluetooth to connect to the EEG cap
- Anaconda or a different venv to run the Python code

### Install the EEG Software

We provided each team a Unicorn Hybrid Black EEG cap from g.tec. This is a wireless, easy to use EEG cap that we will use to collect EEG data.
The software to connect to the record EEG cap and stream the data can be downloaded from the g.tec website [here](https://www.gtec.at/product/unicorn-suite/).  

Once you installed the software, start the Unicorn suite, go to the tab "My Unicorn" and connect to the EEG cap.
Make sure that the EEG cap is charged and turned on. Also check your bluetooth settings to make sure that the EEG cap is connected to your PC.

Once the EEG cap is connected, go to the tab "Apps" and download the Unicorn Recorder.
If you want you can also install the Unicorn Bandpower for free for testing purposes.

### Install the Python Libraries

To handle the EEG data, we will use the MNE-Python library. This is a powerful library for processing and analyzing EEG data.
The provided code is written in Python. Please run the following to install the required libraries assuming you have (Anaconda)[https://www.anaconda.com/docs/getting-started/miniconda/install#power-shell] already installed:

```bash
conda create -n eeg python=3.13 -y
conda activate eeg
pip install -r requirements.txt
```

### Download the Game

The game that we are using is the J&R classic "Canabalt" - a simple endless runner where the player has to jump over obstacles by hitting nothing else then the space bar. \
This gives us a clean way to collect EEG data while the subject is playing the game, because the stimulus onset is clearly defined by the space bar press.

Please follow the instructions for the open source GitHub repo of the game [here](https://github.com/ninjamuffin99/canabalt-hf).

If you are facing issues compiling the game for windows, try compiling it for html5. You can then run the game in your browser but you need to install [node.js](https://nodejs.org/en/download) to run it.

