# EEG Hackathon 🧠🎮

Hey hackers! 👋

Welcome to the EEG Hackathon repository! You've come to the right place to dive into EEG data analysis in a fun and engaging way.
In this repository, you'll find all the tools you need to collect and analyze EEG data while playing the game "Canabalt."

The goal of this project is to process EEG data to detect when a player presses a key and classify the keystroke based on a short signal window. Let’s get started!

## 🚀 Getting Started & Dependencies

### General System Requirements ⚙️

Before you begin, ensure your system meets the following requirements:

- Operating System: Windows 10 or later
- Hardware: Bluetooth capability (to connect to the EEG cap)
- Python Environment: Anaconda (recommended) or an alternative virtual environment manager

### Install the EEG Software 🖥️

Each team has been provided with a Unicorn Hybrid Black EEG cap from g.tec. This wireless, user-friendly EEG cap will be used for data collection.

To set it up:

1. Download the Unicorn Suite software from the g.tec website.
1. Install the software and launch it.
1. Navigate to the "My Unicorn" tab and connect to your EEG cap:
    - Ensure the EEG cap is fully charged and powered on.
    - Check your Bluetooth settings to confirm that your PC is paired with the cap.
1. Once connected, go to the "Apps" tab in Unicorn Suite and download Unicorn Recorder.
    - Optionally, you can also install Unicorn Bandpower for testing purposes.

### Install Python Libraries 🐍

We’ll use [MNE-Python](https://mne.tools/stable/index.html), a powerful library for processing and analyzing EEG data.

Follow these steps to set up your Python environment:

- Ensure you have [Anaconda](https://www.anaconda.com/docs/getting-started/miniconda/install#power-shell) installed on your system.
- Run the following commands in your terminal or PowerShell:

```bash
conda create -n eeg python=3.13 -y
conda activate eeg
pip install -r requirements.txt
This will create a virtual environment named eeg and install all required dependencies listed in requirements.txt.
```

### Download the Game 🎮
For this hackathon, we’ll use "Canabalt" a classic endless runner game where players jump over obstacles by pressing the space bar. This setup provides us with clean, well-defined stimulus onset events (space bar presses) for EEG data collection.

We’ve also included an alternative game, "2048" which offers a more relaxed gameplay experience but involves more complex controls.

To get started with _Canabalt_:

1. Clone or download the open-source repository from GitHub: [Canabalt GitHub Repo](https://github.com/ninjamuffin99/canabalt-hf).
1. Follow the instructions in the repository to compile the game for Windows. \
    If you encounter issues compiling it for Windows:
    - Try compiling it for HTML5 instead.
    - You’ll need [Node.js](https://nodejs.org/en/download) installed on your system to run it in your browser.

Let us know if you encounter any issues or have questions during setup — happy hacking! 🚀
