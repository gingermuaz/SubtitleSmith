# Movie Translator Studio

A local, AI-powered desktop application that extracts audio from video files and converts it to synchronised `.srt` subtitles using the NVIDIA CUDA engine and `faster-whisper`. 

All processing runs locally on your machine—no internet connection or API keys required!

## Features
- **Hardware Accelerated:** Processes video files using your NVIDIA GPU (RTX 3070 or similar).
- **Multiple Source Languages:** Supports Bengali, Hindi, Spanish, French, Japanese, Russian, German, Polish, and Hungarian, plus Auto-Detection.
- **Multimodal Output:** Transcribe original audio or translate directly into English subtitles.
- **Live UI Progress:** Sleek Dark Mode interface with live translation progress bars and process cancellation.
- **Auto-Namer:** Automatically saves the `.srt` in the same directory as the source video.

## Requirements
- **NVIDIA GPU** (8GB+ VRAM recommended for `large-v3` models)
- **CUDA 12 Toolkit** installed on your Windows machine.
- **Python 3.12** or higher.

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/gingermuaz/MovieTranslation.git
   cd MovieTranslation

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate

3. **Install dependencies:**
   ```bash
   python -m pip install faster-whisper ctranslate2 customtkinter nvidia-cublas-cu12 nvidia-cudnn-cu12

4. **Run the Application:**
   ```bash
   python GUI.py

---

# Usage:
Click Browse to select your .mp4 video file (the save location automatically appears next to it).

Choose your Source Language and AI Model Size.

Select your desired Action (Translate to English or Transcribe in the original language).

Click Start Processing and monitor the progress.


---

### How to Save This on GitHub
1. In the GitHub text box shown in your screenshot, type `README.md` into the name field.
2. Paste the template above into the main editor window.
3. Scroll down to the **Commit changes...** button in the top right.
4. Add a short description (e.g., *"Create initial README"*).
5. Click **Commit directly to the `main` branch**.

***
