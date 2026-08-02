# SubtitleSmith

SubtitleSmith is a local, AI-powered desktop application that extracts audio from video files and converts it to synchronised `.srt` subtitles using the NVIDIA CUDA engine and `faster-whisper`. 

Featuring drag-and-drop file support and an intelligent hardware scanner that automatically recommends the best AI model for your system's VRAM. All processing runs locally on your machine—no internet connection or API keys required!

## Features

* **Hardware Auto-Profiling:** A built-in scanner analyses your CPU, RAM (speed/slots), and GPU VRAM to automatically recommend the optimal `faster-whisper` AI model for your specific machine.
* **Universal Media & Batch Processing:** Drag-and-drop a single file or a massive batch of Video (MP4, MKV, AVI, MOV) or Audio (MP3, WAV, FLAC) files directly into the UI for automated queue processing.
* **Expanded Language Support:** Transcribe or translate 20 major global languages—including Arabic, Chinese, Japanese, Spanish, and Urdu—or just use Auto-Detect.
* **Flexible Export Options:** Export your text as standard subtitles (`.srt`), web subtitles (`.vtt`), raw transcripts (`.txt`), or **permanently burn (hardcode) the subtitles directly into a new video file!**
* **Sleek & Interactive UI:** Light and Dark mode interface featuring drag-and-drop targets, double-click file browsing, live batch progress bars, and safe process cancellation.
* **Smart Auto-Namer:** Automatically generates and saves your output files in the exact same directory as your source media.

## Requirements

* **OS:** Windows 10/11 or Linux.
* **GPU:** NVIDIA GPU (8GB+ VRAM highly recommended for `large-v3` models).
* **Toolkit:** CUDA 12 Toolkit installed on your machine.
* **Python:** Python 3.12 or higher.
* **FFmpeg:** Must be installed and added to your System PATH (Required for audio extraction & video hardcoding).

## Setup & Installation

1. **Install FFmpeg (Required for audio extraction & video hardcoding):**
   ```bash
   winget install ffmpeg

2. **Clone the repository:**
   ```bash
   git clone https://github.com/gingermuaz/SubtitleSmith.git
   cd SubtitleSmith

3. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate

4. **Install dependencies:**
   ```bash
   python -m pip install faster-whisper ctranslate2 customtkinter nvidia-cublas-cu12 nvidia-cudnn-cu12 psutil tkinterdnd2 tqdm

5. **Run the Application:**
   ```bash
   python GUI.py

## Usage

1. **Import Media:** Drag and drop one or multiple video/audio files directly into the application window, or click **Browse** to select them. (Batch queue processing is fully supported!)
2. **Configure AI Settings:** Choose your **Source Language** (or use Auto-Detect) and select an **AI Model**. The built-in hardware scanner will automatically recommend the best model for your PC.
3. **Choose Action:** Select whether to **Translate to English** or **Keep Original Language** (Transcribe).
4. **Tweak Advanced Settings (Optional):**
    * **Format:** Choose between `.srt`, `.vtt`, or plain `.txt` outputs.
    * **Compute:** Force the app to use your CPU if you are troubleshooting GPU issues.
    * **VAD Filter:** Leave this enabled to automatically ignore silent parts of the audio, preventing AI hallucinations.
    * **Burn Subtitles to Video:** Enable this to permanently hardcode the generated text into a brand new video file.
5. **Process:** Click **Start Processing**. SubtitleSmith will process your queue and automatically save the outputs in the exact same folder as your original media!
---
<img width="1222" height="957" alt="image" src="https://github.com/user-attachments/assets/923db541-22ff-41c7-a457-447769c23c5c" />

