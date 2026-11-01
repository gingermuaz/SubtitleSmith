# SubtitleSmith

SubtitleSmith is a local, AI-powered desktop application that extracts audio from video files and converts it to synchronised `.srt` subtitles using the NVIDIA CUDA engine and `faster-whisper`.

Featuring drag-and-drop file support and an intelligent hardware scanner that automatically recommends the best AI model for your system's VRAM. All processing runs locally on your machine—no API keys required, and no internet connection once each AI model has downloaded on first use!

## Features

* **Hardware Auto-Profiling:** A built-in scanner analyses your CPU, RAM (speed/slots), and GPU VRAM to automatically recommend the optimal `faster-whisper` AI model for your specific machine, for both translating and keeping the original language.
* **Universal Media & Batch Processing:** Drag-and-drop a single file or a massive batch of Video (MP4, MKV, AVI, MOV) or Audio (MP3, WAV, FLAC) files directly into the UI for automated queue processing.
* **Expanded Language Support:** Transcribe or translate 20 major global languages—including Arabic, Chinese, Japanese, Spanish, and Urdu—or just use Auto-Detect.
* **Subtitles in Any Language:** Keep the original language and add English in the same run, and/or translate into another language (French, Bengali, Japanese...) with Meta's NLLB-200 translation model, running locally.
* **Speaker Labels (optional):** Mark who's talking with "Speaker 1:", "Speaker 2:"... using pyannote.audio.
* **Review Before Saving:** Check and fix the subtitles in a built-in editor (with Replace All) before they're saved or burned in.
* **Readable Subtitles:** Long passages are split into proper subtitles—at most two lines, each on screen for 1–7 seconds, broken at sentence ends and pauses—with shorter lines for Chinese, Japanese and Korean.
* **Fast Mode:** On an NVIDIA GPU, SubtitleSmith transcribes several pieces of audio at once, which is often several times faster on long films.
* **Flexible Export Options:** Export your text as standard subtitles (`.srt`), web subtitles (`.vtt`) or raw transcripts (`.txt`). Add them to a copy of your video as a **subtitle track viewers can switch on and off** (takes seconds), or **permanently burn (hardcode) them into the picture** in the size, colour and position you choose, using your NVIDIA GPU's video encoder!
* **Sleek & Interactive UI:** Light and Dark mode interface featuring drag-and-drop targets, double-click file browsing, a queue list showing each file's status, an estimate of the time left, and a Cancel button that stops straight away—even mid-download or mid-encode. Your choices are remembered for next time.
* **Smart Auto-Namer:** Automatically generates and saves your output files in the exact same directory as your source media.
* **Command Line:** `cli.py` does everything the window does, for scripts and whole folders.

## Requirements

* **OS:** Windows 10/11. Linux is untested: the app may run from source, but its GPU setup code is Windows-only.
* **GPU:** NVIDIA GPU with an up-to-date driver (8GB+ VRAM highly recommended for `large-v3` models). You don't need the CUDA Toolkit: the CUDA libraries (cuBLAS and cuDNN) are installed by pip with the other dependencies. No NVIDIA GPU? Choose **Force CPU** (much slower).
* **Python:** Python 3.12 or higher.
* **FFmpeg (optional):** Only needed for the **Video** options (adding a subtitle track or burning subtitles in). Reading audio from your media files is built in.
* **Internet (first use only):** Each AI model downloads from Hugging Face the first time you use it, from about 75 MB (`tiny`) to about 3 GB (`large-v3`). The status bar shows the download's progress, and an interrupted download carries on from where it stopped next time.

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/gingermuaz/SubtitleSmith.git
   cd SubtitleSmith
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   ```

3. **Install dependencies** (pinned versions, including the CUDA libraries):
   ```bash
   python -m pip install -r requirements.txt
   ```

4. **Install FFmpeg (optional, only for adding subtitles to videos)** and make sure it's on your System PATH:
   ```bash
   winget install ffmpeg
   ```

5. **Run the Application:**
   ```bash
   python GUI.py
   ```

## Usage

1. **Import Media:** Drag and drop one or multiple video/audio files directly into the application window, or click **Browse** to select them. (Batch queue processing is fully supported!)
2. **Configure AI Settings:** Choose your **Source Language** (or use Auto-Detect) and select an **AI Model**. The built-in hardware scanner will automatically recommend the best model for your PC. With Auto-Detect, SubtitleSmith tells you which language it heard (e.g. "Detected language: Japanese (97%)").
    * **Names & Terms (optional):** List character names or special words (e.g. `Frodo, Gandalf, Mordor`) so the AI spells them correctly.
3. **Choose Action:** **Translate to English**, **Keep Original Language** (transcribe), or **Original + English** (both: `film_subtitle.srt` in the original language and `film_subtitle.en.srt` in English).
    * **Also translate to:** Adds a translation into another language too (e.g. `film_subtitle.bn.srt`). Translations use Meta's NLLB-200 model, downloaded the first time. Choose its size with **Translation Model** in the **System** tab: *Fast* (600 MB), *Better* (1.4 GB) or *Best* (3.4 GB, for GPUs with 10 GB+ VRAM). Bigger translates better but takes longer; the hardware scan picks the one that suits your PC. Its licence allows **non-commercial use only**.
    * **Speaker Labels:** Starts a subtitle with "Speaker 1:", "Speaker 2:"... whenever the speaker changes. Needs the optional extra packages and a one-time setup (see [Speaker Labels](#speaker-labels)).
    * **Review Before Saving:** Opens each file's subtitles in an editor before they're saved, so you can fix words (Replace All fixes a misheard name everywhere at once). The queue waits for you.
4. **Tweak Advanced Settings (Optional):**
    * **Format:** Choose between `.srt`, `.vtt`, or plain `.txt` outputs.
    * **Compute:** Force the app to use your CPU if you are troubleshooting GPU issues.
    * **VAD Filter:** Leave this enabled to automatically ignore silent parts of the audio, preventing AI hallucinations. Fast Mode needs it.
    * **Fast Mode:** On by default. With an NVIDIA GPU, transcribes several pieces of audio at once, which is much faster on long files (if the GPU runs out of memory, it automatically uses smaller batches). Turn it off if a film comes out with more mistakes than usual: normal mode re-checks unclear passages.
    * **Video:** *Subtitles only* saves just the subtitle file. *Add subtitle track* also saves a copy of your video (`_subtitled`) with the subtitles as a track viewers can switch on and off. Nothing is re-encoded, so it takes seconds (AVI videos become MKV). With translations, every language becomes its own track. *Burn into video* saves a copy (`_hardcoded.mp4`) with the subtitles drawn permanently into the picture, using your GPU's NVENC encoder when FFmpeg can (much faster than the CPU); **Style…** sets the size, colour and position. Both need FFmpeg.
    * **Readable Lines:** On by default. Splits long passages into subtitles of up to two lines that stay on screen for 1–7 seconds. Turn it off to get one subtitle per AI segment, however long.
5. **Process:** Click **Start Processing**. SubtitleSmith will process your queue and automatically save the outputs in the exact same folder as your original media! The **Queue** tab shows each file as waiting, in progress, done or failed, and the status bar estimates the time left. **Cancel** stops straight away. Your settings are remembered for next time (in `%APPDATA%\SubtitleSmith\settings.json`).

---
<img width="1222" height="957" alt="image" src="https://github.com/user-attachments/assets/923db541-22ff-41c7-a457-447769c23c5c" />

## Command Line

`cli.py` does everything the window does, for scripts and whole folders:

```bash
python cli.py film.mkv
python cli.py "D:\Films" --recursive --task transcribe --also-english --video track
python cli.py talk.mp4 --language en --translate-to bn --translator best --format vtt --output-dir subs
```

Run `python cli.py --help` for every option. It exits with 0 if every file worked and 1 if any failed, so scripts can check.

## Speaker Labels

Speaker labels are optional because they need PyTorch and pyannote.audio (a few GB):

1. For GPU speed, install PyTorch with CUDA support using the command from [pytorch.org](https://pytorch.org/get-started/locally/) (otherwise you get the slower CPU version).
2. Install the rest: `python -m pip install -r requirements-speakers.txt`
3. With a free Hugging Face account, accept the terms of [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) and create a "Read" [access token](https://huggingface.co/settings/tokens). The first time you switch on Speaker Labels, SubtitleSmith asks for the token (from the command line, run `hf auth login` instead).

pyannote's usage reporting is switched off, so nothing leaves your computer apart from the one-time model download.

## Building the Windows App

```bash
python -m pip install pyinstaller==6.20.0
pyinstaller SubtitleSmith.spec --noconfirm --clean
```

The app is created in `dist\SubtitleSmith\` (run `SubtitleSmith.exe`). The folder is about 2 GB because it includes the CUDA libraries, so the people you give it to only need an NVIDIA driver (plus FFmpeg for the Video options).

## Project Layout

| File | What it does |
| --- | --- |
| `GUI.py` | The window (the `App` class). Run this to start the app. |
| `engine.py` | Loads AI models and turns one media file into subtitles (`process_file`). |
| `writers.py` | Formats the results as `.srt`, `.vtt` or `.txt`. |
| `translation.py` | Translates subtitles with NLLB-200 (CTranslate2). |
| `speakers.py` | Speaker labels with pyannote.audio (optional). |
| `dialogs.py` | The review, burn-in style and token windows. |
| `cli.py` | The command-line version. |
| `cancellable.py` | Lets Cancel stop waiting for slow steps. |
| `video.py` | Adds subtitles to a copy of a video with FFmpeg: as a switchable track, or burned in. |
| `languages.py` | Language names, subtitle track labels and line lengths per language. |
| `progress.py` | The queue list and the time-left estimate. |
| `settings.py` | Remembers your choices between launches. |
| `hardware.py` | The hardware scan and model recommendations. |
| `errors.py` | Error types shared by the modules above. |
| `tests/` | Tests that run without a GPU or a model download. |

## Running Tests

```bash
python -m unittest
```

The video tests are skipped if FFmpeg isn't installed.

## License

SubtitleSmith is released under the [MIT License](LICENSE). The AI models it downloads have their own licences: Whisper (MIT), NLLB-200 (CC-BY-NC 4.0, **non-commercial use only**) and pyannote's speaker model (see its Hugging Face page).
