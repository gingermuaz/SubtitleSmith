import os
import sys
import subprocess
import shutil
from faster_whisper import WhisperModel
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# --- AGGRESSIVE WINDOWS DLL INJECTION ---
venv_dir = os.path.dirname(os.path.dirname(sys.executable))
site_packages_dir = os.path.join(venv_dir, "Lib", "site-packages")
cublas_bin = os.path.join(site_packages_dir, "nvidia", "cublas", "bin")
cudnn_bin = os.path.join(site_packages_dir, "nvidia", "cudnn", "bin")

os.environ["PATH"] = f"{cublas_bin};{cudnn_bin};" + os.environ.get("PATH", "")
if hasattr(os, 'add_dll_directory'):
    if os.path.exists(cublas_bin): os.add_dll_directory(cublas_bin)
    if os.path.exists(cudnn_bin): os.add_dll_directory(cudnn_bin)


# ----------------------------------------

def format_timestamp(seconds: float, is_vtt=False):
    """Formats time to SRT (00:00:00,000) or VTT (00:00:00.000) standards."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    separator = "." if is_vtt else ","
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def translate_movie(video_path, source_language, output_file, model_size="large-v3", task="translate",
                    compute_device="auto", vad_filter=True, output_format=".srt", hardcode_subtitles=False,
                    progress_callback=None, cancel_check=None):
    if progress_callback: progress_callback(0, 100, f"Loading '{model_size}' model onto {compute_device.upper()}...")

    c_type = "int8" if compute_device == "cpu" else "int8_float16"
    model = WhisperModel(model_size, device=compute_device, compute_type=c_type)

    if progress_callback: progress_callback(0, 100, "Extracting audio and analyzing...")

    lang_param = None if source_language == "auto" else source_language
    segments, info = model.transcribe(video_path, task=task, language=lang_param, vad_filter=vad_filter)
    total_duration = round(info.duration, 2)

    content = "WEBVTT\n\n" if output_format == ".vtt" else ""
    is_vtt = (output_format == ".vtt")

    with tqdm(total=total_duration, unit="sec",
              bar_format="{l_bar}{bar}| {n:.1f}/{total:.1f} [{elapsed}<{remaining}]") as pbar:
        for index, segment in enumerate(segments, start=1):
            if cancel_check and cancel_check():
                if progress_callback: progress_callback(0, 100, "Process Canceled.")
                print("\nProcess aborted by user.")
                return

            if output_format in [".srt", ".vtt"]:
                start_time = format_timestamp(segment.start, is_vtt)
                end_time = format_timestamp(segment.end, is_vtt)
                content += f"{index}\n"
                content += f"{start_time} --> {end_time}\n"
                content += f"{segment.text.strip()}\n\n"
            elif output_format == ".txt":
                content += f"{segment.text.strip()}\n"

            pbar.update(segment.end - pbar.n)

            if progress_callback:
                percentage = round((segment.end / total_duration) * 100, 1)
                action_text = "Translating" if task == "translate" else "Transcribing"
                progress_callback(segment.end, total_duration, f"{action_text}... {percentage}%")

    if progress_callback: progress_callback(100, 100, "Saving text file...")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(content)

    # --- PHASE 3: HARDCODE SUBTITLES ---
    if hardcode_subtitles and output_format in [".srt", ".vtt"] and video_path.lower().endswith(
            ('.mp4', '.mkv', '.avi', '.mov')):
        if not shutil.which("ffmpeg"):
            if progress_callback: progress_callback(100, 100, "Error: FFmpeg not found on system.")
            return

        if progress_callback: progress_callback(100, 100, "Burning subtitles to video... (This takes a while)")

        base_name = os.path.splitext(video_path)[0]
        output_video = f"{base_name}_hardcoded.mp4"

        # Windows path escaping for FFmpeg filters is notoriously tricky.
        # We temporarily change the working directory to make the paths relative.
        original_cwd = os.getcwd()
        target_dir = os.path.dirname(os.path.abspath(output_file))
        sub_filename = os.path.basename(output_file)
        vid_filename = os.path.basename(video_path)
        out_vid_filename = os.path.basename(output_video)

        try:
            os.chdir(target_dir)

            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-i", vid_filename,
                "-vf", f"subtitles='{sub_filename}'",
                "-c:a", "copy",  # Copy the audio exactly to save time and preserve quality
                out_vid_filename
            ]

            creationflags = 0x08000000 if os.name == 'nt' else 0
            process = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     creationflags=creationflags)

            if process.returncode == 0:
                if progress_callback: progress_callback(100, 100, "Success! Subtitles hardcoded and saved.")
            else:
                if progress_callback: progress_callback(100, 100, "Error during video encoding. Check console.")
                print(process.stderr.decode('utf-8'))
        except Exception as e:
            if progress_callback: progress_callback(100, 100, f"Encoding error: {e}")
        finally:
            os.chdir(original_cwd)
    else:
        if progress_callback: progress_callback(100, 100, "Success! File saved.")


if __name__ == "__main__":
    print("Please run GUI.py to start the application.")