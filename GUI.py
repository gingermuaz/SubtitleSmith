import os
import shutil
import threading
import ctypes
import platform
from tkinter import filedialog, messagebox

import customtkinter as ctk
from tkinterdnd2 import TkinterDnD, DND_FILES

import main
import hardware

# --- Configuration & Constants ---
LANGUAGES = [
    "Arabic / Yemeni (ar)", "Bengali (bn)", "Chinese (zh)", "Dutch (nl)",
    "English (en)", "French (fr)", "German (de)", "Hindi (hi)",
    "Hungarian (hu)", "Indonesian (id)", "Italian (it)", "Japanese (ja)",
    "Korean (ko)", "Polish (pl)", "Portuguese (pt)", "Russian (ru)",
    "Spanish (es)", "Turkish (tr)", "Urdu / Pakistani (ur)", "Vietnamese (vi)",
    "Auto-Detect (auto)"
]

MODELS = [
    "tiny (Fastest / Low Accuracy)",
    "base (Fast / Basic Accuracy)",
    "small (Balanced)",
    "medium (Slower / High Accuracy)",
    "distil-large-v3 (6x Faster / English Only)",
    "large-v3-turbo (3x Faster / No Translation)",
    "large-v3 (Slowest / Best Accuracy)"
]

FORMATS = [".srt", ".vtt", ".txt"]
COMPUTE_OPTIONS = ["Auto (GPU if available)", "Force CPU"]

SUPPORTED_MEDIA = (
    '.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm',
    '.mp3', '.wav', '.aac', '.m4a', '.flac', '.ogg', '.wma'
)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")
cancel_event = threading.Event()

# Global variables for batch processing
selected_files = []
batch_prefix = ""

# Filled in by the hardware scan at startup; these defaults are used if the scan fails
recommended_models = {"transcribe": "large-v3-turbo", "translate": "large-v3"}
gpu_index = 0


def set_title_bar_dark(window):
    if platform.system() == "Windows":
        try:
            hwnd = ctypes.windll.user32.GetParent(window.winfo_id())
            ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 20, ctypes.byref(ctypes.c_int(1)),
                                                       ctypes.sizeof(ctypes.c_int(4)))
        except Exception:
            pass


def toggle_theme():
    mode = "light" if theme_var.get() else "dark"
    ctk.set_appearance_mode(mode)


def model_code(label):
    """'large-v3 (Slowest / Best Accuracy)' -> 'large-v3'"""
    return label.split(" ")[0].lower()


def model_label(code):
    """'large-v3' -> the full label shown in the AI Model menu"""
    return next(m for m in MODELS if model_code(m) == code)


def current_task():
    return "translate" if "Translate" in task_var.get() else "transcribe"


def on_task_change(value=None):
    """Keeps the AI model suitable when switching between Translate and Keep Original."""
    task = current_task()
    current = model_code(model_var.get())
    if current in recommended_models.values():
        # Still on an auto-pick, so move to the auto-pick for this job
        model_var.set(model_label(recommended_models[task]))
    elif task == "translate" and current in hardware.NO_TRANSLATION:
        model_var.set(model_label(recommended_models["translate"]))
        status_var.set(f"Switched AI model to '{recommended_models['translate']}': '{current}' can't translate.")


def browse_video(event=None):
    all_exts = " ".join([f"*{ext}" for ext in SUPPORTED_MEDIA])
    filepaths = filedialog.askopenfilenames(
        title="Select Media Files",
        filetypes=[("Media Files", all_exts), ("Video Files", "*.mp4 *.mkv *.avi *.mov"),
                   ("Audio Files", "*.mp3 *.wav *.aac"), ("All Files", "*.*")]
    )
    if filepaths:
        process_selected_files(filepaths)


def browse_save(event=None):
    if len(selected_files) > 1:
        directory = filedialog.askdirectory(title="Select Batch Output Folder")
        if directory:
            save_path_var.set(directory)
    else:
        ext = format_var.get()
        filepath = filedialog.asksaveasfilename(
            title="Save File As",
            defaultextension=ext,
            filetypes=[(f"{ext.upper()} File", f"*{ext}"), ("All Files", "*.*")]
        )
        if filepath: save_path_var.set(filepath)


def process_selected_files(filepaths):
    global selected_files
    selected_files = list(filepaths)
    ext = format_var.get()

    if len(selected_files) == 1:
        filepath = selected_files[0]
        video_path_var.set(filepath)
        directory = os.path.dirname(filepath)
        base_name = os.path.splitext(os.path.basename(filepath))[0]
        save_path_var.set(os.path.join(directory, f"{base_name}_subtitle{ext}"))
    elif len(selected_files) > 1:
        video_path_var.set(f"{len(selected_files)} media files selected (Batch Queue)")
        save_path_var.set("<Auto-save to original directories>")


def update_extension_in_path(*args):
    current_path = save_path_var.get()
    if current_path and not current_path.startswith("<"):
        if len(selected_files) > 1 and os.path.isdir(current_path):
            pass
        else:
            base, _ = os.path.splitext(current_path)
            save_path_var.set(f"{base}{format_var.get()}")


def drop_event(event):
    files = root.tk.splitlist(event.data)
    valid_files = [f for f in files if f.lower().endswith(SUPPORTED_MEDIA)]
    if valid_files:
        process_selected_files(valid_files)
    else:
        messagebox.showwarning("Invalid File", "Please drop supported video or audio files.")


def update_gui_progress(current, total, status_text):
    percent = current / total if total > 0 else 0
    root.after(0, lambda: progress_bar.set(percent))
    display_text = f"{batch_prefix}{status_text}" if batch_prefix else status_text
    root.after(0, lambda: status_var.set(display_text))


def check_cancel(): return cancel_event.is_set()


def cancel_process():
    cancel_event.set()
    status_var.set("Canceling... Please wait.")
    cancel_button.configure(state="disabled")


def output_path_for(video, save_val, fmt, total_files):
    """Works out where one file's subtitles go."""
    if total_files == 1:
        return save_val
    base_name = os.path.splitext(os.path.basename(video))[0]
    if os.path.isdir(save_val):
        return os.path.join(save_val, f"{base_name}_subtitle{fmt}")
    return os.path.join(os.path.dirname(video), f"{base_name}_subtitle{fmt}")


def show_summary(total, succeeded, failed, notes, cancelled, load_error):
    """Runs on the main thread once the queue finishes."""
    if load_error:
        status_var.set("Couldn't load the AI model.")
        progress_bar.set(0)
        messagebox.showerror("Couldn't load the AI model", load_error)
        return

    if cancelled:
        status_var.set(f"Canceled. {len(succeeded)} of {total} file(s) finished.")
    elif failed:
        status_var.set(f"Finished with errors: {len(succeeded)} of {total} file(s) succeeded, {len(failed)} failed.")
    else:
        status_var.set(f"Successfully processed {total} file(s).")
        progress_bar.set(1.0)

    lines = [f"{len(succeeded)} of {total} file(s) processed."]
    if failed:
        lines += ["", f"Failed ({len(failed)}):"]
        lines += [f"\u2022 {name}: {error[:300]}" for name, error in failed[:10]]
        if len(failed) > 10: lines.append(f"...and {len(failed) - 10} more.")
    if notes:
        lines += ["", "Notes:"]
        lines += [f"\u2022 {note}" for note in notes[:10]]
        if len(notes) > 10: lines.append(f"...and {len(notes) - 10} more.")

    if failed:
        messagebox.showwarning("Finished with errors", "\n".join(lines))
    elif notes or (total > 1 and not cancelled):
        messagebox.showinfo("Finished", "\n".join(lines))


def run_translation_thread(videos, lang_code, save_val, model_sz, task_type, compute, vad, fmt, hardcode, word_ts):
    global batch_prefix
    total_files = len(videos)
    device_index = 0 if compute == "cpu" else gpu_index
    succeeded, failed, notes = [], [], []
    cancelled = False
    load_error = None

    try:
        # Load the model once for the whole queue. If it won't load, every file would fail the same way.
        try:
            main.get_model(model_sz, compute, device_index, update_gui_progress)
        except Exception as e:
            load_error = str(e) or type(e).__name__
            return

        for i, video in enumerate(videos):
            if check_cancel():
                cancelled = True
                break
            name = os.path.basename(video)
            batch_prefix = f"[{i + 1}/{total_files}] " if total_files > 1 else ""
            try:
                result = main.translate_movie(
                    video_path=video, source_language=lang_code,
                    output_file=output_path_for(video, save_val, fmt, total_files),
                    model_size=model_sz, task=task_type, compute_device=compute, device_index=device_index,
                    vad_filter=vad, output_format=fmt, hardcode_subtitles=hardcode,
                    word_timestamps=word_ts, progress_callback=update_gui_progress, cancel_check=check_cancel
                )
            except main.ProcessCancelled:
                cancelled = True
                break
            except Exception as e:
                # Note the failure and carry on with the rest of the queue
                failed.append((name, str(e) or type(e).__name__))
                continue
            succeeded.append(name)
            notes += [f"{name}: {note}" for note in result["notes"]]
    finally:
        batch_prefix = ""
        root.after(0, lambda: show_summary(total_files, succeeded, failed, notes, cancelled, load_error))
        root.after(0, lambda: start_button.configure(state="normal"))
        root.after(0, lambda: cancel_button.configure(state="disabled"))


def start_process():
    if not selected_files or not save_path_var.get():
        messagebox.showwarning("Missing Information", "Please select media files and a save location.")
        return

    lang_code = lang_var.get().split("(")[-1].replace(")", "")
    model_sz = model_code(model_var.get())
    task_code = current_task()
    compute_code = "cpu" if "CPU" in compute_var.get() else "auto"
    fmt = format_var.get()
    hardcode = hardcode_var.get()

    # Catch a model that can't do this job before spending time on it
    problem = hardware.model_job_problem(model_sz, task_code, lang_code)
    if problem:
        better = recommended_models[task_code]
        if hardware.model_job_problem(better, task_code, lang_code):
            better = "large-v3"
        answer = messagebox.askyesnocancel(
            "AI model can't do this job",
            f"{problem}\n\nSwitch to '{better}' for this job?\n\n"
            f"Yes: switch and start\nNo: start with '{model_sz}' anyway\nCancel: go back")
        if answer is None:
            return
        if answer:
            model_var.set(model_label(better))
            model_sz = better

    # Check for FFmpeg now rather than after a long transcription
    if hardcode and fmt in (".srt", ".vtt") and not shutil.which("ffmpeg"):
        if not messagebox.askyesno(
                "FFmpeg not found",
                "Burn Subtitles to Video is on, but FFmpeg isn't installed or isn't on your PATH.\n\n"
                "Continue and make subtitle files only?"):
            return
        hardcode = False

    cancel_event.clear()
    start_button.configure(state="disabled")
    cancel_button.configure(state="normal")
    progress_bar.set(0)

    threading.Thread(
        target=run_translation_thread,
        args=(selected_files.copy(), lang_code, save_path_var.get(), model_sz, task_code, compute_code, vad_var.get(),
              fmt, hardcode, word_timestamps_var.get()),
        daemon=True
    ).start()


if __name__ == "__main__":
    root = TkinterDnD.Tk()
    root.title("SubtitleSmith")
    root.iconbitmap("SubtitleSmith.ico")
    root.geometry("780x580")
    root.resizable(False, False)
    root.attributes('-alpha', 1.0)
    try:
        root.tk_setPalette(background='#2b2b2b', foreground='#ffffff')
    except Exception:
        pass

    video_path_var = ctk.StringVar()
    save_path_var = ctk.StringVar()
    lang_var = ctk.StringVar(value="Auto-Detect (auto)")
    model_var = ctk.StringVar(value=MODELS[-1])
    task_var = ctk.StringVar(value="Translate to English")
    format_var = ctk.StringVar(value=".srt")
    compute_var = ctk.StringVar(value=COMPUTE_OPTIONS[0])
    vad_var = ctk.BooleanVar(value=True)
    theme_var = ctk.BooleanVar(value=False)
    hardcode_var = ctk.BooleanVar(value=False)
    word_timestamps_var = ctk.BooleanVar(value=False)  # New variable for Word-Level Timestamps
    status_var = ctk.StringVar(value="Ready")

    format_var.trace_add("write", update_extension_in_path)

    root.drop_target_register(DND_FILES)
    root.dnd_bind('<<Drop>>', drop_event)

    frame = ctk.CTkFrame(root)
    frame.pack(pady=20, padx=20, fill="both", expand=True)
    frame.grid_columnconfigure(0, weight=1)
    frame.grid_columnconfigure(3, weight=1)

    # 0. Media Selection
    ctk.CTkLabel(frame, text="Media File(s):", width=100).grid(row=0, column=0, padx=10, pady=(15, 10), sticky="e")
    video_entry = ctk.CTkEntry(frame, textvariable=video_path_var, width=400)
    video_entry.grid(row=0, column=1, pady=(15, 10))
    video_entry.bind("<Double-1>", browse_video)
    ctk.CTkButton(frame, text="Browse", width=80, command=browse_video).grid(row=0, column=2, padx=10, pady=(15, 10))

    # 1. Save Location
    ctk.CTkLabel(frame, text="Save To:", width=100).grid(row=1, column=0, padx=10, pady=5, sticky="e")
    save_entry = ctk.CTkEntry(frame, textvariable=save_path_var, width=400)
    save_entry.grid(row=1, column=1, pady=5)
    save_entry.bind("<Double-1>", browse_save)
    ctk.CTkButton(frame, text="Browse", width=80, command=browse_save).grid(row=1, column=2, padx=10, pady=5)

    # 2. Options
    options_frame = ctk.CTkFrame(frame, fg_color="transparent")
    options_frame.grid(row=2, column=0, columnspan=4, pady=10)
    ctk.CTkLabel(options_frame, text="Source Language:").pack(side="left", padx=(0, 5))
    ctk.CTkOptionMenu(options_frame, variable=lang_var, values=LANGUAGES, width=170).pack(side="left", padx=(0, 25))
    ctk.CTkLabel(options_frame, text="AI Model:").pack(side="left", padx=(0, 5))
    ctk.CTkOptionMenu(options_frame, variable=model_var, values=MODELS, width=270).pack(side="left")

    # 3. Action Toggle
    action_frame = ctk.CTkFrame(frame, fg_color="transparent")
    action_frame.grid(row=3, column=0, columnspan=4, pady=10)
    ctk.CTkLabel(action_frame, text="Action:").pack(side="left", padx=10)
    ctk.CTkSegmentedButton(action_frame, variable=task_var, values=["Translate to English", "Keep Original Language"],
                           command=on_task_change, width=320).pack(side="left", padx=10)

    # 4. Advanced Settings Row 1
    adv_frame = ctk.CTkFrame(frame, fg_color="transparent")
    adv_frame.grid(row=4, column=0, columnspan=4, pady=(5, 0))
    ctk.CTkLabel(adv_frame, text="Format:").pack(side="left", padx=(0, 5))
    ctk.CTkOptionMenu(adv_frame, variable=format_var, values=FORMATS, width=70).pack(side="left", padx=(0, 15))
    ctk.CTkLabel(adv_frame, text="Compute:").pack(side="left", padx=(0, 5))
    ctk.CTkOptionMenu(adv_frame, variable=compute_var, values=COMPUTE_OPTIONS, width=150).pack(side="left",
                                                                                               padx=(0, 15))
    ctk.CTkSwitch(adv_frame, text="VAD Filter", variable=vad_var).pack(side="left", padx=(0, 15))

    # 5. Advanced Settings Row 2 (Added Word Timestamps)
    adv2_frame = ctk.CTkFrame(frame, fg_color="transparent")
    adv2_frame.grid(row=5, column=0, columnspan=4, pady=(5, 5))
    ctk.CTkSwitch(adv2_frame, text="Burn Subtitles to Video", variable=hardcode_var).pack(side="left", padx=(0, 20))
    ctk.CTkSwitch(adv2_frame, text="Word Timestamps", variable=word_timestamps_var).pack(side="left", padx=(0, 20))
    ctk.CTkSwitch(adv2_frame, text="Light UI", variable=theme_var, command=toggle_theme).pack(side="left")

    # 6. Status
    status_frame = ctk.CTkFrame(frame, fg_color="transparent")
    status_frame.grid(row=6, column=0, columnspan=4, pady=(15, 0))
    ctk.CTkLabel(status_frame, textvariable=status_var, text_color="gray").pack(pady=(0, 5))
    progress_bar = ctk.CTkProgressBar(status_frame, width=460)
    progress_bar.set(0)
    progress_bar.pack()

    # 7. Buttons
    button_frame = ctk.CTkFrame(frame, fg_color="transparent")
    button_frame.grid(row=7, column=0, columnspan=4, pady=(15, 10))
    start_button = ctk.CTkButton(button_frame, text="Start Processing", command=start_process, fg_color="#28a745",
                                 hover_color="#218838")
    start_button.pack(side="left", padx=15)
    cancel_button = ctk.CTkButton(button_frame, text="Cancel", command=cancel_process, fg_color="#dc3545",
                                  hover_color="#c82333", state="disabled")
    cancel_button.pack(side="left", padx=15)

    # 8. Hardware Info
    hw_frame = ctk.CTkFrame(frame, fg_color="transparent")
    hw_frame.grid(row=8, column=0, columnspan=4, pady=(0, 10))
    hw_textbox = ctk.CTkTextbox(hw_frame, width=600, height=130, text_color="#28a745", fg_color="#1e1e1e")
    hw_textbox.pack()

    try:
        scan = hardware.scan_system()
        hw_textbox.insert("0.0", scan["report"])
        recommended_models = scan["recommended"]
        gpu_index = scan["gpu_index"]
        model_var.set(model_label(recommended_models[current_task()]))
    except Exception as e:
        hw_textbox.insert("0.0", f"Could not perform hardware scan.\nError: {e}")
    hw_textbox.configure(state="disabled")

    set_title_bar_dark(root)
    root.mainloop()