import customtkinter as ctk
from tkinter import filedialog, messagebox
import threading
import os
import main

# --- UI Theme Setup ---
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

cancel_event = threading.Event()


def browse_video():
    filepath = filedialog.askopenfilename(
        title="Select Video File",
        filetypes=[("MP4 Videos", "*.mp4"), ("All Files", "*.*")]
    )
    if filepath:
        video_path_var.set(filepath)
        directory = os.path.dirname(filepath)
        base_name = os.path.splitext(os.path.basename(filepath))[0]
        auto_save_path = os.path.join(directory, f"{base_name}_subtitle.srt")
        save_path_var.set(auto_save_path)


def browse_save():
    filepath = filedialog.asksaveasfilename(
        title="Save Subtitle File As",
        defaultextension=".srt",
        filetypes=[("SRT Subtitles", "*.srt"), ("All Files", "*.*")]
    )
    if filepath:
        save_path_var.set(filepath)


def update_gui_progress(current, total, status_text):
    percent = current / total if total > 0 else 0
    root.after(0, lambda: progress_bar.set(percent))
    root.after(0, lambda: status_var.set(status_text))


def check_cancel():
    return cancel_event.is_set()


def cancel_process():
    cancel_event.set()
    status_var.set("Canceling... Please wait.")
    cancel_button.configure(state="disabled")


def run_translation_thread(video, lang_code, save, model_sz, task_type):
    try:
        main.translate_movie(
            video_path=video,
            source_language=lang_code,
            output_file=save,
            model_size=model_sz,
            task=task_type,
            progress_callback=update_gui_progress,
            cancel_check=check_cancel
        )
    except Exception as e:
        update_gui_progress(0, 100, f"Error: {str(e)}")
    finally:
        root.after(0, lambda: start_button.configure(state="normal"))
        root.after(0, lambda: cancel_button.configure(state="disabled"))


def start_process():
    video = video_path_var.get()
    save = save_path_var.get()
    lang = lang_var.get()
    model = model_var.get()
    task = task_var.get()

    if not video or not save:
        messagebox.showwarning("Missing Information", "Please select both a video file and a save location.")
        return

    lang_code = lang.split("(")[-1].replace(")", "")
    model_code = model.split(" ")[0].lower()
    task_code = "translate" if "Translate" in task else "transcribe"

    cancel_event.clear()
    start_button.configure(state="disabled")
    cancel_button.configure(state="normal")
    progress_bar.set(0)

    threading.Thread(target=run_translation_thread, args=(video, lang_code, save, model_code, task_code),
                     daemon=True).start()


# --- GUI Window Setup ---
root = ctk.CTk()
root.title("Movie Translator Studio")
root.geometry("780x480")
root.resizable(False, False)

video_path_var = ctk.StringVar()
save_path_var = ctk.StringVar()
lang_var = ctk.StringVar(value="Bengali (bn)")
model_var = ctk.StringVar(value="large-v3 (Slowest / Best Accuracy)")
task_var = ctk.StringVar(value="Translate to English")
status_var = ctk.StringVar(value="Ready")

frame = ctk.CTkFrame(root)
frame.pack(pady=20, padx=20, fill="both", expand=True)

# Distribute weight to column 0 and 3 to act as side springs
frame.grid_columnconfigure(0, weight=1)
frame.grid_columnconfigure(3, weight=1)

# 1. Video Selection Row
ctk.CTkLabel(frame, text="Video File:", width=100).grid(row=0, column=0, padx=10, pady=(20, 10), sticky="e")
ctk.CTkEntry(frame, textvariable=video_path_var, width=400).grid(row=0, column=1, pady=(20, 10))
ctk.CTkButton(frame, text="Browse", width=80, command=browse_video).grid(row=0, column=2, padx=10, pady=(20, 10))

# 2. Save Location Row
ctk.CTkLabel(frame, text="Save .srt To:", width=100).grid(row=1, column=0, padx=10, pady=10, sticky="e")
ctk.CTkEntry(frame, textvariable=save_path_var, width=400).grid(row=1, column=1, pady=10)
ctk.CTkButton(frame, text="Browse", width=80, command=browse_save).grid(row=1, column=2, padx=10, pady=10)

# 3. Language & Model Options (Centered Frame)
options_frame = ctk.CTkFrame(frame, fg_color="transparent")
options_frame.grid(row=2, column=0, columnspan=3, pady=10)

ctk.CTkLabel(options_frame, text="Source Language:").pack(side="left", padx=(0, 5))
languages = [
    "Bengali (bn)", "Hindi (hi)", "Spanish (es)", "French (fr)",
    "Japanese (ja)", "Russian (ru)", "German (de)", "Polish (pl)",
    "Hungarian (hu)", "Auto-Detect (auto)"
]
ctk.CTkOptionMenu(options_frame, variable=lang_var, values=languages, width=170).pack(side="left", padx=(0, 25))

ctk.CTkLabel(options_frame, text="AI Model:").pack(side="left", padx=(0, 5))
models = [
    "tiny (Fastest / Low Accuracy)",
    "base (Fast / Basic Accuracy)",
    "small (Balanced)",
    "medium (Slower / High Accuracy)",
    "large-v3 (Slowest / Best Accuracy)"
]
ctk.CTkOptionMenu(options_frame, variable=model_var, values=models, width=240).pack(side="left")

# 4. Action / Task Toggle Row (Centered Frame)
action_frame = ctk.CTkFrame(frame, fg_color="transparent")
action_frame.grid(row=3, column=0, columnspan=3, pady=10)

ctk.CTkLabel(action_frame, text="Action:").pack(side="left", padx=10)
ctk.CTkSegmentedButton(action_frame, variable=task_var, values=["Translate to English", "Keep Original Language"],
                       width=320).pack(side="left", padx=10)

# 5. Status text & Progress Bar (Centered Frame)
status_frame = ctk.CTkFrame(frame, fg_color="transparent")
status_frame.grid(row=4, column=0, columnspan=3, pady=(15, 0))

ctk.CTkLabel(status_frame, textvariable=status_var, text_color="gray").pack(pady=(0, 5))
progress_bar = ctk.CTkProgressBar(status_frame, width=460)
progress_bar.set(0)
progress_bar.pack()

# 6. Action Buttons (Centered Frame)
button_frame = ctk.CTkFrame(frame, fg_color="transparent")
button_frame.grid(row=5, column=0, columnspan=3, pady=(25, 20))

start_button = ctk.CTkButton(button_frame, text="Start Processing", command=start_process, fg_color="#28a745",
                             hover_color="#218838")
start_button.pack(side="left", padx=15)

cancel_button = ctk.CTkButton(button_frame, text="Cancel", command=cancel_process, fg_color="#dc3545",
                              hover_color="#c82333", state="disabled")
cancel_button.pack(side="left", padx=15)

root.mainloop()