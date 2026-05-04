import os
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
    "tiny (Fastest / Low Accuracy)", "base (Fast / Basic Accuracy)",
    "small (Balanced)", "medium (Slower / High Accuracy)",
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


def set_title_bar_dark(window):
    if platform.system() == "Windows":
        try:
            hwnd = ctypes.windll.user32.GetParent(window.winfo_id())
            ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 20, ctypes.byref(ctypes.c_int(1)),
                                                       ctypes.sizeof(ctypes.c_int(4)))
        except Exception:
            pass


def browse_video(event=None):
    all_exts = " ".join([f"*{ext}" for ext in SUPPORTED_MEDIA])
    filepath = filedialog.askopenfilename(
        title="Select Media File",
        filetypes=[("Media Files", all_exts), ("Video Files", "*.mp4 *.mkv *.avi *.mov"),
                   ("Audio Files", "*.mp3 *.wav *.aac"), ("All Files", "*.*")]
    )
    if filepath: process_selected_file(filepath)


def browse_save(event=None):
    ext = format_var.get()
    filepath = filedialog.asksaveasfilename(
        title="Save File As",
        defaultextension=ext,
        filetypes=[(f"{ext.upper()} File", f"*{ext}"), ("All Files", "*.*")]
    )
    if filepath: save_path_var.set(filepath)


def process_selected_file(filepath):
    video_path_var.set(filepath)
    directory = os.path.dirname(filepath)
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    ext = format_var.get()
    save_path_var.set(os.path.join(directory, f"{base_name}_subtitle{ext}"))


def update_extension_in_path(*args):
    """Updates the save path extension instantly when the user changes the dropdown."""
    current_path = save_path_var.get()
    if current_path:
        base, _ = os.path.splitext(current_path)
        save_path_var.set(f"{base}{format_var.get()}")


def drop_event(event):
    filepath = event.data.strip('{}')
    if filepath.lower().endswith(SUPPORTED_MEDIA):
        process_selected_file(filepath)
    else:
        messagebox.showwarning("Invalid File", "Please drop a supported video or audio file.")


def update_gui_progress(current, total, status_text):
    percent = current / total if total > 0 else 0
    root.after(0, lambda: progress_bar.set(percent))
    root.after(0, lambda: status_var.set(status_text))


def check_cancel(): return cancel_event.is_set()


def cancel_process():
    cancel_event.set()
    status_var.set("Canceling... Please wait.")
    cancel_button.configure(state="disabled")


def run_translation_thread(video, lang_code, save, model_sz, task_type, compute, vad, fmt):
    try:
        main.translate_movie(
            video_path=video, source_language=lang_code, output_file=save,
            model_size=model_sz, task=task_type, compute_device=compute,
            vad_filter=vad, output_format=fmt,
            progress_callback=update_gui_progress, cancel_check=check_cancel
        )
    except Exception as e:
        update_gui_progress(0, 100, f"Error: {str(e)}")
    finally:
        root.after(0, lambda: start_button.configure(state="normal"))
        root.after(0, lambda: cancel_button.configure(state="disabled"))


def start_process():
    video = video_path_var.get()
    save = save_path_var.get()
    if not video or not save:
        messagebox.showwarning("Missing Information", "Please select a media file and save location.")
        return

    lang_code = lang_var.get().split("(")[-1].replace(")", "")
    model_code = model_var.get().split(" ")[0].lower()
    task_code = "translate" if "Translate" in task_var.get() else "transcribe"
    compute_code = "cpu" if "CPU" in compute_var.get() else "auto"

    cancel_event.clear()
    start_button.configure(state="disabled")
    cancel_button.configure(state="normal")
    progress_bar.set(0)

    threading.Thread(
        target=run_translation_thread,
        args=(video, lang_code, save, model_code, task_code, compute_code, vad_var.get(), format_var.get()),
        daemon=True
    ).start()


if __name__ == "__main__":
    root = TkinterDnD.Tk()
    root.title("SubtitleSmith")
    root.geometry("780x680")
    root.resizable(False, False)
    root.attributes('-alpha', 1.0)
    try:
        root.tk_setPalette(background='#2b2b2b', foreground='#ffffff')
    except Exception:
        pass

    video_path_var = ctk.StringVar()
    save_path_var = ctk.StringVar()
    lang_var = ctk.StringVar(value=LANGUAGES[-2])
    model_var = ctk.StringVar(value=MODELS[-1])
    task_var = ctk.StringVar(value="Translate to English")
    format_var = ctk.StringVar(value=".srt")
    compute_var = ctk.StringVar(value=COMPUTE_OPTIONS[0])
    vad_var = ctk.BooleanVar(value=True)  # VAD on by default
    status_var = ctk.StringVar(value="Ready")

    format_var.trace_add("write", update_extension_in_path)

    root.drop_target_register(DND_FILES)
    root.dnd_bind('<<Drop>>', drop_event)

    frame = ctk.CTkFrame(root)
    frame.pack(pady=20, padx=20, fill="both", expand=True)
    frame.grid_columnconfigure(0, weight=1)
    frame.grid_columnconfigure(3, weight=1)

    # 0. Media Selection
    ctk.CTkLabel(frame, text="Media File:", width=100).grid(row=0, column=0, padx=10, pady=(15, 10), sticky="e")
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
    ctk.CTkOptionMenu(options_frame, variable=model_var, values=MODELS, width=240).pack(side="left")

    # 3. Action Toggle
    action_frame = ctk.CTkFrame(frame, fg_color="transparent")
    action_frame.grid(row=3, column=0, columnspan=4, pady=10)
    ctk.CTkLabel(action_frame, text="Action:").pack(side="left", padx=10)
    ctk.CTkSegmentedButton(action_frame, variable=task_var, values=["Translate to English", "Keep Original Language"],
                           width=320).pack(side="left", padx=10)

    # 4. Advanced Settings Row
    adv_frame = ctk.CTkFrame(frame, fg_color="transparent")
    adv_frame.grid(row=4, column=0, columnspan=4, pady=5)
    ctk.CTkLabel(adv_frame, text="Format:").pack(side="left", padx=(0, 5))
    ctk.CTkOptionMenu(adv_frame, variable=format_var, values=FORMATS, width=70).pack(side="left", padx=(0, 15))
    ctk.CTkLabel(adv_frame, text="Compute:").pack(side="left", padx=(0, 5))
    ctk.CTkOptionMenu(adv_frame, variable=compute_var, values=COMPUTE_OPTIONS, width=160).pack(side="left",
                                                                                               padx=(0, 15))
    ctk.CTkSwitch(adv_frame, text="VAD Filter (Ignore Silence)", variable=vad_var).pack(side="left")

    # 5. Status
    status_frame = ctk.CTkFrame(frame, fg_color="transparent")
    status_frame.grid(row=5, column=0, columnspan=4, pady=(15, 0))
    ctk.CTkLabel(status_frame, textvariable=status_var, text_color="gray").pack(pady=(0, 5))
    progress_bar = ctk.CTkProgressBar(status_frame, width=460)
    progress_bar.set(0)
    progress_bar.pack()

    # 6. Buttons
    button_frame = ctk.CTkFrame(frame, fg_color="transparent")
    button_frame.grid(row=6, column=0, columnspan=4, pady=(15, 10))
    start_button = ctk.CTkButton(button_frame, text="Start Processing", command=start_process, fg_color="#28a745",
                                 hover_color="#218838")
    start_button.pack(side="left", padx=15)
    cancel_button = ctk.CTkButton(button_frame, text="Cancel", command=cancel_process, fg_color="#dc3545",
                                  hover_color="#c82333", state="disabled")
    cancel_button.pack(side="left", padx=15)

    # 7. Hardware Info
    hw_frame = ctk.CTkFrame(frame, fg_color="transparent")
    hw_frame.grid(row=7, column=0, columnspan=4, pady=(0, 10))
    hw_textbox = ctk.CTkTextbox(hw_frame, width=600, height=130, text_color="#28a745", fg_color="#1e1e1e")
    hw_textbox.pack()

    try:
        report, recommended_model = hardware.scan_system()
        hw_textbox.insert("0.0", report)
        model_var.set(recommended_model)
    except Exception as e:
        hw_textbox.insert("0.0", f"Could not perform hardware scan.\nError: {e}")
    hw_textbox.configure(state="disabled")

    set_title_bar_dark(root)
    root.mainloop()