import os
import sys
import threading
import time
import ctypes
import platform
from tkinter import filedialog, messagebox

# The packaged .exe has no console (console=False), so sys.stdout/sys.stderr are None and any
# library that prints would crash, e.g. the tqdm progress bars in Hugging Face model downloads.
# Send that output nowhere instead.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")

import customtkinter as ctk
from tkinterdnd2 import TkinterDnD, DND_FILES

import engine
import hardware
import languages
import progress
import settings
import video

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

AUTO_SAVE = "<Auto-save to original directories>"

# Video menu label -> engine.process_file(video_output=...)
VIDEO_OUTPUTS = {"Subtitles only": "none", "Add subtitle track": "track", "Burn into video": "burn"}
TASKS = ["Translate to English", "Keep Original Language"]

# Choices remembered between launches: settings key -> (variable on App, allowed values; None = on/off)
REMEMBERED = {
    "language": ("lang_var", LANGUAGES), "model": ("model_var", MODELS), "action": ("task_var", TASKS),
    "format": ("format_var", FORMATS), "compute": ("compute_var", COMPUTE_OPTIONS),
    "vad_filter": ("vad_var", None), "fast_mode": ("fast_mode_var", None),
    "video": ("video_output_var", list(VIDEO_OUTPUTS)), "readable_lines": ("readable_lines_var", None),
    "light_ui": ("theme_var", None),
}

QUEUE_COLOURS = {"waiting": "gray55", "working": "#3b8ed0", "done": "#28a745", "failed": "#dc3545",
                 "canceled": "gray55"}

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


def resource_path(name):
    """Finds a bundled file both when running from source and inside the packaged .exe."""
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, name)


def set_title_bar_dark(window):
    if platform.system() == "Windows":
        try:
            hwnd = ctypes.windll.user32.GetParent(window.winfo_id())
            ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 20, ctypes.byref(ctypes.c_int(1)),
                                                       ctypes.sizeof(ctypes.c_int(4)))
        except Exception:
            pass


def model_code(label):
    """'large-v3 (Slowest / Best Accuracy)' -> 'large-v3'"""
    return label.split(" ")[0].lower()


def model_label(code):
    """'large-v3' -> the full label shown in the AI Model menu"""
    return next(m for m in MODELS if model_code(m) == code)


def output_path_for(media_path, save_val, fmt, total_files):
    """Works out where one file's subtitles go."""
    if total_files == 1:
        return save_val
    base_name = os.path.splitext(os.path.basename(media_path))[0]
    if os.path.isdir(save_val):
        return os.path.join(save_val, f"{base_name}_subtitle{fmt}")
    return os.path.join(os.path.dirname(media_path), f"{base_name}_subtitle{fmt}")


class App:
    """The SubtitleSmith window. Jobs run on a worker thread that reports back through root.after()."""

    def __init__(self, root, settings_path=None):
        self.root = root
        self.settings_path = settings_path or settings.default_path()
        self.saved = settings.load(self.settings_path)
        folder = self.saved.get("last_folder")
        self.last_folder = folder if isinstance(folder, str) and os.path.isdir(folder) else ""
        self.selected_files = []
        self.queue = []  # progress.QueueItem for each file in the Queue tab
        self.current_index = None  # the queue item being worked on
        self.eta = None
        self.running = False
        self.model_chosen = False  # True once the model comes from saved settings or the user picks one
        self.batch_prefix = ""
        self.cancel_event = threading.Event()
        # Filled in by the hardware scan at startup; these defaults are used if the scan fails
        self.recommended_models = {"transcribe": "large-v3-turbo", "translate": "large-v3"}
        self.gpu_index = 0
        self.vram_gb = 0

        self._setup_window()
        self._create_variables()
        self._build_ui()
        self._start_hardware_scan()
        set_title_bar_dark(self.root)

    # ------------------------------------------------------------------ setup

    def _setup_window(self):
        self.root.title("SubtitleSmith")
        try:
            self.root.iconbitmap(resource_path("SubtitleSmith.ico"))
        except Exception:
            pass  # a missing icon shouldn't stop the app (and Linux can't use .ico files)
        self.root.geometry("780x650")
        self.root.resizable(False, False)
        self.root.attributes('-alpha', 1.0)
        try:
            self.root.tk_setPalette(background='#2b2b2b', foreground='#ffffff')
        except Exception:
            pass
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _create_variables(self):
        self.video_path_var = ctk.StringVar()
        self.save_path_var = ctk.StringVar()
        self.lang_var = ctk.StringVar(value="Auto-Detect (auto)")
        self.model_var = ctk.StringVar(value=MODELS[-1])
        self.task_var = ctk.StringVar(value="Translate to English")
        self.format_var = ctk.StringVar(value=".srt")
        self.compute_var = ctk.StringVar(value=COMPUTE_OPTIONS[0])
        self.vad_var = ctk.BooleanVar(value=True)
        self.theme_var = ctk.BooleanVar(value=False)
        self.video_output_var = ctk.StringVar(value="Subtitles only")
        self.fast_mode_var = ctk.BooleanVar(value=True)
        self.readable_lines_var = ctk.BooleanVar(value=True)
        self.status_var = ctk.StringVar(value="Ready")

        self._apply_saved_settings()
        self.format_var.trace_add("write", self.update_extension_in_path)

    def _build_ui(self):
        self.root.drop_target_register(DND_FILES)
        self.root.dnd_bind('<<Drop>>', self.drop_event)

        frame = ctk.CTkFrame(self.root)
        frame.pack(pady=20, padx=20, fill="both", expand=True)
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_columnconfigure(3, weight=1)

        # 0. Media Selection
        ctk.CTkLabel(frame, text="Media File(s):", width=100).grid(row=0, column=0, padx=10, pady=(15, 10), sticky="e")
        video_entry = ctk.CTkEntry(frame, textvariable=self.video_path_var, width=400)
        video_entry.grid(row=0, column=1, pady=(15, 10))
        video_entry.bind("<Double-1>", self.browse_video)
        ctk.CTkButton(frame, text="Browse", width=80, command=self.browse_video).grid(row=0, column=2, padx=10,
                                                                                     pady=(15, 10))

        # 1. Save Location
        ctk.CTkLabel(frame, text="Save To:", width=100).grid(row=1, column=0, padx=10, pady=5, sticky="e")
        save_entry = ctk.CTkEntry(frame, textvariable=self.save_path_var, width=400)
        save_entry.grid(row=1, column=1, pady=5)
        save_entry.bind("<Double-1>", self.browse_save)
        ctk.CTkButton(frame, text="Browse", width=80, command=self.browse_save).grid(row=1, column=2, padx=10, pady=5)

        # 2. Options
        options_frame = ctk.CTkFrame(frame, fg_color="transparent")
        options_frame.grid(row=2, column=0, columnspan=4, pady=10)
        ctk.CTkLabel(options_frame, text="Source Language:").pack(side="left", padx=(0, 5))
        ctk.CTkOptionMenu(options_frame, variable=self.lang_var, values=LANGUAGES, width=170).pack(side="left",
                                                                                                  padx=(0, 25))
        ctk.CTkLabel(options_frame, text="AI Model:").pack(side="left", padx=(0, 5))
        ctk.CTkOptionMenu(options_frame, variable=self.model_var, values=MODELS, width=270,
                          command=self.on_model_chosen).pack(side="left")

        # 3. Names & Terms (optional spelling hints for the AI)
        names_frame = ctk.CTkFrame(frame, fg_color="transparent")
        names_frame.grid(row=3, column=0, columnspan=4, pady=(0, 0))
        ctk.CTkLabel(names_frame, text="Names & Terms:").pack(side="left", padx=(0, 5))
        self.names_entry = ctk.CTkEntry(
            names_frame, width=523,
            placeholder_text="Optional: names or words to spell correctly, e.g. Frodo, Gandalf, Mordor")
        self.names_entry.pack(side="left")
        saved_names = self.saved.get("names_and_terms")
        if isinstance(saved_names, str) and saved_names:
            self.names_entry.insert(0, saved_names)

        # 4. Action Toggle
        action_frame = ctk.CTkFrame(frame, fg_color="transparent")
        action_frame.grid(row=4, column=0, columnspan=4, pady=10)
        ctk.CTkLabel(action_frame, text="Action:").pack(side="left", padx=10)
        ctk.CTkSegmentedButton(action_frame, variable=self.task_var,
                               values=["Translate to English", "Keep Original Language"],
                               command=self.on_task_change, width=320).pack(side="left", padx=10)

        # 5. Advanced Settings Row 1
        adv_frame = ctk.CTkFrame(frame, fg_color="transparent")
        adv_frame.grid(row=5, column=0, columnspan=4, pady=(5, 0))
        ctk.CTkLabel(adv_frame, text="Format:").pack(side="left", padx=(0, 5))
        ctk.CTkOptionMenu(adv_frame, variable=self.format_var, values=FORMATS, width=70).pack(side="left",
                                                                                             padx=(0, 15))
        ctk.CTkLabel(adv_frame, text="Compute:").pack(side="left", padx=(0, 5))
        ctk.CTkOptionMenu(adv_frame, variable=self.compute_var, values=COMPUTE_OPTIONS, width=150).pack(side="left",
                                                                                                       padx=(0, 15))
        ctk.CTkSwitch(adv_frame, text="VAD Filter", variable=self.vad_var).pack(side="left", padx=(0, 15))
        ctk.CTkSwitch(adv_frame, text="Fast Mode", variable=self.fast_mode_var).pack(side="left")

        # 6. Advanced Settings Row 2
        adv2_frame = ctk.CTkFrame(frame, fg_color="transparent")
        adv2_frame.grid(row=6, column=0, columnspan=4, pady=(5, 5))
        ctk.CTkLabel(adv2_frame, text="Video:").pack(side="left", padx=(0, 5))
        ctk.CTkOptionMenu(adv2_frame, variable=self.video_output_var, values=list(VIDEO_OUTPUTS),
                          width=170).pack(side="left", padx=(0, 20))
        ctk.CTkSwitch(adv2_frame, text="Readable Lines", variable=self.readable_lines_var).pack(side="left",
                                                                                              padx=(0, 20))
        ctk.CTkSwitch(adv2_frame, text="Light UI", variable=self.theme_var, command=self.toggle_theme).pack(side="left")

        # 7. Status
        status_frame = ctk.CTkFrame(frame, fg_color="transparent")
        status_frame.grid(row=7, column=0, columnspan=4, pady=(15, 0))
        ctk.CTkLabel(status_frame, textvariable=self.status_var, text_color="gray").pack(pady=(0, 5))
        self.progress_bar = ctk.CTkProgressBar(status_frame, width=460)
        self.progress_bar.set(0)
        self.progress_bar.pack()

        # 8. Buttons
        button_frame = ctk.CTkFrame(frame, fg_color="transparent")
        button_frame.grid(row=8, column=0, columnspan=4, pady=(15, 10))
        self.start_button = ctk.CTkButton(button_frame, text="Start Processing", command=self.start_process,
                                          fg_color="#28a745", hover_color="#218838")
        self.start_button.pack(side="left", padx=15)
        self.cancel_button = ctk.CTkButton(button_frame, text="Cancel", command=self.cancel_process,
                                           fg_color="#dc3545", hover_color="#c82333", state="disabled")
        self.cancel_button.pack(side="left", padx=15)

        # 9. Queue and system info
        self.tabs = ctk.CTkTabview(frame, width=620, height=185)
        self.tabs.grid(row=9, column=0, columnspan=4, pady=(0, 10))
        monospace = ctk.CTkFont(family="Consolas" if platform.system() == "Windows" else "DejaVu Sans Mono", size=12)
        self.queue_box = ctk.CTkTextbox(self.tabs.add("Queue"), width=600, height=130, font=monospace, wrap="none")
        self.queue_box.pack(fill="both", expand=True)
        for status, colour in QUEUE_COLOURS.items():
            self.queue_box.tag_config(status, foreground=colour)
        self.queue_box.insert("0.0", "Choose or drop media files and they'll be listed here.")
        self.queue_box.configure(state="disabled")
        self.hw_textbox = ctk.CTkTextbox(self.tabs.add("System"), width=600, height=130, text_color="#28a745",
                                         fg_color="#1e1e1e")
        self.hw_textbox.pack(fill="both", expand=True)
        self.tabs.set("System")

    def _start_hardware_scan(self):
        self._set_hardware_text("Scanning your hardware...")
        threading.Thread(target=self._scan_hardware, daemon=True).start()

    def _scan_hardware(self):
        """Runs on a helper thread, so the window opens straight away."""
        try:
            scan, error = hardware.scan_system(), None
        except Exception as e:
            scan, error = None, e
        try:
            self.root.after(0, lambda: self._show_hardware_scan(scan, error))
        except Exception:
            pass  # the window was closed before the scan finished

    def _show_hardware_scan(self, scan, error):
        if scan is None:
            self._set_hardware_text(f"Could not perform hardware scan.\nError: {error}")
            return
        self._set_hardware_text(scan["report"])
        self.recommended_models = scan["recommended"]
        self.gpu_index = scan["gpu_index"]
        self.vram_gb = scan["vram_gb"]
        if not self.model_chosen:
            self.model_var.set(model_label(self.recommended_models[self.current_task()]))

    def _set_hardware_text(self, text):
        self.hw_textbox.configure(state="normal")
        self.hw_textbox.delete("0.0", "end")
        self.hw_textbox.insert("0.0", text)
        self.hw_textbox.configure(state="disabled")

    # ------------------------------------------------------------------ remembered settings

    def _apply_saved_settings(self):
        for key, (attribute, allowed) in REMEMBERED.items():
            value = self.saved.get(key)
            if isinstance(value, bool) if allowed is None else value in allowed:
                getattr(self, attribute).set(value)
                if key == "model":
                    self.model_chosen = True
        if self.theme_var.get():
            ctk.set_appearance_mode("light")

    def _save_settings(self):
        values = {key: getattr(self, attribute).get() for key, (attribute, _) in REMEMBERED.items()}
        values["names_and_terms"] = self.names_entry.get().strip()
        if self.last_folder:
            values["last_folder"] = self.last_folder
        settings.save(values, self.settings_path)

    def on_close(self):
        self._save_settings()
        self.cancel_event.set()  # stops a model download in progress
        video.stop_all()  # don't leave FFmpeg encoding after the window has gone
        self.root.destroy()

    # ------------------------------------------------------------------ options

    def toggle_theme(self):
        mode = "light" if self.theme_var.get() else "dark"
        ctk.set_appearance_mode(mode)

    def current_task(self):
        return "translate" if "Translate" in self.task_var.get() else "transcribe"

    def on_model_chosen(self, value=None):
        self.model_chosen = True

    def on_task_change(self, value=None):
        """Keeps the AI model suitable when switching between Translate and Keep Original."""
        task = self.current_task()
        current = model_code(self.model_var.get())
        if current in self.recommended_models.values():
            # Still on an auto-pick, so move to the auto-pick for this job
            self.model_var.set(model_label(self.recommended_models[task]))
        elif task == "translate" and current in hardware.NO_TRANSLATION:
            better = self.recommended_models["translate"]
            self.model_var.set(model_label(better))
            self.status_var.set(f"Switched AI model to '{better}': '{current}' can't translate.")

    # ------------------------------------------------------------------ choosing files

    def _busy(self):
        """True (after telling the user) if a queue is running, when new files can't be chosen."""
        if self.running:
            messagebox.showinfo("Still working", "SubtitleSmith is still working through the queue. Wait for it "
                                                 "to finish, or press Cancel, before choosing new files.")
        return self.running

    def browse_video(self, event=None):
        if self._busy():
            return
        all_exts = " ".join([f"*{ext}" for ext in SUPPORTED_MEDIA])
        filepaths = filedialog.askopenfilenames(
            title="Select Media Files", initialdir=self.last_folder or None,
            filetypes=[("Media Files", all_exts), ("Video Files", "*.mp4 *.mkv *.avi *.mov"),
                       ("Audio Files", "*.mp3 *.wav *.aac"), ("All Files", "*.*")]
        )
        if filepaths:
            self.process_selected_files(filepaths)

    def browse_save(self, event=None):
        if len(self.selected_files) > 1:
            directory = filedialog.askdirectory(title="Select Batch Output Folder", initialdir=self.last_folder or None)
            if directory:
                self.save_path_var.set(directory)
        else:
            ext = self.format_var.get()
            filepath = filedialog.asksaveasfilename(
                title="Save File As", initialdir=self.last_folder or None,
                defaultextension=ext,
                filetypes=[(f"{ext.upper()} File", f"*{ext}"), ("All Files", "*.*")]
            )
            if filepath: self.save_path_var.set(filepath)

    def process_selected_files(self, filepaths):
        self.selected_files = list(filepaths)
        ext = self.format_var.get()
        if self.selected_files:
            self.last_folder = os.path.dirname(self.selected_files[0])
        self.queue = [progress.QueueItem(path) for path in self.selected_files]
        self._show_queue()
        self.tabs.set("Queue")

        if len(self.selected_files) == 1:
            filepath = self.selected_files[0]
            self.video_path_var.set(filepath)
            directory = os.path.dirname(filepath)
            base_name = os.path.splitext(os.path.basename(filepath))[0]
            self.save_path_var.set(os.path.join(directory, f"{base_name}_subtitle{ext}"))
        elif len(self.selected_files) > 1:
            self.video_path_var.set(f"{len(self.selected_files)} media files selected (Batch Queue)")
            self.save_path_var.set(AUTO_SAVE)

    def update_extension_in_path(self, *args):
        current_path = self.save_path_var.get()
        if current_path and not current_path.startswith("<"):
            if len(self.selected_files) > 1 and os.path.isdir(current_path):
                pass
            else:
                base, _ = os.path.splitext(current_path)
                self.save_path_var.set(f"{base}{self.format_var.get()}")

    def drop_event(self, event):
        if self._busy():
            return
        files = self.root.tk.splitlist(event.data)
        valid_files = [f for f in files if f.lower().endswith(SUPPORTED_MEDIA)]
        if valid_files:
            self.process_selected_files(valid_files)
        else:
            messagebox.showwarning("Invalid File", "Please drop supported video or audio files.")

    # ------------------------------------------------------------------ running jobs

    def _show_queue(self):
        """Redraws the whole Queue tab. Main thread only."""
        self.queue_box.configure(state="normal")
        self.queue_box.delete("0.0", "end")
        for i, item in enumerate(self.queue):
            status, name, detail = progress.queue_columns(item)
            self.queue_box.insert("end", status, item.status)
            self.queue_box.insert("end", name + detail + ("\n" if i < len(self.queue) - 1 else ""))
        self.queue_box.configure(state="disabled")

    def _set_item(self, index, status, detail):
        """Updates one line of the Queue tab. Main thread only."""
        if index >= len(self.queue):
            return
        item = self.queue[index]
        item.status, item.detail = status, detail
        line = f"{index + 1}.0"
        status_text, name, detail_text = progress.queue_columns(item)
        self.queue_box.configure(state="normal")
        self.queue_box.delete(line, f"{index + 1}.end")
        self.queue_box.insert(line, status_text, status)
        self.queue_box.insert(f"{index + 1}.end", name + detail_text)
        self.queue_box.configure(state="disabled")
        self.queue_box.see(line)

    def _post_item(self, index, status, detail=""):
        """_set_item from the worker thread."""
        self.root.after(0, lambda: self._set_item(index, status, detail))

    @staticmethod
    def _working_detail(stage, fraction, status_text):
        """What the Queue tab shows for the file being worked on, or None to leave it."""
        if stage == "prepare":
            return "Reading audio"
        if stage == "transcribe":
            return f"{round(fraction * 100)}%"
        if stage == "save":
            return "Saving"
        if stage == "video":
            return "Adding track" if "track" in status_text else f"Burning {round(fraction * 100)}%"
        return None

    def update_gui_progress(self, current, total, status_text, stage=""):
        """Progress callback for the engine. Called from the worker thread."""
        fraction = current / total if total > 0 else 0
        index = self.current_index
        display_text = f"{self.batch_prefix}{status_text}" if self.batch_prefix else status_text
        if self.eta and index is not None:
            if stage == "transcribe":
                self.eta.progress(index, fraction)
            elif stage in ("save", "video"):
                self.eta.progress(index, 1.0)
            left = self.eta.seconds_left(time.monotonic())
            if left is not None and stage in ("prepare", "transcribe", "save", "video"):
                display_text += f"  \u00b7  {progress.format_time_left(left)}"
        detail = self._working_detail(stage, fraction, status_text) if index is not None else None

        def show():
            self.progress_bar.set(fraction)
            self.status_var.set(display_text)
            if detail is not None:
                self._set_item(index, "working", detail)
        self.root.after(0, show)

    def check_cancel(self):
        return self.cancel_event.is_set()

    def cancel_process(self):
        self.cancel_event.set()
        self.status_var.set("Canceling... Please wait.")
        self.cancel_button.configure(state="disabled")

    def start_process(self):
        if not self.selected_files or not self.save_path_var.get():
            messagebox.showwarning("Missing Information", "Please select media files and a save location.")
            return

        lang_code = self.lang_var.get().split("(")[-1].replace(")", "")
        model_sz = model_code(self.model_var.get())
        task_code = self.current_task()
        compute_code = "cpu" if "CPU" in self.compute_var.get() else "auto"
        fmt = self.format_var.get()
        video_output = VIDEO_OUTPUTS[self.video_output_var.get()]

        # Catch a model that can't do this job before spending time on it
        problem = hardware.model_job_problem(model_sz, task_code, lang_code)
        if problem:
            better = self.recommended_models[task_code]
            if hardware.model_job_problem(better, task_code, lang_code):
                better = "large-v3"
            answer = messagebox.askyesnocancel(
                "AI model can't do this job",
                f"{problem}\n\nSwitch to '{better}' for this job?\n\n"
                f"Yes: switch and start\nNo: start with '{model_sz}' anyway\nCancel: go back")
            if answer is None:
                return
            if answer:
                self.model_var.set(model_label(better))
                model_sz = better

        # Check for FFmpeg now rather than after a long transcription
        if video_output != "none" and fmt in (".srt", ".vtt") and not video.ffmpeg_available():
            if not messagebox.askyesno(
                    "FFmpeg not found",
                    "Adding subtitles to videos needs FFmpeg, which isn't installed or isn't on your PATH.\n\n"
                    "Continue and make subtitle files only?"):
                return
            video_output = "none"

        # Fast Mode: transcribe several pieces at once on the GPU, as many as its memory allows
        batch_size = hardware.batch_size_for(self.vram_gb) if self.fast_mode_var.get() and compute_code != "cpu" else 0

        self._save_settings()
        self.queue = [progress.QueueItem(path) for path in self.selected_files]
        self._show_queue()
        self.tabs.set("Queue")
        self.running = True
        self.cancel_event.clear()
        self.start_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.progress_bar.set(0)

        threading.Thread(
            target=self.run_queue,
            args=(self.selected_files.copy(), lang_code, self.save_path_var.get(), model_sz, task_code, compute_code,
                  self.vad_var.get(), fmt, video_output, self.readable_lines_var.get(),
                  self.names_entry.get().strip(), batch_size),
            daemon=True
        ).start()

    def run_queue(self, media_files, lang_code, save_val, model_sz, task_type, compute, vad, fmt, video_output,
                  readable_lines, names_and_terms, batch_size=0):
        """Worker thread: processes each file in turn, then shows a summary."""
        total_files = len(media_files)
        device_index = 0 if compute == "cpu" else self.gpu_index
        succeeded, failed, notes, detected = [], [], [], []
        cancelled = False
        load_error = None
        self.eta = progress.Eta([engine.media_duration(path) for path in media_files])  # quick: reads headers only

        try:
            # Load the model once for the whole queue (downloading it on first use). If it won't load,
            # every file would fail the same way.
            try:
                engine.get_model(model_sz, compute, device_index, self.update_gui_progress, self.check_cancel)
            except engine.ProcessCancelled:
                cancelled = True
                return
            except Exception as e:
                load_error = str(e) or type(e).__name__
                return
            self.eta.start(time.monotonic())

            for i, media in enumerate(media_files):
                if self.check_cancel():
                    cancelled = True
                    break
                name = os.path.basename(media)
                self.batch_prefix = f"[{i + 1}/{total_files}] " if total_files > 1 else ""
                self.current_index = i
                self._post_item(i, "working", "Starting")
                file_started = time.monotonic()
                try:
                    result = engine.process_file(
                        media_path=media, output_file=output_path_for(media, save_val, fmt, total_files),
                        model_size=model_sz, source_language=lang_code, task=task_type,
                        compute_device=compute, device_index=device_index, vad_filter=vad,
                        output_format=fmt, readable_lines=readable_lines, names_and_terms=names_and_terms,
                        video_output=video_output, batch_size=batch_size,
                        progress_callback=self.update_gui_progress, cancel_check=self.check_cancel
                    )
                except engine.ProcessCancelled:
                    cancelled = True
                    break
                except Exception as e:
                    # Note the failure and carry on with the rest of the queue
                    error = str(e) or type(e).__name__
                    failed.append((name, error))
                    self._post_item(i, "failed", error.splitlines()[0][:150])
                    self.eta.file_finished(i)
                    continue
                succeeded.append(name)
                notes += [f"{name}: {note}" for note in result["notes"]]
                details = [progress.format_duration(time.monotonic() - file_started)]
                if result["language_detected"]:
                    language = languages.describe(result["language"], result["language_probability"])
                    detected.append((name, language))
                    details.append(language)
                if result["notes"]:
                    details.append("see notes")
                self._post_item(i, "done", " \u00b7 ".join(details))
                self.eta.file_finished(i)
        finally:
            self.batch_prefix = ""
            self.current_index = None
            self.root.after(0, lambda: self.show_summary(total_files, succeeded, failed, notes, cancelled, load_error,
                                                         detected))
            self.root.after(0, lambda: self.start_button.configure(state="normal"))
            self.root.after(0, lambda: self.cancel_button.configure(state="disabled"))

    def show_summary(self, total, succeeded, failed, notes, cancelled, load_error, detected=()):
        """Runs on the main thread once the queue finishes."""
        self.running = False
        if cancelled or load_error:
            for item in self.queue:
                if item.status in ("waiting", "working"):
                    item.status, item.detail = "canceled", ""
            self._show_queue()
        if load_error:
            self.status_var.set("Couldn't load the AI model.")
            self.progress_bar.set(0)
            messagebox.showerror("Couldn't load the AI model", load_error)
            return

        if cancelled:
            self.status_var.set(f"Canceled. {len(succeeded)} of {total} file(s) finished.")
        elif failed:
            self.status_var.set(f"Finished with errors: {len(succeeded)} of {total} file(s) succeeded, "
                                f"{len(failed)} failed.")
        else:
            status = f"Successfully processed {total} file(s)."
            if total == 1 and detected:
                status += f" Detected language: {detected[0][1]}."
            self.status_var.set(status)
            self.progress_bar.set(1.0)

        lines = [f"{len(succeeded)} of {total} file(s) processed."]
        if failed:
            lines += ["", f"Failed ({len(failed)}):"]
            lines += [f"• {name}: {error[:300]}" for name, error in failed[:10]]
            if len(failed) > 10: lines.append(f"...and {len(failed) - 10} more.")
        if detected and total > 1:
            lines += ["", "Detected languages:"]
            lines += [f"\u2022 {name}: {language}" for name, language in detected[:10]]
            if len(detected) > 10: lines.append(f"...and {len(detected) - 10} more.")
        if notes:
            lines += ["", "Notes:"]
            lines += [f"• {note}" for note in notes[:10]]
            if len(notes) > 10: lines.append(f"...and {len(notes) - 10} more.")

        if failed:
            messagebox.showwarning("Finished with errors", "\n".join(lines))
        elif notes or (total > 1 and not cancelled):
            messagebox.showinfo("Finished", "\n".join(lines))


def main():
    root = TkinterDnD.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
