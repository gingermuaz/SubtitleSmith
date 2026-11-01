import os
import sys
import threading
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

import burn
import engine
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

AUTO_SAVE = "<Auto-save to original directories>"

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


def output_path_for(video, save_val, fmt, total_files):
    """Works out where one file's subtitles go."""
    if total_files == 1:
        return save_val
    base_name = os.path.splitext(os.path.basename(video))[0]
    if os.path.isdir(save_val):
        return os.path.join(save_val, f"{base_name}_subtitle{fmt}")
    return os.path.join(os.path.dirname(video), f"{base_name}_subtitle{fmt}")


class App:
    """The SubtitleSmith window. Jobs run on a worker thread that reports back through root.after()."""

    def __init__(self, root):
        self.root = root
        self.selected_files = []
        self.batch_prefix = ""
        self.cancel_event = threading.Event()
        # Filled in by the hardware scan at startup; these defaults are used if the scan fails
        self.recommended_models = {"transcribe": "large-v3-turbo", "translate": "large-v3"}
        self.gpu_index = 0

        self._setup_window()
        self._create_variables()
        self._build_ui()
        self._run_hardware_scan()
        set_title_bar_dark(self.root)

    # ------------------------------------------------------------------ setup

    def _setup_window(self):
        self.root.title("SubtitleSmith")
        try:
            self.root.iconbitmap(resource_path("SubtitleSmith.ico"))
        except Exception:
            pass  # a missing icon shouldn't stop the app (and Linux can't use .ico files)
        self.root.geometry("780x580")
        self.root.resizable(False, False)
        self.root.attributes('-alpha', 1.0)
        try:
            self.root.tk_setPalette(background='#2b2b2b', foreground='#ffffff')
        except Exception:
            pass

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
        self.hardcode_var = ctk.BooleanVar(value=False)
        self.word_timestamps_var = ctk.BooleanVar(value=False)
        self.status_var = ctk.StringVar(value="Ready")

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
        ctk.CTkOptionMenu(options_frame, variable=self.model_var, values=MODELS, width=270).pack(side="left")

        # 3. Action Toggle
        action_frame = ctk.CTkFrame(frame, fg_color="transparent")
        action_frame.grid(row=3, column=0, columnspan=4, pady=10)
        ctk.CTkLabel(action_frame, text="Action:").pack(side="left", padx=10)
        ctk.CTkSegmentedButton(action_frame, variable=self.task_var,
                               values=["Translate to English", "Keep Original Language"],
                               command=self.on_task_change, width=320).pack(side="left", padx=10)

        # 4. Advanced Settings Row 1
        adv_frame = ctk.CTkFrame(frame, fg_color="transparent")
        adv_frame.grid(row=4, column=0, columnspan=4, pady=(5, 0))
        ctk.CTkLabel(adv_frame, text="Format:").pack(side="left", padx=(0, 5))
        ctk.CTkOptionMenu(adv_frame, variable=self.format_var, values=FORMATS, width=70).pack(side="left",
                                                                                             padx=(0, 15))
        ctk.CTkLabel(adv_frame, text="Compute:").pack(side="left", padx=(0, 5))
        ctk.CTkOptionMenu(adv_frame, variable=self.compute_var, values=COMPUTE_OPTIONS, width=150).pack(side="left",
                                                                                                       padx=(0, 15))
        ctk.CTkSwitch(adv_frame, text="VAD Filter", variable=self.vad_var).pack(side="left", padx=(0, 15))

        # 5. Advanced Settings Row 2
        adv2_frame = ctk.CTkFrame(frame, fg_color="transparent")
        adv2_frame.grid(row=5, column=0, columnspan=4, pady=(5, 5))
        ctk.CTkSwitch(adv2_frame, text="Burn Subtitles to Video", variable=self.hardcode_var).pack(side="left",
                                                                                                  padx=(0, 20))
        ctk.CTkSwitch(adv2_frame, text="Word Timestamps", variable=self.word_timestamps_var).pack(side="left",
                                                                                                 padx=(0, 20))
        ctk.CTkSwitch(adv2_frame, text="Light UI", variable=self.theme_var, command=self.toggle_theme).pack(side="left")

        # 6. Status
        status_frame = ctk.CTkFrame(frame, fg_color="transparent")
        status_frame.grid(row=6, column=0, columnspan=4, pady=(15, 0))
        ctk.CTkLabel(status_frame, textvariable=self.status_var, text_color="gray").pack(pady=(0, 5))
        self.progress_bar = ctk.CTkProgressBar(status_frame, width=460)
        self.progress_bar.set(0)
        self.progress_bar.pack()

        # 7. Buttons
        button_frame = ctk.CTkFrame(frame, fg_color="transparent")
        button_frame.grid(row=7, column=0, columnspan=4, pady=(15, 10))
        self.start_button = ctk.CTkButton(button_frame, text="Start Processing", command=self.start_process,
                                          fg_color="#28a745", hover_color="#218838")
        self.start_button.pack(side="left", padx=15)
        self.cancel_button = ctk.CTkButton(button_frame, text="Cancel", command=self.cancel_process,
                                           fg_color="#dc3545", hover_color="#c82333", state="disabled")
        self.cancel_button.pack(side="left", padx=15)

        # 8. Hardware Info
        hw_frame = ctk.CTkFrame(frame, fg_color="transparent")
        hw_frame.grid(row=8, column=0, columnspan=4, pady=(0, 10))
        self.hw_textbox = ctk.CTkTextbox(hw_frame, width=600, height=130, text_color="#28a745", fg_color="#1e1e1e")
        self.hw_textbox.pack()

    def _run_hardware_scan(self):
        try:
            scan = hardware.scan_system()
            self.hw_textbox.insert("0.0", scan["report"])
            self.recommended_models = scan["recommended"]
            self.gpu_index = scan["gpu_index"]
            self.model_var.set(model_label(self.recommended_models[self.current_task()]))
        except Exception as e:
            self.hw_textbox.insert("0.0", f"Could not perform hardware scan.\nError: {e}")
        self.hw_textbox.configure(state="disabled")

    # ------------------------------------------------------------------ options

    def toggle_theme(self):
        mode = "light" if self.theme_var.get() else "dark"
        ctk.set_appearance_mode(mode)

    def current_task(self):
        return "translate" if "Translate" in self.task_var.get() else "transcribe"

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

    def browse_video(self, event=None):
        all_exts = " ".join([f"*{ext}" for ext in SUPPORTED_MEDIA])
        filepaths = filedialog.askopenfilenames(
            title="Select Media Files",
            filetypes=[("Media Files", all_exts), ("Video Files", "*.mp4 *.mkv *.avi *.mov"),
                       ("Audio Files", "*.mp3 *.wav *.aac"), ("All Files", "*.*")]
        )
        if filepaths:
            self.process_selected_files(filepaths)

    def browse_save(self, event=None):
        if len(self.selected_files) > 1:
            directory = filedialog.askdirectory(title="Select Batch Output Folder")
            if directory:
                self.save_path_var.set(directory)
        else:
            ext = self.format_var.get()
            filepath = filedialog.asksaveasfilename(
                title="Save File As",
                defaultextension=ext,
                filetypes=[(f"{ext.upper()} File", f"*{ext}"), ("All Files", "*.*")]
            )
            if filepath: self.save_path_var.set(filepath)

    def process_selected_files(self, filepaths):
        self.selected_files = list(filepaths)
        ext = self.format_var.get()

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
        files = self.root.tk.splitlist(event.data)
        valid_files = [f for f in files if f.lower().endswith(SUPPORTED_MEDIA)]
        if valid_files:
            self.process_selected_files(valid_files)
        else:
            messagebox.showwarning("Invalid File", "Please drop supported video or audio files.")

    # ------------------------------------------------------------------ running jobs

    def update_gui_progress(self, current, total, status_text):
        """Progress callback for the engine. Called from the worker thread."""
        percent = current / total if total > 0 else 0
        self.root.after(0, lambda: self.progress_bar.set(percent))
        display_text = f"{self.batch_prefix}{status_text}" if self.batch_prefix else status_text
        self.root.after(0, lambda: self.status_var.set(display_text))

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
        hardcode = self.hardcode_var.get()

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
        if hardcode and fmt in (".srt", ".vtt") and not burn.ffmpeg_available():
            if not messagebox.askyesno(
                    "FFmpeg not found",
                    "Burn Subtitles to Video is on, but FFmpeg isn't installed or isn't on your PATH.\n\n"
                    "Continue and make subtitle files only?"):
                return
            hardcode = False

        self.cancel_event.clear()
        self.start_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.progress_bar.set(0)

        threading.Thread(
            target=self.run_queue,
            args=(self.selected_files.copy(), lang_code, self.save_path_var.get(), model_sz, task_code, compute_code,
                  self.vad_var.get(), fmt, hardcode, self.word_timestamps_var.get()),
            daemon=True
        ).start()

    def run_queue(self, videos, lang_code, save_val, model_sz, task_type, compute, vad, fmt, hardcode, word_ts):
        """Worker thread: processes each file in turn, then shows a summary."""
        total_files = len(videos)
        device_index = 0 if compute == "cpu" else self.gpu_index
        succeeded, failed, notes = [], [], []
        cancelled = False
        load_error = None

        try:
            # Load the model once for the whole queue. If it won't load, every file would fail the same way.
            try:
                engine.get_model(model_sz, compute, device_index, self.update_gui_progress)
            except Exception as e:
                load_error = str(e) or type(e).__name__
                return

            for i, video in enumerate(videos):
                if self.check_cancel():
                    cancelled = True
                    break
                name = os.path.basename(video)
                self.batch_prefix = f"[{i + 1}/{total_files}] " if total_files > 1 else ""
                try:
                    result = engine.process_file(
                        media_path=video, output_file=output_path_for(video, save_val, fmt, total_files),
                        model_size=model_sz, source_language=lang_code, task=task_type,
                        compute_device=compute, device_index=device_index, vad_filter=vad,
                        output_format=fmt, word_timestamps=word_ts, hardcode_subtitles=hardcode,
                        progress_callback=self.update_gui_progress, cancel_check=self.check_cancel
                    )
                except engine.ProcessCancelled:
                    cancelled = True
                    break
                except Exception as e:
                    # Note the failure and carry on with the rest of the queue
                    failed.append((name, str(e) or type(e).__name__))
                    continue
                succeeded.append(name)
                notes += [f"{name}: {note}" for note in result["notes"]]
        finally:
            self.batch_prefix = ""
            self.root.after(0, lambda: self.show_summary(total_files, succeeded, failed, notes, cancelled, load_error))
            self.root.after(0, lambda: self.start_button.configure(state="normal"))
            self.root.after(0, lambda: self.cancel_button.configure(state="disabled"))

    def show_summary(self, total, succeeded, failed, notes, cancelled, load_error):
        """Runs on the main thread once the queue finishes."""
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
            self.status_var.set(f"Successfully processed {total} file(s).")
            self.progress_bar.set(1.0)

        lines = [f"{len(succeeded)} of {total} file(s) processed."]
        if failed:
            lines += ["", f"Failed ({len(failed)}):"]
            lines += [f"• {name}: {error[:300]}" for name, error in failed[:10]]
            if len(failed) > 10: lines.append(f"...and {len(failed) - 10} more.")
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
