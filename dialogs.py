"""SubtitleSmith's pop-up windows: reviewing subtitles before they're saved, the burn-in style, and the
Hugging Face token that speaker labels need."""
import os
import webbrowser
from tkinter import messagebox

import customtkinter as ctk

import speakers
import video
import writers


class ReviewDialog(ctk.CTkToplevel):
    """Shows one file's subtitles (a tab per language) so words can be fixed before they're saved.

    on_done(texts) is called once: with {path: text} on Save, or None on Cancel Queue.
    """

    def __init__(self, parent, title, texts, on_done):
        super().__init__(parent)
        self.title(f"Review Subtitles - {title}")
        self.geometry("760x560")
        self.transient(parent)
        self.on_done = on_done
        self.finished = False
        self.boxes = {}

        ctk.CTkLabel(self, text="Check the subtitles and fix any words, then press Save and Continue. "
                                "Leave each number and timing line as it is.",
                     wraplength=720, justify="left").pack(padx=15, pady=(12, 6), anchor="w")
        find_row = ctk.CTkFrame(self, fg_color="transparent")
        find_row.pack(fill="x", padx=15)
        ctk.CTkLabel(find_row, text="Find:").pack(side="left", padx=(0, 5))
        self.find_entry = ctk.CTkEntry(find_row, width=190)
        self.find_entry.pack(side="left")
        ctk.CTkLabel(find_row, text="Replace with:").pack(side="left", padx=(12, 5))
        self.replace_entry = ctk.CTkEntry(find_row, width=190)
        self.replace_entry.pack(side="left")
        ctk.CTkButton(find_row, text="Replace All", width=100, command=self.replace_all).pack(side="left", padx=10)
        self.replaced_label = ctk.CTkLabel(find_row, text="", text_color="gray")
        self.replaced_label.pack(side="left")

        tabs = ctk.CTkTabview(self)
        tabs.pack(fill="both", expand=True, padx=15, pady=5)
        for path, text in texts.items():
            box = ctk.CTkTextbox(tabs.add(os.path.basename(path)), wrap="word", undo=True)
            box.pack(fill="both", expand=True)
            box.insert("0.0", text)
            self.boxes[path] = box

        buttons = ctk.CTkFrame(self, fg_color="transparent")
        buttons.pack(fill="x", padx=15, pady=(5, 12))
        ctk.CTkButton(buttons, text="Save and Continue", fg_color="#28a745", hover_color="#218838",
                      command=self.save).pack(side="right")
        ctk.CTkButton(buttons, text="Cancel Queue", fg_color="#dc3545", hover_color="#c82333",
                      command=lambda: self.finish(None)).pack(side="right", padx=10)
        self.protocol("WM_DELETE_WINDOW", self.save)  # closing the window keeps the subtitles as they are
        self.after(150, self.focus_force)

    def edited(self):
        return {path: box.get("0.0", "end-1c") for path, box in self.boxes.items()}

    def replace_all(self):
        find, replacement = self.find_entry.get(), self.replace_entry.get()
        if not find:
            return
        count = 0
        for box in self.boxes.values():
            text = box.get("0.0", "end-1c")
            if find in text:
                count += text.count(find)
                box.delete("0.0", "end")
                box.insert("0.0", text.replace(find, replacement))
        self.replaced_label.configure(text=f"{count} replaced")

    def save(self):
        texts = self.edited()
        found = [f"{os.path.basename(path)}: {problem}"
                 for path, text in texts.items() for problem in writers.problems(text, os.path.splitext(path)[1].lower())]
        if found and not messagebox.askyesno(
                "Check the subtitles", "Some subtitles may not work in video players:\n\n" + "\n".join(found[:8])
                + ("\n..." if len(found) > 8 else "") + "\n\nSave anyway?", parent=self):
            return
        self.finish(texts)

    def finish(self, texts):
        if not self.finished:
            self.finished = True
            self.on_done(texts)
        self.destroy()


class StyleDialog(ctk.CTkToplevel):
    """Size, colour and position of burned-in subtitles."""

    def __init__(self, parent, size_var, colour_var, position_var):
        super().__init__(parent)
        self.title("Burn-in Style")
        self.geometry("330x210")
        self.resizable(False, False)
        self.transient(parent)
        ctk.CTkLabel(self, text="How burned-in subtitles look:").grid(row=0, column=0, columnspan=2, padx=15,
                                                                      pady=(15, 10), sticky="w")
        for row, (label, variable, values) in enumerate(
                [("Size:", size_var, video.BURN_SIZES), ("Colour:", colour_var, video.BURN_COLOURS),
                 ("Position:", position_var, video.BURN_POSITIONS)], start=1):
            ctk.CTkLabel(self, text=label).grid(row=row, column=0, padx=(15, 8), pady=5, sticky="e")
            ctk.CTkOptionMenu(self, variable=variable, values=list(values), width=170).grid(row=row, column=1, pady=5,
                                                                                          sticky="w")
        ctk.CTkButton(self, text="Done", width=90, command=self.destroy).grid(row=4, column=1, pady=(12, 10), sticky="e")
        self.after(150, self.focus_force)


class TokenDialog(ctk.CTkToplevel):
    """Asks for the Hugging Face access token speaker labels need. self.token is None if cancelled."""

    def __init__(self, parent):
        super().__init__(parent)
        self.title("Speaker Labels Setup")
        self.geometry("520x330")
        self.resizable(False, False)
        self.transient(parent)
        self.token = None
        ctk.CTkLabel(self, wraplength=480, justify="left", text=(
            "Speaker labels use pyannote's speaker model, which needs a free Hugging Face account. One-time setup:\n\n"
            "1.  Sign in to Hugging Face and accept the model's terms.\n"
            "2.  Create an access token (the \"Read\" type is enough).\n"
            "3.  Paste the token below. It's saved where Hugging Face tools keep it, not in SubtitleSmith.")
        ).pack(padx=20, pady=(18, 10), anchor="w")
        links = ctk.CTkFrame(self, fg_color="transparent")
        links.pack(padx=20, anchor="w")
        ctk.CTkButton(links, text="1. Open the model's terms", width=200,
                      command=lambda: webbrowser.open(speakers.TERMS_URL)).pack(side="left")
        ctk.CTkButton(links, text="2. Create a token", width=160,
                      command=lambda: webbrowser.open(speakers.TOKEN_URL)).pack(side="left", padx=10)
        self.entry = ctk.CTkEntry(self, width=480, show="•", placeholder_text="hf_...")
        self.entry.pack(padx=20, pady=(18, 10))
        buttons = ctk.CTkFrame(self, fg_color="transparent")
        buttons.pack(fill="x", padx=20, pady=(0, 15))
        ctk.CTkButton(buttons, text="Save Token", width=110, command=self.ok).pack(side="right")
        ctk.CTkButton(buttons, text="Cancel", width=90, fg_color="gray40", command=self.destroy).pack(side="right",
                                                                                                    padx=10)
        self.after(150, self.entry.focus_force)

    def ok(self):
        self.token = self.entry.get().strip() or None
        self.destroy()
