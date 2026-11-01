"""SubtitleSmith from the command line, for scripts and whole folders.

Examples:
  python cli.py film.mkv
  python cli.py "D:\\Films" --recursive --task transcribe --also-english --video track
  python cli.py talk.mp4 --language en --translate-to bn --translator best --format vtt --output-dir subs

Run  python cli.py --help  to see every option. The exit code is 0 if every file worked, 1 if any
failed, 2 if nothing could start, and 130 if interrupted with Ctrl+C.
"""
import argparse
import os
import sys
import time

import psutil

import engine
import hardware
import languages
import progress
import speakers
import translation
import video

LANGUAGE_CODES = sorted(languages.NAMES)


def find_media(inputs, recursive=False):
    """The media files in a list of files and folders, plus any inputs that don't exist."""
    files, missing = [], []
    for item in inputs:
        if os.path.isdir(item):
            walk = os.walk(item) if recursive else [(item, [], os.listdir(item))]
            for folder, _, names in walk:
                files += [os.path.join(folder, name) for name in sorted(names)
                          if name.lower().endswith(engine.SUPPORTED_MEDIA)]
        elif os.path.isfile(item):
            files.append(item)
        else:
            missing.append(item)
    return files, missing


def output_path(media_path, output_format, output_dir=None):
    """Where a file's subtitles go: <name>_subtitle<.ext>, next to it or in output_dir."""
    folder = output_dir or os.path.dirname(os.path.abspath(media_path))
    return os.path.join(folder, f"{os.path.splitext(os.path.basename(media_path))[0]}_subtitle{output_format}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="cli.py", description="Make subtitles for video and audio files, locally, with faster-whisper.",
        epilog="Folders are searched for media files (add --recursive to include subfolders).")
    parser.add_argument("inputs", nargs="+", help="media files and/or folders")
    parser.add_argument("--task", choices=["translate", "transcribe"], default="translate",
                        help="translate: English subtitles (default); transcribe: subtitles in the spoken language")
    parser.add_argument("--also-english", action="store_true",
                        help="subtitles in the spoken language plus an English file (<name>.en.srt)")
    parser.add_argument("--translate-to", choices=LANGUAGE_CODES, metavar="CODE",
                        help="also save a translation into this language, e.g. fr, bn, ja (<name>.fr.srt)")
    parser.add_argument("--translator", choices=list(translation.MODELS),
                        help="translation model size for --also-english and --translate-to: fast (600 MB), "
                             "better (1.4 GB) or best (3.4 GB) (default: the best one for this computer)")
    parser.add_argument("--language", choices=["auto"] + LANGUAGE_CODES, default="auto", metavar="CODE",
                        help="the spoken language, e.g. ja (default: detect it)")
    parser.add_argument("--model", choices=list(engine.MODEL_REPOS),
                        help="AI model (default: the best one for this computer and task)")
    parser.add_argument("--format", choices=["srt", "vtt", "txt"], default="srt", help="subtitle format (default srt)")
    parser.add_argument("--output-dir", help="save subtitles here instead of next to each file")
    parser.add_argument("--recursive", action="store_true", help="include subfolders of input folders")
    parser.add_argument("--video", choices=["none", "track", "burn"], default="none",
                        help="also make a video copy: track = switchable subtitle track(s), burn = drawn into the picture")
    parser.add_argument("--burn-size", choices=list(video.BURN_SIZES), default="Medium")
    parser.add_argument("--burn-colour", choices=list(video.BURN_COLOURS), default="White")
    parser.add_argument("--burn-position", choices=list(video.BURN_POSITIONS), default="Bottom")
    parser.add_argument("--names", default="", metavar="TEXT", help='names and terms to spell correctly, e.g. "Frodo, Mordor"')
    parser.add_argument("--speakers", action="store_true", help='start subtitles with "Speaker 1:" etc. (needs pyannote)')
    parser.add_argument("--num-speakers", type=int, metavar="N", help="how many speakers, if you know")
    parser.add_argument("--cpu", action="store_true", help="don't use the GPU")
    parser.add_argument("--no-vad", action="store_true", help="don't skip silent parts")
    parser.add_argument("--no-fast-mode", action="store_true", help="transcribe one piece at a time on the GPU")
    parser.add_argument("--no-readable-lines", action="store_true", help="one subtitle per AI segment, however long")
    parser.add_argument("--quiet", action="store_true", help="only print the summary")
    return parser.parse_args(argv)


class ProgressLine:
    """Shows engine progress on one line that updates in place."""

    def __init__(self, stream, enabled=True):
        self.stream, self.enabled = stream, enabled
        self.prefix, self.last_length, self.last_time = "", 0, 0.0

    def __call__(self, current, total, text, stage=""):
        now = time.monotonic()
        if not self.enabled or (now - self.last_time < 0.2 and stage not in ("done", "load")):
            return
        self.last_time = now
        line = f"{self.prefix}{text}"[:110]
        self.stream.write("\r" + line.ljust(self.last_length))
        self.stream.flush()
        self.last_length = len(line)

    def finish(self, text):
        if self.enabled and self.last_length:
            self.stream.write("\r" + " " * self.last_length + "\r")
        self.last_length = 0
        print(text, file=self.stream)


def main(argv=None, stream=None):
    stream = stream or sys.stderr
    args = parse_args(argv)
    files, missing = find_media(args.inputs, args.recursive)
    for item in missing:
        print(f"Not found: {item}", file=stream)
    if not files:
        print("No media files to process.", file=stream)
        return 2

    output_format = "." + args.format
    task = "transcribe" if args.also_english else args.task
    compute = "cpu" if args.cpu else "auto"
    gpus = [] if args.cpu else hardware.get_nvidia_gpus()
    gpu_index, _, vram_mb = max(gpus, key=lambda gpu: gpu[2]) if gpus else (0, "", 0)
    vram_gb = round(vram_mb / 1024)
    ram_gb = round(psutil.virtual_memory().total / 1024 ** 3)
    model = args.model or hardware.recommend_models(bool(gpus), vram_gb, ram_gb)[task]
    translator = args.translator or hardware.recommend_translator(bool(gpus), vram_gb, ram_gb)
    problem = hardware.model_job_problem(model, task, args.language)
    if problem:
        print(f"Warning: {problem}", file=stream)
    if args.video != "none" and args.format != "txt" and not video.ffmpeg_available():
        print("--video needs FFmpeg, which isn't installed or isn't on your PATH.", file=stream)
        return 2
    if args.speakers and not speakers.available():
        print(f"--speakers needs extra packages: {speakers.INSTALL_COMMAND}", file=stream)
        return 2
    if args.speakers and not speakers.has_token():
        print(f"--speakers needs a Hugging Face access token. Accept the terms at {speakers.TERMS_URL}, create a "
              f"token at {speakers.TOKEN_URL}, then run:  hf auth login", file=stream)
        return 2

    job = dict(
        model_size=model, source_language=args.language, task=task, compute_device=compute,
        device_index=gpu_index, vad_filter=not args.no_vad, output_format=output_format,
        readable_lines=not args.no_readable_lines, names_and_terms=args.names, video_output=args.video,
        batch_size=0 if args.no_fast_mode or args.cpu else hardware.batch_size_for(vram_gb),
        also_english=args.also_english, translate_to=args.translate_to, translator=translator,
        speaker_labels=args.speakers, num_speakers=args.num_speakers,
        burn_style={"size": args.burn_size, "colour": args.burn_colour, "position": args.burn_position},
    )
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    line = ProgressLine(stream, enabled=not args.quiet)
    translating = bool(args.also_english or args.translate_to)
    models = f"model '{model}'" + (f" and the '{translator}' translation model" if translating else "")
    print(f"{len(files)} file(s), {models} on {'CPU' if args.cpu or not gpus else 'GPU'}", file=stream)
    succeeded, failed = 0, 0
    try:
        engine.prepare(model, compute, gpu_index, translate=translating, translator=translator,
                       speaker_labels=args.speakers, progress_callback=line)
        for i, media in enumerate(files):
            name = os.path.basename(media)
            line.prefix = f"[{i + 1}/{len(files)}] {name}: " if len(files) > 1 else f"{name}: "
            started = time.monotonic()
            try:
                result = engine.process_file(media, output_path(media, output_format, args.output_dir),
                                             progress_callback=line, **job)
            except Exception as e:
                failed += 1
                line.finish(f"FAILED  {name}: {str(e) or type(e).__name__}")
                continue
            succeeded += 1
            made = [result["subtitle_file"]] + result["extra_files"] + ([result["video_file"]] if result["video_file"] else [])
            details = progress.format_duration(time.monotonic() - started)
            if result["language_detected"]:
                details += ", " + languages.describe(result["language"], result["language_probability"])
            line.finish(f"OK      {name} ({details}) -> " + ", ".join(os.path.basename(p) for p in made))
            for note in result["notes"]:
                print(f"        note: {note}", file=stream)
    except KeyboardInterrupt:
        video.stop_all()
        line.finish("Interrupted.")
        return 130
    except Exception as e:  # the models couldn't be downloaded or loaded
        line.finish(f"Couldn't start: {e}")
        return 2
    print(f"Done: {succeeded} succeeded, {failed} failed.", file=stream)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
