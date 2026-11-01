import psutil
import subprocess
import platform

# OS Detection
IS_WINDOWS = platform.system() == "Windows"
CREATE_NO_WINDOW = 0x08000000 if IS_WINDOWS else 0

# Models that can't do every job (codes match the start of each label in GUI.MODELS)
NO_TRANSLATION = {"large-v3-turbo", "distil-large-v3"}  # can't translate to English
ENGLISH_ONLY = {"distil-large-v3"}  # only understands English audio

def get_clean_cpu_name():
    if IS_WINDOWS:
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            return name.replace("(R)", "").replace("(TM)", "").replace(" CPU", "").strip()
        except Exception:
            return platform.processor() or "Unknown CPU"
    else:
        try:
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if 'model name' in line:
                        return line.split(':')[1].strip()
        except Exception:
            pass
        return platform.processor() or "Unknown CPU"

def get_advanced_ram_info(fallback_gb):
    if IS_WINDOWS:
        try:
            ps_cmd = (
                "$sticks = @(Get-CimInstance Win32_PhysicalMemory); "
                "$slots = (Get-CimInstance Win32_PhysicalMemoryArray | Measure-Object -Property MemoryDevices -Sum).Sum; "
                "$speed = $sticks[0].Speed; "
                "$form = if ($sticks[0].FormFactor -eq 12) { 'SODIMM' } else { 'DIMM' }; "
                "Write-Output \"$($sticks.Count),$slots,$speed,$form\""
            )
            output = subprocess.check_output(['powershell', '-NoProfile', '-Command', ps_cmd], encoding='utf-8',
                                             creationflags=CREATE_NO_WINDOW, timeout=20).strip()
            if output:
                stick_count, total_slots, speed, form_factor = output.split(',')
                return f"{fallback_gb} GB ({stick_count}/{total_slots} Slots Filled | {form_factor} @ {speed} MT/s)"
        except Exception:
            pass
    return f"{fallback_gb} GB"

def model_job_problem(model_code, task, lang_code):
    """Returns why a model can't do this job, or None if it can."""
    if task == "translate" and lang_code != "en" and model_code in NO_TRANSLATION:
        if model_code in ENGLISH_ONLY:
            return f"'{model_code}' only understands English audio, so it can't translate."
        return f"'{model_code}' wasn't trained to translate, so it returns the original language."
    if model_code in ENGLISH_ONLY and lang_code not in ("en", "auto"):
        return f"'{model_code}' only understands English audio."
    return None


def recommend_models(has_gpu, vram_gb, ram_gb):
    """Picks the best model this PC can run for each job: {"transcribe": code, "translate": code}."""
    if has_gpu:
        if vram_gb >= 8:
            return {"transcribe": "large-v3-turbo", "translate": "large-v3"}
        if vram_gb >= 6:
            # turbo is about the same size as distil-large-v3 (the old 6GB pick) but handles every language
            return {"transcribe": "large-v3-turbo", "translate": "medium"}
        if vram_gb >= 4:
            return {"transcribe": "small", "translate": "small"}
        return {"transcribe": "base", "translate": "base"}
    if ram_gb >= 16:
        return {"transcribe": "base", "translate": "base"}
    return {"transcribe": "tiny", "translate": "tiny"}


def batch_size_for(vram_gb):
    """How many 30-second pieces Fast Mode transcribes at once (0 = Fast Mode off).

    Bigger batches are faster but use more VRAM. If a batch still runs out of memory,
    engine.transcribe halves it and tries again.
    """
    if vram_gb >= 12:
        return 16
    if vram_gb >= 6:
        return 8
    if vram_gb >= 4:
        return 4
    if vram_gb >= 2:
        return 2
    return 0


def get_nvidia_gpus():
    """Returns [(index, name, vram_mb), ...] for every NVIDIA GPU, or [] if there are none."""
    kwargs = {'encoding': 'utf-8', 'timeout': 15}  # a stuck driver shouldn't leave the scan hanging
    if IS_WINDOWS: kwargs['creationflags'] = CREATE_NO_WINDOW
    try:
        output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,name,memory.total', '--format=csv,noheader,nounits'], **kwargs)
    except Exception:
        return []

    gpus = []
    for line in output.strip().splitlines():  # nvidia-smi prints one line per GPU
        try:
            index, rest = line.split(',', 1)
            name, vram_mb = rest.rsplit(',', 1)  # rsplit in case a GPU name ever contains a comma
            gpus.append((int(index), name.strip(), int(float(vram_mb))))
        except ValueError:
            continue  # skip a line we can't read instead of losing every GPU
    return gpus


def scan_system():
    """Returns {"report": text, "recommended": {task: model_code}, "gpu_index": int, "vram_gb": int}."""
    cpu_name = get_clean_cpu_name()
    cores = psutil.cpu_count(logical=False) or "Unknown"
    threads = psutil.cpu_count(logical=True) or "Unknown"
    cpu_display = f"{cpu_name} ({cores} Cores, {threads} Threads)"

    ram_bytes = psutil.virtual_memory().total
    ram_gb = round(ram_bytes / (1024 ** 3))
    ram_display = get_advanced_ram_info(ram_gb)

    gpus = get_nvidia_gpus()
    if gpus:
        # Use the GPU with the most VRAM (main.py makes CUDA number GPUs the same way nvidia-smi does)
        gpu_index, gpu_name, vram_mb = max(gpus, key=lambda gpu: gpu[2])
        vram_gb = round(vram_mb / 1024)
        gpu_display = f"{gpu_name} ({vram_gb} GB VRAM)"
        if len(gpus) > 1:
            gpu_display += " - used for processing"
            for index, name, mb in gpus:
                if index != gpu_index:
                    gpu_display += f"\nGPU: {name} ({round(mb / 1024)} GB VRAM)"
    else:
        gpu_index, vram_gb = 0, 0
        gpu_display = "No NVIDIA GPU detected (0 GB VRAM)"

    recommended = recommend_models(bool(gpus), vram_gb, ram_gb)
    if recommended["translate"] == recommended["transcribe"]:
        rec_line = f"Recommendation: Auto-selected '{recommended['translate']}' based on your system."
    else:
        rec_line = (f"Recommendation: '{recommended['translate']}' for Translate to English, "
                    f"'{recommended['transcribe']}' for Keep Original Language.")

    report_text = (
        f"--- System Hardware Scan ---\n"
        f"CPU: {cpu_display}\n"
        f"RAM: {ram_display}\n"
        f"GPU: {gpu_display}\n\n"
        f"{rec_line}"
    )
    return {"report": report_text, "recommended": recommended, "gpu_index": gpu_index, "vram_gb": vram_gb}
