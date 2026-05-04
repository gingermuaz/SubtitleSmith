import psutil
import subprocess
import platform

# Windows constant to hide the command prompt from popping up
CREATE_NO_WINDOW = 0x08000000


def get_clean_cpu_name():
    """Attempts to get a clean CPU model name from the Windows Registry."""
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
        return name.replace("(R)", "").replace("(TM)", "").replace(" CPU", "").strip()
    except Exception:
        return platform.processor() or "Unknown CPU"


def get_advanced_ram_info(fallback_gb):
    """Uses a hidden PowerShell query to grab physical RAM details from the motherboard."""
    try:
        # PowerShell command to get Sticks, Slots, Speed, and Form Factor
        ps_cmd = (
            "$sticks = @(Get-CimInstance Win32_PhysicalMemory); "
            "$slots = (Get-CimInstance Win32_PhysicalMemoryArray | Measure-Object -Property MemoryDevices -Sum).Sum; "
            "$speed = $sticks[0].Speed; "
            "$form = if ($sticks[0].FormFactor -eq 12) { 'SODIMM' } else { 'DIMM' }; "
            "Write-Output \"$($sticks.Count),$slots,$speed,$form\""
        )

        output = subprocess.check_output(
            ['powershell', '-NoProfile', '-Command', ps_cmd],
            encoding='utf-8',
            creationflags=CREATE_NO_WINDOW
        ).strip()

        if output:
            stick_count, total_slots, speed, form_factor = output.split(',')
            # Formats to: 16 GB (2/4 Slots Filled | DIMM @ 3200 MT/s)
            return f"{fallback_gb} GB ({stick_count}/{total_slots} Slots Filled | {form_factor} @ {speed} MT/s)"
    except Exception:
        pass  # If Windows blocks the PowerShell query, it silently fails

    # Fallback to standard readout if the deep scan fails
    return f"{fallback_gb} GB"


def scan_system():
    """Scans CPU, RAM, and GPU, returning a formatted report string and a model recommendation."""
    # 1. Get CPU Info
    cpu_name = get_clean_cpu_name()
    cores = psutil.cpu_count(logical=False) or "Unknown"
    threads = psutil.cpu_count(logical=True) or "Unknown"
    cpu_display = f"{cpu_name} ({cores} Cores, {threads} Threads)"

    # 2. Get RAM Info
    ram_bytes = psutil.virtual_memory().total
    ram_gb = round(ram_bytes / (1024 ** 3))
    ram_display = get_advanced_ram_info(ram_gb)

    # 3. Get NVIDIA GPU
    gpu_name = "No NVIDIA GPU detected"
    vram_gb = 0
    has_gpu = False

    try:
        smi_output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
            encoding='utf-8',
            creationflags=CREATE_NO_WINDOW
        )
        if smi_output.strip():
            parts = smi_output.strip().split(', ')
            gpu_name = parts[0]
            vram_mb = int(parts[1].replace(' MiB', '').replace(' MB', ''))
            vram_gb = round(vram_mb / 1024)
            has_gpu = True
    except Exception:
        pass

        # 4. Recommendation Logic
    rec_model = "tiny (Fastest / Low Accuracy)"

    if has_gpu:
        if vram_gb >= 8:
            rec_model = "large-v3 (Slowest / Best Accuracy)"
        elif vram_gb >= 4:
            rec_model = "small (Balanced)"
        else:
            rec_model = "base (Fast / Basic Accuracy)"
    elif ram_gb >= 16:
        rec_model = "base (Fast / Basic Accuracy)"

    # 5. Format the Text Box Output
    report_text = (
        f"--- System Hardware Scan ---\n"
        f"CPU: {cpu_display}\n"
        f"RAM: {ram_display}\n"
        f"GPU: {gpu_name} ({vram_gb} GB VRAM)\n\n"
        f"Recommendation: Auto-selected '{rec_model.split(' ')[0]}' based on your system."
    )

    return report_text, rec_model