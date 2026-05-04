import psutil
import subprocess
import platform
import os

# OS Detection
IS_WINDOWS = platform.system() == "Windows"

# Windows constant to hide the command prompt from popping up
CREATE_NO_WINDOW = 0x08000000 if IS_WINDOWS else 0


def get_clean_cpu_name():
    """Attempts to get a clean CPU model name dynamically based on the OS."""
    if IS_WINDOWS:
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            return name.replace("(R)", "").replace("(TM)", "").replace(" CPU", "").strip()
        except Exception:
            return platform.processor() or "Unknown CPU"
    else:
        # Linux Fallback
        try:
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if 'model name' in line:
                        return line.split(':')[1].strip()
        except Exception:
            pass
        return platform.processor() or "Unknown CPU"


def get_advanced_ram_info(fallback_gb):
    """Uses OS-specific commands to grab physical RAM details."""
    if IS_WINDOWS:
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
                return f"{fallback_gb} GB ({stick_count}/{total_slots} Slots Filled | {form_factor} @ {speed} MT/s)"
        except Exception:
            pass

            # Fallback for Linux (Deep RAM inspection on Linux requires sudo/root access, so we stick to the safe total amount)
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
        # Build the arguments dynamically so we don't pass Windows flags to Linux
        subprocess_kwargs = {'encoding': 'utf-8'}
        if IS_WINDOWS:
            subprocess_kwargs['creationflags'] = CREATE_NO_WINDOW

        smi_output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
            **subprocess_kwargs
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