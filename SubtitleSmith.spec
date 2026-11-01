# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs

datas = [('SubtitleSmith.ico', '.')]  # window icon, loaded at runtime by GUI.py
binaries = []
hiddenimports = []
tmp_ret = collect_all('customtkinter')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('tkinterdnd2')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('faster_whisper')  # import name, not 'faster-whisper'; brings the Silero VAD model
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('ctranslate2')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# cuBLAS + cuDNN for the GPU, from the nvidia-cublas-cu12 / nvidia-cudnn-cu12 pip packages.
# They land in _internal/nvidia/<lib>/bin, which main.py adds to the DLL search path (~1.7 GB).
binaries += collect_dynamic_libs('nvidia.cublas')
binaries += collect_dynamic_libs('nvidia.cudnn')


a = Analysis(
    ['GUI.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SubtitleSmith',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # see COLLECT below
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['SubtitleSmith.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,  # UPX can break the CUDA DLLs and makes builds very slow
    upx_exclude=[],
    name='SubtitleSmith',
)
