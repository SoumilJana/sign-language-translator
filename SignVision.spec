# SignVision.spec

import os
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.building.build_main import Analysis, PYZ, EXE, COLLECT

# collect mediapipe data
mediapipe_datas = collect_data_files('mediapipe')

# required data files
datas = [
    ('app/ui.kv', '.'),
    ('app/assets', 'app/assets'),
] + mediapipe_datas

# hidden imports for used modules
hiddenimports = [
    'mediapipe',
    'mediapipe.python',
    'mediapipe.python.solutions.hands',
    'mediapipe.python.solutions.drawing_utils',
    'tensorflow',
    'cv2',
    'numpy',
    'pyttsx3',
    'kivy',
]

block_cipher = None

a = Analysis(
    ['app/main.py'],
    pathex=[os.getcwd()],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SignVision',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name='SignVision'
)
