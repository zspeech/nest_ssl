@echo off
REM Quick NeMo installation script for Windows
REM Skips packages that require C++ compilation

echo ================================================================================
echo Quick NeMo Installation for Windows
echo ================================================================================
echo.
echo This script will install NeMo with minimal dependencies.
echo It will skip packages that require C++ compilation (ctc_segmentation, etc.)
echo.

REM Step 1: Install core dependencies
echo [1/4] Installing core dependencies...
pip install torch torchaudio lightning hydra-core omegaconf einops packaging braceexpand ruamel.yaml librosa soundfile scipy resampy numpy

REM Step 2: Check NeMo path
echo.
echo [2/4] Checking NeMo installation...
set NEMO_PATH=%~dp0..\NeMo
if not exist "%NEMO_PATH%" (
    echo ERROR: NeMo directory not found at %NEMO_PATH%
    echo Please ensure NeMo is cloned in the parent directory.
    pause
    exit /b 1
)
echo Found NeMo at %NEMO_PATH%

REM Step 3: Install NeMo dependencies (skip optional ones)
echo.
echo [3/4] Installing NeMo dependencies...
cd /d "%NEMO_PATH%"

if exist "requirements\requirements.txt" (
    pip install -r requirements\requirements.txt
)

if exist "requirements\requirements_lightning.txt" (
    pip install -r requirements\requirements_lightning.txt
)

if exist "requirements\requirements_common.txt" (
    pip install -r requirements\requirements_common.txt
)

REM Install ASR dependencies manually (skip ctc_segmentation and texterrors)
echo.
echo Installing ASR dependencies (skipping ctc_segmentation and texterrors)...
pip install braceexpand editdistance einops jiwer kaldi-python-io librosa marshmallow optuna packaging pyannote.core pyannote.metrics pydub pyloudnorm resampy ruamel.yaml scipy soundfile sox kaldialign whisper_normalizer diskcache

REM Install NeMo in editable mode
echo.
echo Installing NeMo in editable mode...
pip install -e . --no-deps

REM Step 4: Verify
echo.
echo [4/4] Verifying installation...
python -c "from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder; print('✓ ConformerEncoder imported successfully')" 2>nul
if errorlevel 1 (
    echo.
    echo ⚠ Import test failed. You may need to add NeMo to PYTHONPATH:
    echo    set PYTHONPATH=%NEMO_PATH%;%%PYTHONPATH%%
    echo.
    echo Or run the Python installation script:
    echo    python install_nemo_minimal.py
) else (
    echo.
    echo ================================================================================
    echo ✓ NeMo minimal installation completed successfully!
    echo ================================================================================
    echo.
    echo You can now run the comparison script:
    echo   python tools\compare_with_nemo.py
)

cd /d "%~dp0"
pause


