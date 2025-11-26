@echo off
REM Quick run script for Windows
REM This script helps run NeMo SSL training on Windows

echo ================================================================================
echo NeMo SSL Training - Windows Quick Run
echo ================================================================================
echo.

REM Check if running from nest_ssl_project directory
if not exist "train.py" (
    echo ERROR: Please run this script from nest_ssl_project directory
    pause
    exit /b 1
)

REM Check for GPU
python -c "import torch; gpu_available = torch.cuda.is_available(); print('GPU Available:', gpu_available); exit(0 if gpu_available else 1)" 2>nul
if %errorlevel% == 0 (
    set ACCELERATOR=gpu
    echo Using GPU
) else (
    set ACCELERATOR=cpu
    echo Using CPU
)

echo.
echo Configuration:
echo   Accelerator: %ACCELERATOR%
echo   Devices: 1
echo   Strategy: auto
echo.

REM Check if manifest files exist
if not exist "data\dummy_ssl\train_manifest.json" (
    echo WARNING: Dummy data not found. Generating...
    python tools\prepare_dummy_ssl_data.py
)

echo.
echo Starting training...
echo.

python train.py ^
    trainer.devices=1 ^
    trainer.accelerator=%ACCELERATOR% ^
    trainer.strategy=auto ^
    trainer.max_steps=10 ^
    trainer.val_check_interval=5 ^
    model.train_ds.manifest_filepath=data/dummy_ssl/train_manifest.json ^
    model.validation_ds.manifest_filepath=data/dummy_ssl/val_manifest.json ^
    model.train_ds.batch_size=2 ^
    model.validation_ds.batch_size=2 ^
    model.train_ds.num_workers=0 ^
    model.validation_ds.num_workers=0

if %errorlevel% == 0 (
    echo.
    echo ================================================================================
    echo Training completed successfully!
    echo ================================================================================
) else (
    echo.
    echo ================================================================================
    echo Training failed. Check the error messages above.
    echo ================================================================================
)

pause

