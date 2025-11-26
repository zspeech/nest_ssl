#!/usr/bin/env python3
"""
Minimal NeMo installation script for Windows.
Skips packages that require C++ compilation (ctc_segmentation, texterrors, megatron_core).
"""

import subprocess
import sys
import os

def run_command(cmd, check=True):
    """Run a shell command."""
    print(f"\n>>> {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if check and result.returncode != 0:
        print(f"ERROR: Command failed with exit code {result.returncode}")
        sys.exit(1)
    return result.returncode == 0

def main():
    print("="*80)
    print("Minimal NeMo Installation for Windows")
    print("="*80)
    print("\nThis script will install NeMo with minimal dependencies.")
    print("It will skip packages that require C++ compilation.")
    print()
    
    # Step 1: Install core dependencies
    print("\n[1/4] Installing core dependencies...")
    core_deps = [
        "torch",
        "torchaudio", 
        "lightning>=2.0.0",
        "hydra-core>=1.3.0",
        "omegaconf>=2.3.0",
        "einops",
        "packaging",
        "braceexpand",
        "ruamel.yaml",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "scipy>=1.9.0",
        "resampy>=0.4.0",
        "numpy>=1.22.0",
    ]
    
    for dep in core_deps:
        run_command(f"pip install {dep}", check=False)
    
    # Step 2: Check if NeMo directory exists
    print("\n[2/4] Checking NeMo installation...")
    nemo_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "NeMo")
    
    if not os.path.exists(nemo_path):
        print(f"ERROR: NeMo directory not found at {nemo_path}")
        print("Please ensure NeMo is cloned in the parent directory.")
        sys.exit(1)
    
    print(f"✓ Found NeMo at {nemo_path}")
    
    # Step 3: Install NeMo in development mode (skip optional deps)
    print("\n[3/4] Installing NeMo from source (skipping optional dependencies)...")
    os.chdir(nemo_path)
    
    # Install NeMo core dependencies
    if os.path.exists("requirements/requirements.txt"):
        run_command("pip install -r requirements/requirements.txt", check=False)
    
    if os.path.exists("requirements/requirements_lightning.txt"):
        run_command("pip install -r requirements/requirements_lightning.txt", check=False)
    
    if os.path.exists("requirements/requirements_common.txt"):
        run_command("pip install -r requirements/requirements_common.txt", check=False)
    
    # Install ASR dependencies (skip ctc_segmentation and texterrors)
    if os.path.exists("requirements/requirements_asr.txt"):
        print("\nInstalling ASR dependencies (skipping ctc_segmentation and texterrors)...")
        with open("requirements/requirements_asr.txt", "r") as f:
            deps = f.readlines()
        
        for dep in deps:
            dep = dep.strip()
            if not dep or dep.startswith("#"):
                continue
            # Skip packages that require compilation
            if "ctc_segmentation" in dep or "texterrors" in dep:
                print(f"  Skipping {dep} (requires C++ compilation)")
                continue
            run_command(f"pip install {dep}", check=False)
    
    # Install NeMo in editable mode
    print("\nInstalling NeMo in editable mode...")
    run_command("pip install -e . --no-deps", check=False)
    
    # Step 4: Verify installation
    print("\n[4/4] Verifying installation...")
    test_imports = [
        "from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder",
        "from nemo.collections.asr.losses.ssl_losses.mlm import MLMLoss",
        "from nemo.core.classes.model_pt import ModelPT",
    ]
    
    all_success = True
    for import_stmt in test_imports:
        try:
            exec(import_stmt)
            print(f"✓ {import_stmt}")
        except Exception as e:
            print(f"✗ {import_stmt}")
            print(f"  Error: {e}")
            all_success = False
    
    if all_success:
        print("\n" + "="*80)
        print("✓ NeMo minimal installation completed successfully!")
        print("="*80)
        print("\nYou can now run the comparison script:")
        print("  python tools/compare_with_nemo.py")
    else:
        print("\n" + "="*80)
        print("⚠ Some imports failed. You may need to:")
        print("1. Add NeMo to PYTHONPATH:")
        print(f"   set PYTHONPATH={nemo_path};%PYTHONPATH%")
        print("2. Or install missing dependencies manually")
        print("="*80)

if __name__ == "__main__":
    main()


