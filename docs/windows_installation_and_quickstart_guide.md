# Windows Installation & Beginner Quickstart Guide

Welcome to **ACI-Target-Scorer** on Microsoft Windows 10 and Windows 11!

This guide is designed specifically for competition shooters, match directors, range safety officers, and computer vision enthusiasts. Whether you have never used the Windows command prompt or PowerShell before, this step-by-step tutorial will take you from a fresh Windows machine to scoring NRA Precision Pistol targets and using the virtual **Eagle Eye Scoring Gauge** in minutes.

---

## Table of Contents
1. [Prerequisites & System Overview](#1-prerequisites--system-overview)
2. [Part 1: Installing CPython from the Microsoft Store](#2-part-1-installing-cpython-from-the-microsoft-store)
3. [Part 2: Installing `uv` (Fast Python Package & Environment Manager)](#3-part-2-installing-uv-fast-python-package--environment-manager)
4. [Part 3: Installing `ACI-Target-Scorer`](#4-part-3-installing-aci-target-scorer)
   * [Pathway A: Zero-Setup 1-Click Global App Install (Novices / Scorers)](#pathway-a-zero-setup-1-click-global-app-install-easiest)
   * [Pathway B: Full Development & Training Setup (Cloning & Retraining)](#pathway-b-full-development--training-setup-power-users)
5. [Part 4: Common Beginner Troubleshooting & FAQ](#5-part-4-common-beginner-troubleshooting--faq)
6. [Part 5: Quickstart Command Reference for Windows](#6-part-5-quickstart-command-reference-for-windows)

---

## 1. Prerequisites & System Overview

ACI-Target-Scorer runs natively on Microsoft Windows. To run the software, your PC needs:
* **Operating System:** Windows 10 (version 1909 or higher) or Windows 11 (64-bit).
* **Terminal:** Windows PowerShell (built into all modern Windows installations) or Windows Terminal.
* **Internet Connection:** To download Python, dependencies, and pre-trained weights.
* **Optional Hardware Acceleration:** Any NVIDIA GeForce or RTX GPU (e.g., GTX 1060, RTX 3060, RTX 4070+) for ultra-fast local retraining. (Standard CPU execution works out of the box for scoring and GUI inspection).

---

## 2. Part 1: Installing CPython from the Microsoft Store

The Microsoft Store provides the cleanest, safest, and most reliable method to install Python on Windows. It automatically configures system paths, integrates with Windows Terminal, and receives security updates without manual PATH tampering.

### Graphical Step-by-Step Instructions

1. **Open the Start Menu:** Click the Windows Start icon in your taskbar or press the `Windows Key` on your keyboard.
2. **Launch the Microsoft Store:** In the search bar, type `Microsoft Store` and click the application icon.
3. **Search for Python 3.11:** In the top search bar of the Microsoft Store, type:
   ```text
   Python 3.11
   ```
4. **Select the Official Package:** Look for the item named **Python 3.11** with the publisher listed as **Python Software Foundation** (free, official, and certified by Microsoft).
   > [!IMPORTANT]
   > Ensure the publisher is **Python Software Foundation**. Avoid third-party repackaged versions.
5. **Install:** Click the blue **Get** or **Install** button. Windows will download and configure CPython 3.11 automatically.

### Why Python 3.11 is Recommended

While [`pyproject.toml`](file:///home/antho/target_scorer/pyproject.toml) supports Python **3.9 through 3.12** (`requires-python = ">=3.9"`), we recommend **Python 3.11** on Windows:
* **PyTorch Official Binary Compatibility:** PyTorch (`torch`) and computer vision libraries publish pre-compiled C++/CUDA binary wheels built and rigorously tested for Python 3.11.
* **Zero C++ Compiler Headaches:** Bleeding-edge Python releases frequently lack pre-compiled binary wheels for Windows, causing standard package managers to attempt building native C++ libraries from source—which fails unless Visual Studio C++ Build Tools are installed.
* **Hardware Acceleration Stability:** NVIDIA CUDA 11.8 and 12.1 runtime libraries have mature, zero-friction support on Python 3.11 for Windows.

### Alternative CLI Option: Windows Package Manager (`winget`)

If you prefer installing via the command line, open PowerShell and run:
```powershell
winget install Python.Python.3.11
```

### Verification in PowerShell

To confirm Python is installed and accessible:
1. Press `Win + X` and select **Terminal** or **Windows PowerShell**.
2. Run the following command:
   ```powershell
   python --version
   ```
3. You should see an output confirming Python 3.11:
   ```text
   Python 3.11.x
   ```

---

## 3. Part 2: Installing `uv` (Fast Python Package & Environment Manager)

[`uv`](https://github.com/astral-sh/uv) is an ultra-fast Python package and environment manager written in Rust by Astral. 

### Why We Use `uv`
* **Blazing Speed:** Installs dependencies 10× to 100× faster than standard `pip`.
* **Zero System Clutter:** Isolates application packages so your primary Windows system Python stays clean and conflict-free.
* **Automatic Virtual Environments:** Manages virtual environments seamlessly without requiring deep knowledge of Python internal path configurations.

### 1-Line PowerShell Installation

Open PowerShell and paste the official installer command:
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

*(Alternative via Windows Package Manager: `winget install astral-sh.uv`)*

### Ensuring `uv` is in Your Windows PATH

The installer places `uv` into your user profile folder (`%USERPROFILE%\.local\bin` or `%USERPROFILE%\.cargo\bin`). To ensure all PowerShell sessions can find `uv`:

1. In PowerShell, run:
   ```powershell
   uv tool update-shell
   ```
2. **Restart PowerShell:** Close your current PowerShell window and open a fresh one so the updated Windows environment variables take effect.
3. **Verify Installation:**
   ```powershell
   uv --version
   ```
   You should see:
   ```text
   uv 0.4.x (or newer)
   ```

---

## 4. Part 3: Installing `ACI-Target-Scorer`

We offer two installation pathways. Choose the one that matches your goals:
* **Pathway A:** For users who just want to score targets, open photos, and use the graphical editor without touching code.
* **Pathway B:** For power users, developers, and researchers who want to clone the repository, run tests, or retrain models on an NVIDIA GPU.

---

### Pathway A: Zero-Setup 1-Click Global App Install (Easiest)

This pathway installs `target-scorer` as an isolated Windows application using `uv tool`. You do not need to download zip files or manage Git folders.

1. Open PowerShell and run:
   ```powershell
   uv tool install "git+https://github.com/aiannetti/ACI-Target-Scorer.git"
   ```
2. Once installed, verify the tool is globally available:
   ```powershell
   target-scorer --help
   ```
3. Launch the Graphical Annotation Editor & Eagle Eye Scoring Gauge:
   ```powershell
   target-scorer annotate
   ```
4. **Upgrading in the future:** Whenever new versions are released, update with a single command:
   ```powershell
   uv tool upgrade target-scorer
   ```

---

### Pathway B: Full Development & Training Setup (Power Users)

Follow this pathway to clone the source code, inspect models, run the 123-test automated test suite across 10 modules, or perform GPU-accelerated neural training.

#### Step 1: Navigate to Documents and Clone Repository
```powershell
cd ~\Documents
git clone https://github.com/aiannetti/ACI-Target-Scorer.git
cd ACI-Target-Scorer
```
*(If you do not have Git installed, you can install it via `winget install Git.Git` or download the ZIP from GitHub).*

#### Step 2: Create an Isolated Python 3.11 Virtual Environment
```powershell
uv venv target_env --python 3.11
```

#### Step 3: Configure PowerShell Script Execution Policy
By default, Windows restricts running PowerShell scripts (such as environment activation scripts) to prevent unauthorized execution. Run this command to allow local scripts signed or created on your machine:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
Press `Y` (Yes) when prompted.

#### Step 4: Activate the Virtual Environment
```powershell
.\target_env\Scripts\Activate.ps1
```
Your PowerShell prompt will now display `(target_env)` at the beginning of the line:
```powershell
(target_env) PS C:\Users\<Username>\Documents\ACI-Target-Scorer>
```

#### Step 5: (Optional) Install NVIDIA CUDA Hardware Acceleration
If your PC has an NVIDIA GPU (GeForce RTX/GTX), install CUDA 12.1-enabled PyTorch:
```powershell
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
*(If you do not have an NVIDIA GPU, skip this step. Standard CPU PyTorch will be installed in the next step automatically).*

#### Step 6: Install ACI-Target-Scorer in Editable Mode
```powershell
uv pip install -e .
```
Optional dependency extras in [`pyproject.toml`](file:///home/antho/target_scorer/pyproject.toml):
* `uv pip install -e ".[dev]"` — Adds `pytest`, `ruff`, `coverage`, and `matplotlib`.
* `uv pip install -e ".[colab]"` — Adds `google-colab-cli` (automatically selected on Python 3.12+; on Python 3.11, install via `uv tool install google-colab-cli`).
* `uv pip install -e ".[all]"` — Installs both `[colab]` and `[dev]` optional tools.

#### Step 7: Verify CUDA Acceleration
Run this quick Python command:
```powershell
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```
* If you have an NVIDIA GPU, it outputs: `CUDA Available: True | Device: NVIDIA GeForce RTX ...`
* If on a standard CPU laptop, it outputs: `CUDA Available: False | Device: CPU`

---

## 5. Part 4: Common Beginner Troubleshooting & FAQ

### Issue 1: `Activate.ps1 cannot be loaded because running scripts is disabled on this system`
* **Cause:** Windows PowerShell disables external script execution by default.
* **Fix:** Open PowerShell and run:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```
  Then reactivate your environment:
  ```powershell
  .\target_env\Scripts\Activate.ps1
  ```

### Issue 2: `No virtual environment found` when running `uv pip install`
* **Cause:** Unlike legacy `pip`, `uv pip` refuses to install packages globally into your Windows system directory to protect system health.
* **Fix:** Make sure your virtual environment is active (your prompt should show `(target_env)`). If not, activate it:
  ```powershell
  .\target_env\Scripts\Activate.ps1
  ```
  Alternatively, specify the target environment explicitly:
  ```powershell
  uv pip install --python .\target_env <package>
  ```

### Issue 3: `The term 'target-scorer' is not recognized as the name of a cmdlet`
* **Cause:** PowerShell cannot find the `target-scorer.exe` binary in your current PATH.
* **Fix:**
  1. If using **Pathway A (`uv tool install`)**: Run `uv tool update-shell`, restart PowerShell, and verify with:
     ```powershell
     Get-Command target-scorer
     ```
  2. If using **Pathway B (Development Clone)**: Ensure you have activated your virtual environment:
     ```powershell
     .\target_env\Scripts\Activate.ps1
     ```
     Once activated, `target-scorer` is directly available.

### Issue 4: Running the Automated Test Battery
To ensure everything is working properly on your Windows machine, run the complete 123-test suite (across 10 test modules):
```powershell
python -m unittest discover tests -v
```
All 123 tests should pass with an `OK` summary.

---

## 6. Part 5: Quickstart Command Reference for Windows

Once installed, use these copy-pasteable commands in PowerShell:

### 1. Launch the Desktop Annotation Editor & Eagle Eye Loupe
```powershell
target-scorer annotate
```
Or open a specific target directory (of JSON annotations or raw range photos) with in-editor auto-labeling settings:
```powershell
target-scorer annotate datasets\calibration_targets `
  --checkpoint weights\target_scoring_hrnet_1024_best.pt `
  --threshold 0.40
```

### 2. Score a Target Photograph (iPhone HEIC, Android JPG, or Camera RAW)
```powershell
target-scorer infer `
  --image "C:\path\to\target_photo.heic" `
  --target B-3 `
  --canvas-size 1024 `
  --scale 4.0 `
  --output "C:\path\to\scorecard.png"
```
*(Note: in PowerShell, the backtick `` ` `` is the line continuation character, equivalent to `\` in Linux/macOS bash).*

### 3. Rectify a Skewed Range Photo into a Flat Canonical Card
```powershell
target-scorer rectify `
  --image "C:\path\to\tilted_shot.jpg" `
  --target B-2 `
  --canvas-size 1024 `
  --output "C:\path\to\rectified.png"
```

### 4. Auto-Label a Directory with Competitor PII Redaction
```powershell
target-scorer autolabel `
  --input-dir "C:\path\to\range_snaps" `
  --output-dir "C:\path\to\annotated_dataset" `
  --target B-3 `
  --threshold 0.40 `
  --redact-pii
```

---

## Related Documentation

* 📖 **[CLI & Python API Reference](file:///home/antho/target_scorer/docs/cli_and_api_reference.md):** Complete command reference with flags and options.
* 🔍 **[Interactive Annotation Editor Guide](file:///home/antho/target_scorer/docs/interactive_annotation_editor_guide.md):** Detailed guide to the Sun Valley desktop metrology workstation and Eagle Eye Loupe.
* 🎯 **[End-to-End Workflow Tutorial](file:///home/antho/target_scorer/docs/end_to_end_training_and_workflow_tutorial.md):** Operational workflows from range photography to active learning.
* 🏗️ **[Technical Architecture & Developer Guide](file:///home/antho/target_scorer/docs/architecture_and_developer_guide.md):** In-depth mathematical formulations and deep learning engine specs.
