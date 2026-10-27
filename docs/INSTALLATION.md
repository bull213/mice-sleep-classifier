# Installation on Windows

Use 64-bit Python 3.12 and a writable project folder. Python 3.12 is the tested
release target; the package metadata permits Python 3.12-3.13. CPU processing is
supported. Initial setup requires internet access unless dependencies, model
files, and the reference CSV are already available locally.

## Python dependencies

Install Python with the Windows Python launcher, extract the source ZIP, then
double-click `Setup.cmd` in the project root. It creates `.venv` and installs
`requirements-lock.txt`. Do not copy a virtual environment from another computer.

The equivalent PowerShell commands, run from the project root, are:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.venv\Scripts\python.exe -m pip check
```

This installs the Python dependencies. Segmentation checkpoints and a trained
sleep classifier are separate setup steps below. Several dependencies are large;
allow multiple gigabytes of disk space for the environment and downloaded data.
The supported launchers run directly from this source checkout. Keep the source
folder in place with its local model directories.

## Segmentation models

Download **`Pretrained_Models.tar`** (about 661 MB) from the authors'
[Zenodo model record](https://zenodo.org/records/5806397). The record publishes
MD5 `695ebca1fd03d783069ab7205d9bd790` for this archive. Compare your download with
the checksum displayed on the record:

```powershell
Get-FileHash -Algorithm MD5 -LiteralPath "C:\Downloads\Pretrained_Models.tar"
```

Extract the archive. From the extracted `Pretrained_Models` tree, copy the three
model directories below into a new **`pretrained_models`** directory beside
`launch.py`. Avoid an extra nested `Pretrained_Models` directory. Copy all three
listed files for each checkpoint, preserving their names:

```text
MouseSleepAnalyzer/
  launch.py
  pretrained_models/
    24Hr_OFA/
      model.ckpt-255000.index
      model.ckpt-255000.data-00000-of-00001
      model.ckpt-255000.meta
    FullModel/
      model.ckpt-415000.index
      model.ckpt-415000.data-00000-of-00001
      model.ckpt-415000.meta
    AllCoatColor_RegularOFA/
      model.ckpt-234000.index
      model.ckpt-234000.data-00000-of-00001
      model.ckpt-234000.meta
```

The `.index` and `.data` files are needed for inference. The `.meta` files are
needed for the optional original-graph parity check. These three checkpoints
occupy about 398 MB once extracted.

| Model | Automatic selection |
| --- | --- |
| `AllCoatColor_RegularOFA` | Dark mouse, Enhanced masking |
| `24Hr_OFA` | Dark mouse, Legacy masking |
| `FullModel` | White/light mouse, Legacy masking |

Verify that all three checkpoints can load and predict:

```powershell
.venv\Scripts\python.exe -m scripts.check_models
```

The command prints one result per model and fails if files are missing or
unreadable. It tests model loading, not biological accuracy. Model descriptions
and original distribution terms belong to the linked Zenodo record.

## Sleep classifier

Open `Start_MouseSleep.cmd`, select **Classifier setup**, and click
**Download / train classifier**. This downloads the official
`df_features_10s_17_full_hu.csv` from the
[MouseSleep reference record](https://zenodo.org/records/5180680), verifies its
size and checksum, and trains the classifier. The download is about 950 MB.
Downloads resume after interruption. A local copy of this same official CSV can
be selected instead.

CLI equivalents:

```powershell
.\MouseSleep.cmd train
# Or use an existing copy of the official CSV:
.\MouseSleep.cmd train --dataset "C:\Datasets\df_features_10s_17_full_hu.csv"
```

Successful training creates `classifier/model.json`, `classifier/schema.json`,
and `classifier/validation.json`. Keep this directory and `pretrained_models/`
beside the source. Cached training data in `data/` is not needed for subsequent
predictions; retain it if you want to retrain without downloading again.

## Verify the installation

```powershell
.venv\Scripts\python.exe -m pip check
.\MouseSleep.cmd --help
.venv\Scripts\python.exe -m pytest -ra
```

Open `Start_MouseSleep.cmd`. A fresh installation opens with no recording
selected. Model downloads are not needed merely to open the GUI or inspect CLI
help. Complete both model setup steps before analyzing your own video.

Load a preset, select an original MP4 and a new output folder, then follow the
[first-analysis steps](../README.md#first-analysis). Preset paths are blank by
design. Two-cage configuration coordinates and calibration are examples that
must be adjusted for the actual recording and camera.

## Troubleshooting

| Problem | Action |
| --- | --- |
| Python launcher cannot find 3.12 | Install 64-bit Python 3.12 with the launcher, then rerun `Setup.cmd`. |
| Dependency installation fails | Read the pip error, check internet access and free disk space, and use a fresh Python 3.12 `.venv`. |
| Segmentation checkpoint is missing | Compare the directory tree above, including checkpoint numbers and the absence of an extra nested archive folder. |
| Classifier not found | Complete **Download / train classifier**, or select an existing compatible classifier directory in Advanced settings. |
| Recording path does not exist | Select your original recording after loading a preset; no experimental recordings are shipped. |
| Model tests are skipped | Install the external checkpoints; ordinary synthetic tests can run without them. |

When reporting an issue, include the error message, Python version, camera/video
properties, and relevant settings with personal paths removed. Keep recordings
and generated reports private unless you intentionally choose to share them.
