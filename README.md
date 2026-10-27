# Mouse Sleep Analyzer

Windows desktop application and command-line tool for estimating a single mouse's
sleep and awake intervals from overhead FLIR optical or near-infrared MP4 video.
It includes cage selection, segmentation previews, resumable analysis, and
sleep/awake/unknown reports. Two separate cages can be analyzed independently;
optional tools align Seek thermal measurements with each mouse's sleep results.

This is a modern Python implementation based on
[KumarLabJax/MouseTracking](https://github.com/KumarLabJax/MouseTracking) and
[KumarLabJax/MouseSleep](https://github.com/KumarLabJax/MouseSleep).
No connected camera, PySpin, or camera-recording software is needed to analyze
existing compatible recordings.

## Install

1. Install **64-bit Python 3.12** with the Windows Python launcher.
2. Download or clone this repository into a writable folder and run `Setup.cmd`.
3. Follow [model installation](docs/INSTALLATION.md#segmentation-models) to
   download the official checkpoints and place them in `pretrained_models/`.
4. Open `Start_MouseSleep.cmd`. Under **Classifier setup**, click
   **Download / train classifier**, or select the official feature CSV locally.
   The first download is about 950 MB. Subsequent analyses can run offline.

**The repository contains source code, documentation, and tests.** Recordings,
model weights, the trained classifier, training data, and generated results are
obtained or created locally and are excluded from Git.

See [installation and troubleshooting](docs/INSTALLATION.md) for the exact model
layout and setup checks. Dependency versions are pinned in
`requirements-lock.txt`; `requirements.txt` describes the supported ranges.

## First analysis

1. Open `Start_MouseSleep.cmd` and select an original MP4 and a new output folder.
2. Select mouse appearance and draw a rectangle around one cage.
3. For a dark mouse, use Enhanced masking and draw its cage boundary. For a
   white/light mouse, use Legacy masking. Check **Test segmentation** at several
   times, including changes in illumination.
4. Run a 60-300 second pilot using **Advanced settings > maximum seconds**.
5. Review the masks and results, then use a new output folder for the full run.

Use a fixed overhead view at approximately 30 fps or higher. There must be only
one mouse in each selected cage region. Keep acquisition timing sidecars beside
the original video when available.

```powershell
.\MouseSleep.cmd analyze --help
.\MouseSleep.cmd analyze --video "C:\Recordings\mouse.mp4" --output "C:\Results\mouse01" --appearance dark --max-seconds 120
```

The results include `epochs.csv`, `sleep_bouts.csv`, `hourly_sleep.csv`,
`timeline.png`, summary/settings JSON, and a database for resuming analysis.
Missing or unreliable observations remain **unknown**.

## Documentation

- [User guide](docs/USER_GUIDE.md): masking, timing, outputs, commands, and resume.
- [Near-infrared recording guide](docs/NEAR_IR_SETUP.md) and
  [Enhanced preset](settings.enhanced_near_ir.json).
- [Two-cage and temperature guide](docs/TWO_CAGE_SETUP.md) and
  [example cage configuration](cages.example.json).
- [Testing and validation](docs/VALIDATION.md): synthetic tests and optional
  checks with external models.
- [Attribution and external assets](docs/ATTRIBUTION.md).

Run the automated tests after setup:

```powershell
.venv\Scripts\python.exe -m pytest -ra
```

The three checkpoint-dependent tests skip when their model files are absent;
all other tests use synthetic or mocked inputs. GitHub Actions runs the same
suite on Windows with Python 3.12.

## Project layout

| Location | Contents |
| --- | --- |
| `mouse_sleep/` | Application, GUI, sleep analysis, and thermal processing |
| `scripts/` | Reusable verification, benchmarking, and thermal-analysis commands |
| `tests/` | Automated tests and synthetic fixture generators |
| `docs/` | Setup, usage, and validation guides |
| `licenses/` | Upstream code license notices |
| Root JSON files | Portable examples and presets; choose paths and regions for your data |

Keep recordings outside the source folder, and use `outputs/` for local results.
Generated directories are ignored by Git, but manually uploading files through
a browser requires selecting only the intended source files.

## Scientific limits and license

Outputs are video-based estimates, not EEG measurements. Validate segmentation
and predictions for your camera, cage, lighting, and mice; the authors' published
accuracy is not claimed for new footage. White-mouse sleep classification is
experimental. Thermal maps are not inputs to the sleep segmentation model.

Source code is distributed under the [MIT license](LICENSE), with upstream
notices preserved in [licenses/](licenses/). External datasets and model weights
retain their own terms; see [attribution](docs/ATTRIBUTION.md).
