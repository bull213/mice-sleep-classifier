# Mouse Sleep Analyzer user guide

Standalone Windows desktop application and command-line tool for estimating a **single mouse's sleep and awake intervals from overhead FLIR MP4 video**. This project ports the segmentation inference and feature calculations from [KumarLabJax/MouseTracking](https://github.com/KumarLabJax/MouseTracking) and [KumarLabJax/MouseSleep](https://github.com/KumarLabJax/MouseSleep) to current Python and TensorFlow. It runs independently of UltramicStudio and does not require PySpin, a connected camera, EEG labels for the input video, or Linux containers.

## Start

Complete [installation](INSTALLATION.md), then double-click **`Start_MouseSleep.cmd`** in the project root. Run all commands below from that root folder.

1. Select the **original MP4** and a new output folder.
2. Choose **Dark / black** or **White / light** appearance.
3. Drag a rectangle around the cage in the original preview. Keep the complete mouse-visible floor in the region; rectangular regions are padded without stretching.
4. For black mice, fresh settings select **Enhanced** masking. Use **Draw cage boundary** to exclude reflected images while including the complete mouse-accessible area; click around it and select **Finish boundary**. Use the time slider and **Test segmentation**, including daytime and nighttime. **Compare with Legacy** shows the original detector beside Enhanced. White/light mice use Legacy.
5. Start with a 60–300 second pilot using **Advanced settings → maximum seconds**. Use a different output folder for the subsequent full recording.
6. Click **Analyze**. Review the timeline and click an interval in **Results and review** to view the original frames.

If the classifier has not been trained, open **Classifier setup** and click **Download / train classifier** once. The official feature CSV is approximately 950 MB; downloads resume and are verified against Zenodo's file size and MD5. A local copy of the same CSV can be selected instead. The reusable classifier is saved in `classifier/`; further videos work offline.

For a fresh installation, install 64-bit Python 3.12 with the Windows Python launcher and run `Setup.cmd`. Tested dependency versions are in `requirements-lock.txt`; `requirements.txt` describes supported ranges. Keep the weights and classifier beside the source code. Virtual environments should be recreated if the project is moved to another computer.

## Video preparation and camera recording

For the near-IR cage, use the [capture setup guide](NEAR_IR_SETUP.md) and [Enhanced preset](../settings.enhanced_near_ir.json). Select your video and output folder, then draw its cage boundary. The [previous near-IR preset](../settings.near_ir.json) retains Legacy masking with `AllCoatColor_RegularOFA`, no refinement, and a 300-second pilot limit. Both presets require preview checks. Check whether your recorder overrides exposure settings when it opens the camera.

**Recording at the model's 480 × 480 resolution is unnecessary.** Record at the camera's normal resolution at a genuine approximately **30 frames/s** (or higher), with stable overhead focus, a fixed camera, and sufficient visible/near-infrared illumination to see the mouse throughout the dark phase. Keep the recorder's `flir_log_camN.csv` and `takeNN_sync.json` beside the MP4. Verify the actual acquisition frame rate in your recorder; spatial conversion is performed during analysis.

The analyzer converts decoded frames to grayscale, crops the selected cage region, pads a non-square region without stretching it, and resizes to 480 × 480 for segmentation. Padded pixels are excluded from the measured mask. It uses the original pixel intensities to refine the dark-mouse mask and maps measurements to the MouseSleep reference 1080 × 1080 coordinate system. These steps run during analysis; a second 24-hour MP4 is not needed.

**Export fitted video** / the `prepare` command optionally creates `prepared_480.mp4`, `prepared_frames.csv`, and a preparation manifest. This is a segmentation preview derivative. Its sidecar records original frame indices and times. Sleep analysis must use the **original MP4**: the preview is compressed, and its fixed-rate playback can shorten acquisition gaps. Known prepared derivatives are rejected as analysis inputs.

Resizing cannot recover breathing or motion details lost to low resolution, blur, low frame rate, clipping, or compression. Videos below 29 fps are rejected rather than duplicating frames and inventing motion. Resolution conversion also cannot establish accuracy for a new camera, arena, lens, mouse size, or lighting setup. Check the mask and compare selected predicted intervals with manual observations; EEG/EMG is needed to establish sleep accuracy for research claims.

This application expects conventional optical grayscale/near-infrared FLIR images, with one mouse inside the selected cage region. For **two separate cages in one video**, analyze each cage independently using the [two-cage setup and temperature guide](TWO_CAGE_SETUP.md). It includes resolution/framing checks, a pilot protocol, and per-mouse thermal postprocessing with [an example configuration](../cages.example.json). Thermal maps are not segmentation inputs, and multiple mice inside the same selected region are not supported.

## Enhanced and Legacy masking

**Enhanced** is the default for fresh black-mouse analyses. It combines the neural mask with dark foreground candidates from fixed background references. Up to 96 sampled frames across the source take establish up to two lighting conditions, supporting light/dark recordings without assuming a switch exactly at hour 12. Background preparation uses the whole take even for a duration-limited pilot; it does not decode another full-day movie. References are selected by image appearance during processing.

References do not continuously adapt, so a resting mouse is not gradually absorbed into a changing background. When automatic references contain a persistent dark region or do not match current lighting, neural evidence can still supply the mask. Ambiguous objects, implausible jumps, and missing detections are rejected; previous masks are never carried forward. Some failures therefore appear as unknown sleep intervals. These checks do not guarantee correct detection.

Among equally supported candidates, one region can be accepted when it has at least three times the area of the next candidate, allowing small changed bedding patches to be ignored. Comparable competing regions remain ambiguous. Thin foreground structures are removed before candidate selection to reduce tail and apparatus inclusion.

The cage boundary is an optional polygon in **original video coordinates**, separate from the rectangular crop. It excludes candidate regions without stretching or changing the neural input. Include positions reached while rearing and drinking. Each cage needs its own boundary. Changing the rectangular crop clears its boundary.

Advanced settings allow up to two **empty-cage reference images**, with the same framing, dimensions, and lighting as the recording. Without these, references are estimated automatically. Enhanced contrast defaults to 12 grayscale levels and a 0.45 relative difference; mouse area defaults to 0.001–0.25 of the allowed cage area. These are adjustable starting values. Legacy's absolute pixel threshold is inactive in Enhanced. Changing bedding, reflections, unfamiliar lighting, or a mouse that never vacates a location can still cause errors.

**Legacy** retains the original model cleanup, largest-region selection, pixel refinement, and measurements. Existing settings files without a masking mode load as Legacy, including `last_settings.json` and old run manifests. Fresh settings use Enhanced; restoring saved settings preserves their mode. Select Legacy in the GUI or pass `--masking-mode legacy` to restore the original behavior. A mode change requires a **new output folder**; it cannot be applied while resuming another mode's run. White/light mice remain supported through Legacy.

**Compare with Legacy** evaluates the displayed frame with both modes. Enhanced previews process up to one preceding second to initialize tracking; they can differ from an ongoing analysis's longer tracking history. With Automatic model selection, Enhanced uses `AllCoatColor_RegularOFA` and Legacy uses `24Hr_OFA` for dark mice. Explicit model selections are retained in both modes.

Backgrounds are cached in `.cache/masking/`, keyed by source metadata, timing, geometry, reference images, and algorithm version. Each Enhanced run also saves its exact references in `masking_background.npz`. Keep that file with the analysis database for resume. Position and lighting state are committed with every epoch. Preparation and cached-load times are reported separately from frame processing.

Enhanced masking is versioned independently of settings files. Version 2 adds the candidate-area margin; analyses created with earlier Enhanced rules require a new output folder. Existing Legacy runs remain resumable.

See [testing and validation](VALIDATION.md) for reproducible checks and their limitations. The analysis tab scrolls when its controls exceed the available window height.

Example CLI usage (a boundary file contains an array such as `[[220,125],[760,125],[760,500],[220,500]]`; choose coordinates for your recording):

```powershell
.\MouseSleep.cmd analyze --config settings.json --masking-mode enhanced --cage-boundary cage_A_boundary.json --output outputs\cage_A_enhanced
.\MouseSleep.cmd analyze --config settings.json --masking-mode legacy --output outputs\cage_A_legacy
```

Optional `--background-image` accepts one reference, or can be repeated for a second. `--background-delta`, `--background-fraction`, `--minimum-mouse-area`, and `--maximum-mouse-area` match the Advanced controls. Python settings expose `masking_mode`, `cage_polygon`, `background_images`, and those four numeric parameters. No segmentation training is performed by this mode.

## Segmentation model selection

Install the following checkpoints using [the model setup guide](INSTALLATION.md#segmentation-models). Their descriptions are from the [authors' Zenodo record](https://zenodo.org/records/5806397).

| Model | Checkpoint | Use |
| --- | --- | --- |
| `24Hr_OFA` | `model.ckpt-255000` | Legacy automatic choice for dark mice: 24-hour arena, bedding, food container, visible and infrared phases; its training mice appear black. |
| `FullModel` | `model.ckpt-415000` | Automatic for white/light mice: broad training across environments and coat colors. |
| `AllCoatColor_RegularOFA` | `model.ckpt-234000` | Enhanced automatic choice for dark mice; also an alternative for white/light mice in Legacy, especially a regular open-field arena. |

For white mice, automatic refinement is **off**, retaining bright fur instead of deleting it using the original dark-pixel cutoff. No inversion is applied. `AllCoatColor_RegularOFA` can be selected manually if its preview fits the cage better. Optional bright-pixel refinement is available for a setup where a chosen threshold helps; inspect its result before analysis.

**White-mouse segmentation is supported; white-mouse sleep classification is experimental.** The reference sleep study used C57BL/6J mice, and segmentation support for white fur does not demonstrate sleep-classifier transfer. The app and reports explicitly retain this limitation. The authors' published sleep accuracy is not claimed for your footage.

## Timing, missing data, and long recordings

An acquisition log is automatically found for `flir_camN.mp4` or `flir_camN_takeNN.mp4` beside `flir_log_camN.csv`. The take number is inferred from the filename; Advanced settings permits an explicit timing CSV and take ID. The supported log columns are `take_id`, `host_received_monotonic_ns`, `frame_id`, and optional `pc_time`. Rows must correspond exactly to decoded frames in the selected take. Extra/missing rows and non-increasing times are errors. Host receipt timestamps preserve capture gaps but are not precise exposure timestamps.

Without a matching log, the application uses the MP4's presentation timestamps. If a recorder already removed capture gaps by encoding at a fixed frame rate, those lost times cannot be inferred from the MP4 alone. Time zero is the first recorded FLIR frame; its wall time and monotonic origin are recorded in `run.json` when available. A take's stop metadata can preserve an unrecorded tail as unknown time.

Frames above 30 fps are sampled into 30 Hz slots, while their original times are retained. Each complete 10-second epoch produces the 225 signal/spectral features selected in the authors' classifier code. Short tolerated timing gaps are interpolated onto a uniform grid only after checking actual observed coverage. Defaults require 95% coverage and reject gaps over 0.12 s. Missing/implausible masks, substantial device-frame gaps, low model probabilities, and an incomplete final epoch are **unknown**, never counted as sleeping. The three-class internal classifier is collapsed to sleep (NREM + REM) versus awake; temporal smoothing stops at every unknown interval.

Video frames are processed in small batches. Only timing arrays and 10-second epoch records grow with duration; the video is never loaded into RAM in full. Epochs are committed to SQLite during processing. CPU inference is supported, with live processing speed and an estimated remaining time. A 24-hour recording may require many hours of processing on a CPU.

To repeat the segmentation-only benchmark against the original implementation, save settings as `settings.json` and run `.venv\Scripts\python.exe -m scripts.benchmark_segmentation --config settings.json`. It checks exact mask equality and reports timings after warmup. Omit `--config` to use synthetic inputs. The normal test suite compares cleanup at image borders and score thresholds. Checkpoint tests run when their external files are installed; otherwise they skip with a setup explanation.

Repeated GUI segmentation previews and subsequent analyses reuse the selected model. Checkpoint contents are checked before reuse; changing weights replaces the session, and closing the application releases it. Model initialization and heavy analysis/training imports run in background tasks.

To measure the complete pipeline on your current recording, use a **new output folder**:

```powershell
.venv\Scripts\python.exe -m scripts.benchmark_analysis --config settings.json --seconds 60.5 --output verification\performance\new_pilot
```

The pilot includes normal analysis outputs and `performance.json` with elapsed time and peak process memory. Add `--compare path\to\baseline` to compare saved results and features against a run made with identical inputs and settings.

**Cancel** saves completed epochs. To continue, select the same video, settings, model, ROI, and output folder and check **Resume**, or load settings from the saved run. In-flight incomplete epochs are recalculated. Resume rejects changes in video metadata, timing log metadata, settings, and model hashes. A crash or command-line Ctrl+C also leaves committed epochs available for resumption. Progress can be reported slightly beyond the last saved epoch only while a frame batch is in flight. Classifier training cancellation waits for the current fit step; downloading can take up to its current network timeout to stop.

## Outputs

| File | Contents |
| --- | --- |
| `epochs.csv` | One row per analyzed 10-second epoch: original start/end seconds, sleep/awake/unknown, class probabilities, tracking coverage, and unknown reason. Probabilities are model scores, not calibrated accuracy guarantees. |
| `sleep_bouts.csv` | Continuous predicted sleep intervals and durations, broken at unknown/awake epochs. |
| `hourly_sleep.csv` | Sleep, awake, and unknown seconds in each hour relative to the first recorded frame. |
| `timeline.png` | Entire-recording timeline with unknown intervals visible. |
| `summary.json` | Sleep/awake/unknown totals, sleep fraction of scorable time, unprocessed duration, and applicability notice. |
| `run.json` | Settings, model hashes, source/timing metadata, completed epochs, and run status. |
| `settings.json` | Resolved settings that can be loaded directly in the GUI or CLI to resume. |
| `analysis.sqlite3` | Resume database and per-epoch features/probabilities. |
| `masking_background.npz` | Enhanced only: immutable background references and source metadata, required for resume. |
| `masking_diagnostics.csv` | Enhanced only: per-epoch detection sources, rejected detections, and lighting changes. |
| `masking_summary.json` | Enhanced only: detection totals, reference identity, and preparation time. Counts are not accuracy scores. |

For interrupted analyses, results cover completed epochs and the summary separately reports unprocessed time. The HMM can update earlier predictions when more contiguous epochs become available after resume.

## Thermal processing cache

Per-cage temperature analysis caches raw patch temperatures, coordinates, integer acquisition timestamps, and a bounded selection of preview frames in `.cache/thermal/`. Sleep labels, calibration, exclusions, take gaps, summaries, and figures are recalculated each run. Raw-file/log paths, sizes and modification times, cage regions, preview selection, and extraction version determine cache validity. Invalid caches are rebuilt; preview-only runs bypass the full-recording cache.

Add `--no-cache` to `scripts.analyze_cage_temperatures` to bypass cache reads and writes, or `--refresh-cache` to rebuild from the original recording. The Python API accepts `use_cache=False` and `refresh_cache=True`. Cached files can be removed to reclaim space; they are regenerated from the original data.

## Command line

Run in PowerShell from this project folder:

```powershell
.\MouseSleep.cmd train
.\MouseSleep.cmd analyze --video "C:\Recordings\flir_cam0_take01.mp4" --output "C:\Results\mouse01" --appearance dark --roi 420 0 1080 1080
.\MouseSleep.cmd analyze --video "C:\Recordings\white.mp4" --output "C:\Results\white01" --appearance white --masking-mode legacy --max-seconds 120
.\MouseSleep.cmd analyze --config "C:\Results\mouse01\settings.json" --resume
.\MouseSleep.cmd prepare --video "C:\Recordings\flir_cam0.mp4" --output "C:\Results\fitted_preview" --roi 420 0 1080 1080
.\MouseSleep.cmd analyze --help
```

The example ROI matches the authors' centered square for a 1920 × 1080 video; select a crop appropriate for your actual camera and cage. Use **Save settings** in the GUI or `--save-config settings.json` on the CLI. `settings.example.json` documents every setting. `--model`, `--refinement`, `--threshold`, `--timing-csv`, `--take-id`, `--batch-size`, and `--classifier` override saved settings.

Alternatively, run `.venv\Scripts\python.exe -m mouse_sleep ...`. If opening the project in PyCharm, choose this project's `.venv\Scripts\python.exe` as its interpreter.

## Classifier and verification

Classifier setup trains XGBoost on the official reference feature CSV. Whole
video/animal identifiers are split before class balancing and evaluated on
held-out animals. Temporal transitions use chronologically adjacent epochs
within each identifier, breaking at missing data. The final reusable classifier
is trained on all reference identifiers.

Your generated `classifier/validation.json` contains held-out results;
`classifier/schema.json` records feature order, seed, library versions, data
checksum, model hash, and temporal probabilities. These describe that reference
split, not validation of your own recording. This modern retraining is not an
author-distributed pretrained sleep model or an exact reproduction of the
published validation experiment.

See [testing and validation](VALIDATION.md) for synthetic tests, optional model
parity and GUI checks, and benchmarking commands. Verification artifacts are
generated locally and are not shipped with the source. Real 24-hour light/dark
sleep accuracy still requires validation on the actual recording.

The training CSV and recovery chunks are cached in `data/`. Once training has
completed, predictions require `classifier/` and `pretrained_models/`, not the
cached CSV. Keep the data cache if you intend to retrain offline.

## Attribution and data licensing

See [attribution and external assets](ATTRIBUTION.md) for upstream code notices,
publications, official downloads, and the distinction between the source license
and external data/model terms.
