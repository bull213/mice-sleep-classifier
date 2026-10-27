# Testing and validation

Run commands from the project root after installing dependencies. The source
release contains test code and fixture generators, not recorded test data,
historical result files, pretrained weights, or a trained classifier.

## Tests without downloaded assets

```powershell
.venv\Scripts\python.exe -m pytest -ra
```

Tests cover reference feature calculations and morphology, timing and dropped
frames, synthetic video preparation, GUI controls, cancellation and resume,
output storage, Enhanced masking, mocked classifier training and downloads,
and thermal extraction/alignment. Videos and tables are generated in temporary
directories. These tests establish software behavior, not sleep accuracy.

The three parameterized `requires_models` tests skip with a setup explanation
when a required `.index` or `.data-00000-of-00001` file is missing. If both files
are present, they are loaded normally: corrupt or incompatible files fail the
test. No downloads occur during the test suite. GitHub Actions runs this suite
on Windows with Python 3.12 and no external models or recordings.

To select only checks that require no checkpoints:

```powershell
.venv\Scripts\python.exe -m pytest -m "not requires_models" -ra
```

## Checks with external models

Install all three checkpoints using [the installation guide](INSTALLATION.md).
Then run:

```powershell
.venv\Scripts\python.exe -m scripts.check_models
.venv\Scripts\python.exe -m pytest -m requires_models -ra
.venv\Scripts\python.exe -m scripts.check_parity
.venv\Scripts\python.exe -m scripts.check_gui
```

`check_models` restores each model and predicts a synthetic frame. The marked
tests compare predicted cleanup with reference morphology. `check_parity`
compares probabilities with the authors' original `.meta` graph files.
`check_gui` generates a synthetic video and exercises GUI segmentation preview.
These scripts fail when required assets are missing rather than downloading
them. They write local artifacts under the ignored `verification/` directory.

After training the sleep classifier, exercise the actual model-to-report path:

```powershell
.venv\Scripts\python.exe -m scripts.check_end_to_end
```

This covers both appearance settings, partial-epoch handling, and output files
using a synthetic video. The optional `scripts.check_white_reference` command
also requires the authors' separately obtained demonstration video; see
[its source notice](REFERENCE_NOTICE.md).

## Benchmarks and scientific validation

Measure performance on your own hardware and recording. Save settings from the
GUI first, then use a new output folder:

```powershell
.venv\Scripts\python.exe -m scripts.benchmark_segmentation --config settings.json
.venv\Scripts\python.exe -m scripts.benchmark_analysis --config settings.json --seconds 60.5 --output verification\performance\new_pilot
```

Omitting `--config` from the segmentation benchmark uses synthetic inputs, but
still requires a checkpoint. The full-analysis benchmark also requires a trained
classifier. Use `--help` for output and baseline-comparison options.

Review masks across cage positions, movement, resting periods, and light/dark
transitions. Compare selected predicted sleep intervals with observations.
EEG/EMG comparison is needed to establish sleep accuracy for research claims.
Held-out reference-data scores in your generated `classifier/validation.json`
describe that training split; they do not establish accuracy for new recordings.
White-mouse sleep classification remains experimental. Unknown intervals must
remain distinct from sleep and awake time.
