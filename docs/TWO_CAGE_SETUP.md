# Two cages: independent sleep and temperature analysis

Record both cages with compatible optical FLIR and Seek cameras, then analyze
the original optical video twice in Mouse Sleep Analyzer. Each run selects one
cage and produces that mouse's sleep labels. The temperature postprocessor reads
the raw thermal recording once for both mice and joins each series to its own
predictions. Keep stable mouse IDs, such as `cage_A` and `cage_B`, in every take.

## 1. Arrange the cameras and cages

- Place identical cages side by side, with their long sides vertical in the image.
  Keep their floors at the same height. Center the fixed overhead cameras so the
  complete accessible floor of each cage is visible, including corners.
- Choose a native optical recording mode that sustains **30 fps** and clearly
  resolves both mice. Test whether both cages fit at a useful mouse image size
  before moving the camera farther away. Available modes depend on the camera.
- Aim for roughly **480 × 480 native pixels or more per cage**, where the cage
  proportions allow. This is a framing target, not a validated accuracy cutoff.
  If necessary, use a higher native sensor resolution that sustains 30 fps.
  Resizing an already recorded video cannot recover missing detail.
- Preserve the source aspect ratio and square pixels. Do not squeeze each cage
  into a square or include the neighboring mouse to make a crop square.
  The app pads rectangular crops before resizing to its 480 × 480 network input.
  For example, a 522 × 588 crop occupies approximately 426 × 480 active pixels
  after padding. The existing 1080-coordinate measurement calculation is retained.
- Check pixel distances against the same floor dimensions in both cages. Keep
  their physical coverage and mouse image scales comparable. Cropping changes
  classifier feature scale even when the image proportions are correct.
- Focus on the mouse/floor plane, provide even visible or near-infrared light,
  and check moving boundaries for blur, shadows, reflections, and clipping.
  Use stable exposure and gain, and check whether the recorder overrides camera
  settings when opening it; see [the capture guide](NEAR_IR_SETUP.md).
- Preserve **30 actual acquisition frames per second**, not just playback fps.
  Keep the original MP4, `flir_log_camN.csv`, `takeNN_sync.json`, `seek_log.csv`,
  and `seek_thermal.csv` together. Keep the Seek frames at their native resolution (for example, **320 × 240**).
  Thermal acquisition cadence is separate from optical frame rate.

The [reference sleep study](https://academic.oup.com/sleep/article/45/2/zsab260/6414386)
recorded at 30 fps. Its accuracy does not automatically transfer to a new framing
or cage. Thermal digital zoom also does not increase measurement resolution;
see [FLIR's explanation](https://www.flir.com/en-eu/discover/professional-tools/how-far-can-you-measure-with-a-thermal-camera/).

## 2. Pilot and save one optical setup per mouse

1. Record a five-minute pilot and extend it until both mice have been seen moving,
   resting, at the walls/corners, and near the drinking spouts. Include each
   lighting condition that will occur in the full recording.
2. Open the **original optical MP4** in the app. Drag a region around Cage A's
   complete accessible area, excluding Cage B. Select the mouse's coat appearance.
3. For black mice, start with masking mode **Enhanced**, model **Automatic**
   (`AllCoatColor_RegularOFA`), batch **4**,
   maximum seconds **300**, coverage **0.95**, maximum gap **0.12 s**, confidence
   **0.60**. Draw and finish a polygon around that cage's complete mouse-accessible
   area to exclude reflected images. The polygon does not resize the model input.
   Enhanced estimates references separately for each cage and lighting condition
   across the take; a 12-hour lighting switch need not be entered manually.
4. Use the preview slider and **Test segmentation** on at least 30 representative
   frames per mouse. Require at least 29 satisfactory body masks, no selection of
   the neighboring mouse, and no recurring wall/spout failure. If necessary,
   compare all three models on the same frames. Select the most satisfactory
   masks; retain the starting model on a tie. Confidence measures classification,
   not mask placement. Do not lower quality thresholds to hide tracking failures.
   Use **Compare with Legacy** to inspect the original detector. If Enhanced is
   worse, select **Legacy**, `AllCoatColor_RegularOFA`, refinement **Off**, and a
   new output folder. White/light mice continue to use Legacy.
5. Save Cage A's settings and use a separate pilot output directory. Repeat for
   Cage B with its own region, settings, and pilot output. Check predicted sleep
   against visible behavior, especially quiet wakefulness.
6. Once both pass, remove the 300-second limit, choose fresh full-run directories,
   save the settings again, and analyze each cage separately for every take.
   Example folders: `outputs/cage_A/take01`, `outputs/cage_B/take01`, and similarly
   for take 2. Both mice must use the same original video for each take.

Keep regions fixed across takes unless the physical setup changes; if it moves,
recheck both optical and thermal regions. Cancel/resume works independently for
each saved run. Resume requires the original crop/settings/output directory.
Finish each run before temperature postprocessing, because resuming can revise
earlier sleep predictions. A completed duration-limited pilot is accepted, but
thermal samples outside its analyzed epochs remain **unknown**.

The pilot is a practical tracking acceptance check, not a measured biological
sleep accuracy. If either cage fails, adjust framing, native resolution, focus,
or lighting and repeat the pilot before the full recording.

## 3. Select thermal regions and inspect the patches

Copy [cages.example.json](../cages.example.json) to `cages.json`. Its left/right
half-frame rectangles are **illustrations**, not measured regions for your setup.
For each mouse, set:

- `mouse_id`: a unique folder-safe ID, used in every output row.
- `thermal_roi`: `[x, y, width, height]` in the **native thermal image**, with
  non-overlapping regions at least 7 pixels wide/high. Match cage identity
  visually; optical and thermal coordinates cannot be interchanged.
- `analysis_dirs`: that mouse's completed analysis folders for all recorded
  takes. Paths are relative to `cages.json`, or can be absolute. Take IDs and
  timestamp origins are read from each folder's `run.json`.

From the project folder in PowerShell, preview the first thermal frames even
before sleep analyses are ready:

```powershell
.venv\Scripts\python.exe -m scripts.analyze_cage_temperatures --recording-dir "C:\Recordings\two_cages" --config cages.json --output outputs\two_cages_preview --preview-only
```

Open `thermal_preview.png`. Its axes show native pixel coordinates; the larger
boxes mark cage regions and the small boxes mark candidate measurement patches.
Adjust the regions in the config and rerun. `--preview-frames 30` increases the
contact sheet from the default 12 to 30 frames. Preview mode checks only these
first frames, not the whole recording or sleep alignment.

The measurement is the **hottest complete 7 × 7 mean patch inside each cage**,
using finite native Celsius pixels. It never searches the other cage or stretches
thermal images. Equal means choose the first patch in row-major order. This is
an explicit per-cage patch definition; the old one-animal calibrated CSV cannot
separate two mice and is not reused.

Confirm patches lie on the mice during activity and rest. A full patch must fit
on the visible mouse at the new distance; warm objects and bedding can otherwise
win the search. The software does not infer tissue identity from heat alone.
If the patch cannot fit reliably, improve the framing before collecting the
full experiment. Edge-touching patches are flagged for inspection, not silently
discarded. No finite patch means an unusable sample for that mouse only.

The example configuration contains these calibration coefficients; replace them with a calibration appropriate for your camera and setup:
`temperature = 0.9674 × camera_mean + 2.2661 °C`. Confirm calibration suitability
at the new distance; change `calibration` only with a measured replacement.
These measurements describe observed surface temperature, not core temperature.

## 4. Produce the final per-mouse reports

```powershell
.venv\Scripts\python.exe -m scripts.analyze_cage_temperatures --recording-dir "C:\Recordings\two_cages" --config cages.json --output outputs\two_cages_temperature
```

Full analysis validates raw frames against `seek_log.csv` and aligns each sample
to its mouse's epoch using the shared host clock. Each configured analysis must
match this recording's FLIR log. Missing take analyses, reused analysis folders,
overlapping optical/thermal regions, or inconsistent timestamps produce an error.
The original recordings and sleep results are read only. Rerunning replaces the
generated temperature reports after the new computation succeeds.

Temperature extraction is cached in the analyzer's `.cache/thermal/` folder.
Repeated runs reapply current sleep labels, calibration and exclusions without
parsing the raw thermal movie again. Changing source-file/log metadata, cage
regions or preview selection triggers a fresh extraction. Add `--refresh-cache`
to rebuild explicitly, or `--no-cache` to bypass cache reads and writes.
Preview-only runs always inspect the source directly.

The root output contains `run_summary.json` and a `thermal_preview.png` sampled
across the complete recording. Each mouse's subfolder contains:

| File | Contents |
| --- | --- |
| `temperature_samples.csv` | Every native sample, integer acquisition timestamps, take-relative time, sleep label, raw/calibrated temperature, native patch coordinates, usability/rejection reason, and edge flag. |
| `state_intervals.csv` | Contiguous predicted awake/sleep/unknown intervals and temperature summaries. |
| `sleep_temperature_changes.csv` | Sleep minus preceding awake, sleep minus last awake minute, first sleep minus last awake minute, and last minus first sleep minute; includes window sample counts. |
| `summary.json` | Per-take/all-take statistics, calibration, exclusions, crop geometry, data gaps, and provenance. |
| `temperature_analysis.md` | Readable statistics and measurement notes. |
| `temperature_awake_vs_sleep.png` / `.pdf` | Panel A: temperature and state timeline. Panel B: awake/sleep means and SD. No panel C. |

Unknown sleep labels never become sleep. Bad temperature samples remain in the
CSV but are omitted from statistics and plotted traces. Raw temperatures with
unknown sleep state may appear in gray. Figures preserve actual recording pauses
from the take metadata; no fixed 16-second offset is assumed for a new recording.
Lines and baselines also break at missing thermal data. The default
`max_thermal_gap_seconds` is **2.5 s**, suitable as a starting threshold for the
a roughly 1 Hz thermal capture; set it for the actual acquisition cadence.

If a candidate patch selects a spout or another object, add a rejection to that
mouse's `exclude_intervals` and rerun, for example:

```json
"exclude_intervals": [
  {"take_id": 1, "start_seconds": 42.0, "end_seconds": 48.0, "reason": "Patch selected drinking spout"}
]
```

These times use **the first FLIR frame of that take as zero** (the CSV's
`take_seconds`), with start included and end excluded. They reject temperature
measurements, not sleep predictions. Inspect the original Seek PNGs for detailed
review; `frame_index` maps to `idx` in `seek_log.csv`, which gives each PNG path.
The postprocessor retains the candidate patch coordinates even for exclusions.

The comparisons use equally weighted original usable samples, with sample SD
(`ddof=1`), not independent-animal confidence intervals. An absent state produces
an empty result, not zero. No baseline is carried across a take pause, unknown
sleep interval, thermal gap, or rejected temperature period. First/last-minute
windows are clipped to the sleep bout and can overlap for short bouts. Inspect
the saved sample counts when interpreting those changes.
