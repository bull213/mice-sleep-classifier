# Near-infrared cage recording

Starting guidance for a fixed overhead optical FLIR camera, a dark mouse, and
near-infrared illumination. Camera settings must be checked with a pilot
recording; they are not a validated exposure/gain combination for every setup.

For dark mice, load [the Enhanced preset](../settings.enhanced_near_ir.json),
choose your recording and output folder, and draw the cage boundary.
[The Legacy preset](../settings.near_ir.json) is retained for comparison with
`AllCoatColor_RegularOFA` and no pixel refinement. Both presets have blank paths
and a 300-second pilot limit. See [masking modes](USER_GUIDE.md#enhanced-and-legacy-masking).

## Camera settings

Stop recording before changing settings in SpinView. Use a stable full view of
the mouse-accessible cage floor and preserve approximately 30 acquisition fps.

| Control | Starting setting |
| --- | --- |
| Acquisition frame rate | 30 fps; enable frame-rate control if available |
| Exposure Auto | Off |
| Exposure Time | 10000 microseconds (10 ms), then inspect movement and brightness |
| Gain Auto | Off |
| Gain | Start at the camera's minimum; increase only enough to distinguish the mouse from bedding |
| Capture ROI | Keep the full cage visible; select the analysis region in the analyzer |

Feature availability and allowed ranges depend on the camera. Record a short
clip containing movement and visits near walls and drinking spouts. If boundaries
blur during movement, test a shorter exposure; if dark regions lose detail,
adjust illumination, gain, or exposure while checking for blur and clipping.
Change one setting at a time and inspect the histogram and original video.

Some recorders apply their own exposure and frame-rate settings when opening a
camera. Verify the actual values after starting your recorder; saving a SpinView
profile does not prevent application overrides. These presets configure only
the analyzer and do not change a connected camera.

## Analyzer checks

1. Load the Enhanced preset for a dark mouse, then select your original video
   and a new pilot output directory.
2. Draw the cage crop and a boundary covering the complete accessible area.
3. Use **Test segmentation** at several positions, during rest and movement,
   and under every illumination condition expected in the full recording.
4. Use **Compare with Legacy** when useful. Check that the mask follows the
   body rather than bedding, a reflection, or a warm/bright object.
5. Review a 300-second pilot. After acceptable checks, remove the duration limit
   and choose a new output directory for the full analysis.

The Legacy preset uses no refinement; its stored threshold of 30 is inactive.
Enhanced uses its background-contrast and area settings instead of that threshold.
Model confidence controls sleep classification, not mask placement. Repeated
tracking errors require correcting the setup or detector settings.

Keep compatible acquisition logs beside recordings. Without a matching log,
the analyzer uses video timestamps and cannot reconstruct acquisition gaps that
fixed-rate encoding has removed. White/light mice use Legacy masking; their
sleep classification remains experimental.

## Camera documentation

- [Teledyne: Using Auto Exposure](https://www.teledynevisionsolutions.com/support/support-center/application-note/iis/using-auto-exposure/)
- [Official SpinView guide](https://softwareservices.flir.com/spinnaker/4.4/guides/spinview.html)
