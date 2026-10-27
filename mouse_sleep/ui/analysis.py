from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from contextlib import closing

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QLineEdit, QComboBox, QLabel,
    QHBoxLayout, QGroupBox, QSpinBox, QDoubleSpinBox, QPushButton,
    QSplitter, QSlider, QGridLayout, QCheckBox, QFileDialog,
)
from PySide6.QtCore import Qt

from ..config import ROOT, MODEL_STEPS, Cancelled, Settings
from ..runtime import InferenceSession
from ..video import Timing, frames, sampled_frames, inspect_video
from ..tasks import check_cancelled
from .widgets import ImageView, clock_text, path_row, pair


class AnalysisPanel(QWidget):
    statusChanged = Signal(str)
    failed = Signal(str)
    analysisFinished = Signal(str, object)

    def __init__(self, tasks, parent=None):
        super().__init__(parent)
        self.tasks = tasks
        self.inference = InferenceSession()
        self.timing = self.info = self.gray = None
        self.loaded_video = ""
        self.frame_index, self.frame_stamp = 0, 0.0
        self.background_images = ()
        self._build()

    def set_busy(self, busy):
        for widget in (self.controls, self.advanced, self.actions, self.seek, self.boundary_controls):
            widget.setEnabled(not busy)
        self.source_view.setEnabled(not busy)
        self.source_view.draw_roi = not busy and not self.source_view.draw_polygon

    def _build(self):
        layout = QVBoxLayout(self)
        self.controls = QWidget()
        form = QFormLayout(self.controls)
        self.video, self.output = QLineEdit(), QLineEdit()
        self.video.setPlaceholderText("Original .mp4 recording")
        self.output.setPlaceholderText("A new folder for this analysis")
        form.addRow("Video", path_row(self.video, self.choose_video))
        form.addRow("Output folder", path_row(self.output, self.choose_output))
        self.masking_mode = QComboBox()
        self.masking_mode.addItem("Enhanced · fixed backgrounds + neural detection", "enhanced")
        self.masking_mode.addItem("Legacy · original masking", "legacy")
        form.addRow("Masking mode", self.masking_mode)
        self.appearance, self.model, self.refinement = QComboBox(), QComboBox(), QComboBox()
        for label, data in (("Dark / black mouse", "dark"), ("White / light mouse · experimental sleep estimates", "white")):
            self.appearance.addItem(label, data)
        self.model.addItem("Automatic for appearance", "auto")
        for name in MODEL_STEPS:
            self.model.addItem(name, name)
        for label, data in (("Automatic", "auto"), ("None", "off"), ("Keep dark pixels", "dark"), ("Keep bright pixels", "bright")):
            self.refinement.addItem(label, data)
        options = QWidget()
        row = QHBoxLayout(options)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.appearance, 2)
        row.addWidget(self.model, 1)
        row.addWidget(self.refinement, 1)
        form.addRow("Appearance / model / refinement", options)
        self.notice = QLabel()
        self.notice.setWordWrap(True)
        form.addRow(self.notice)
        self.appearance.currentIndexChanged.connect(self.update_mode)
        self.model.currentIndexChanged.connect(self.update_notice)
        layout.addWidget(self.controls)

        self.advanced = QGroupBox("Advanced settings")
        advanced = QFormLayout(self.advanced)
        self.threshold = QSpinBox()
        self.threshold.setRange(0, 255)
        self.batch = QSpinBox()
        self.batch.setRange(1, 32)
        self.confidence, self.coverage, self.max_gap = QDoubleSpinBox(), QDoubleSpinBox(), QDoubleSpinBox()
        for spin in (self.confidence, self.coverage):
            spin.setRange(0.5, 1)
            spin.setSingleStep(0.05)
        self.max_gap.setRange(0.01, 1)
        self.max_gap.setSingleStep(0.01)
        settings = QWidget()
        row = QHBoxLayout(settings)
        row.setContentsMargins(0, 0, 0, 0)
        for text, field in (("Pixel threshold", self.threshold), ("Batch", self.batch), ("Confidence", self.confidence), ("Coverage", self.coverage), ("Max gap (s)", self.max_gap)):
            row.addWidget(QLabel(text))
            row.addWidget(field)
        advanced.addRow(settings)
        self.enhanced_options = QWidget()
        enhanced_form = QFormLayout(self.enhanced_options)
        enhanced_form.setContentsMargins(0, 0, 0, 0)
        self.background_delta, self.background_fraction = QDoubleSpinBox(), QDoubleSpinBox()
        self.background_delta.setRange(1, 255)
        self.background_fraction.setRange(0.01, 0.99)
        self.background_fraction.setSingleStep(0.05)
        self.minimum_mouse_area, self.maximum_mouse_area = QDoubleSpinBox(), QDoubleSpinBox()
        for spin in (self.minimum_mouse_area, self.maximum_mouse_area):
            spin.setDecimals(4)
            spin.setRange(0.0001, 0.4)
            spin.setSingleStep(0.001)
        enhanced_form.addRow("Background contrast · levels / fraction", pair(self.background_delta, self.background_fraction))
        enhanced_form.addRow("Mouse area · minimum / maximum fraction", pair(self.minimum_mouse_area, self.maximum_mouse_area))
        self.background_label = QLabel("Backgrounds: estimated automatically for each lighting condition")
        self.background_label.setWordWrap(True)
        background_buttons = QWidget()
        background_row = QHBoxLayout(background_buttons)
        background_row.setContentsMargins(0, 0, 0, 0)
        for text, callback in (("Select empty-cage images…", self.choose_backgrounds), ("Use automatic backgrounds", self.clear_backgrounds)):
            button = QPushButton(text)
            button.clicked.connect(callback)
            background_row.addWidget(button)
        enhanced_form.addRow(self.background_label)
        enhanced_form.addRow(background_buttons)
        advanced.addRow(self.enhanced_options)
        self.timing_csv, self.classifier = QLineEdit(), QLineEdit()
        self.timing_csv.setPlaceholderText("Automatic discovery of flir_log_camN.csv beside the MP4")
        self.classifier.setPlaceholderText(str(ROOT / "classifier"))
        self.take = QSpinBox()
        self.take.setRange(0, 999999)
        self.take.setSpecialValueText("Auto")
        self.limit = QDoubleSpinBox()
        self.limit.setRange(0, 1000000)
        self.limit.setDecimals(1)
        self.limit.setSpecialValueText("Entire video")
        self.limit.setToolTip("Use 60–300 seconds for a quick first check.")
        advanced.addRow("Timing CSV", path_row(self.timing_csv, self.choose_csv))
        advanced.addRow("Take ID / maximum seconds", pair(self.take, self.limit))
        advanced.addRow("Classifier folder", path_row(self.classifier, self.choose_classifier))
        self.advanced.setVisible(False)
        toggle = QPushButton("Show / hide advanced settings")
        toggle.clicked.connect(lambda: self.advanced.setVisible(not self.advanced.isVisible()))
        layout.addWidget(toggle)
        layout.addWidget(self.advanced)

        preview = QSplitter()
        self.source_view, self.mask_view, self.legacy_view = ImageView(), ImageView(), ImageView()
        self.source_view.draw_roi = True
        self.source_view.roiPicked.connect(self.set_roi)
        self.source_view.polygonPicked.connect(self.set_polygon)
        for label, view in (("Original · drag a rectangle around the cage", self.source_view),
                            ("Segmentation · green = tracked mouse", self.mask_view), ("Legacy comparison", self.legacy_view)):
            panel = QWidget()
            column = QVBoxLayout(panel)
            column.addWidget(QLabel(label))
            column.addWidget(view, 1)
            preview.addWidget(panel)
            if view is self.legacy_view:
                self.legacy_panel = panel
                panel.hide()
        layout.addWidget(preview, 1)
        self.roi_label = QLabel("Cage region: whole frame. A square crop around the cage is recommended.")
        layout.addWidget(self.roi_label)
        self.boundary_controls = QWidget()
        boundary_row = QHBoxLayout(self.boundary_controls)
        boundary_row.setContentsMargins(0, 0, 0, 0)
        for text, callback in (("Draw cage boundary", self.begin_boundary), ("Undo point", self.source_view.undo_polygon),
                               ("Finish boundary", self.finish_boundary), ("Reset boundary", self.source_view.clear_polygon)):
            button = QPushButton(text)
            button.clicked.connect(callback)
            boundary_row.addWidget(button)
        layout.addWidget(self.boundary_controls)
        self.seek = QSlider(Qt.Orientation.Horizontal)
        self.seek.setRange(0, 0)
        self.seek.sliderReleased.connect(self.preview_at_seek)
        self.seek.valueChanged.connect(lambda value: self.seek_label.setText(clock_text(value)))
        self.seek_label = QLabel("00:00:00")
        row = QHBoxLayout()
        row.addWidget(self.seek, 1)
        row.addWidget(self.seek_label)
        layout.addLayout(row)
        self.actions = QWidget()
        row = QGridLayout(self.actions)
        for index, (text, callback) in enumerate((("Reload preview", self.load_video), ("Reset cage region", lambda: self.set_roi(None)), ("Test segmentation", self.test_segmentation),
                               ("Save settings", self.save_settings), ("Load settings", self.load_settings), ("Export fitted video", self.prepare_video), ("Analyze", self.start_analysis),
                               ("Compare with Legacy", lambda: self.test_segmentation(compare=True)))):
            button = QPushButton(text)
            button.clicked.connect(callback)
            row.addWidget(button, index // 4, index % 4)
            if text == "Analyze":
                button.setObjectName("primary")
            if text == "Compare with Legacy":
                self.compare_button = button
        self.resume = QCheckBox("Resume")
        row.addWidget(self.resume, 2, 3)
        layout.addWidget(self.actions)
        self.masking_mode.currentIndexChanged.connect(self.update_mode)

    def settings(self):
        return Settings(video=self.video.text().strip(), output=self.output.text().strip(), appearance=self.appearance.currentData(),
                        model=self.model.currentData(), refinement=self.refinement.currentData(), threshold=self.threshold.value(), roi=self.source_view.roi,
                        batch_size=self.batch.value(), minimum_confidence=self.confidence.value(), minimum_coverage=self.coverage.value(),
                        maximum_gap_seconds=self.max_gap.value(), timing_csv=self.timing_csv.text().strip(), take_id=self.take.value() or None,
                        classifier=self.classifier.text().strip(), max_seconds=self.limit.value() or None,
                        masking_mode=self.masking_mode.currentData(), cage_polygon=self.source_view.polygon,
                        background_images=self.background_images, background_delta=self.background_delta.value(),
                        background_fraction=self.background_fraction.value(), minimum_mouse_area=self.minimum_mouse_area.value(),
                        maximum_mouse_area=self.maximum_mouse_area.value())

    def apply_settings(self, settings):
        self.video.setText(settings.video)
        self.output.setText(settings.output)
        for combo, value in ((self.appearance, settings.appearance), (self.model, settings.model),
                             (self.refinement, settings.refinement), (self.masking_mode, settings.masking_mode)):
            combo.blockSignals(True)
            combo.setCurrentIndex(combo.findData(value))
            combo.blockSignals(False)
        self.threshold.setValue(settings.threshold)
        self.batch.setValue(settings.batch_size)
        self.confidence.setValue(settings.minimum_confidence)
        self.coverage.setValue(settings.minimum_coverage)
        self.max_gap.setValue(settings.maximum_gap_seconds)
        self.timing_csv.setText(settings.timing_csv)
        self.take.setValue(settings.take_id or 0)
        self.classifier.setText(settings.classifier)
        self.limit.setValue(settings.max_seconds or 0)
        self.set_roi(settings.roi)
        self.set_polygon(settings.cage_polygon)
        self.background_images = settings.background_images
        self.background_delta.setValue(settings.background_delta)
        self.background_fraction.setValue(settings.background_fraction)
        self.minimum_mouse_area.setValue(settings.minimum_mouse_area)
        self.maximum_mouse_area.setValue(settings.maximum_mouse_area)
        self.update_background_label()
        self.update_mode()

    def update_mode(self):
        white = self.appearance.currentData() == "white"
        self.masking_mode.model().item(0).setEnabled(not white)
        if white and self.masking_mode.currentData() == "enhanced":
            self.masking_mode.setCurrentIndex(self.masking_mode.findData("legacy"))
        enhanced = self.masking_mode.currentData() == "enhanced"
        self.source_view.show_polygon = enhanced
        self.enhanced_options.setVisible(enhanced)
        self.boundary_controls.setVisible(enhanced)
        self.refinement.setEnabled(not enhanced)
        self.threshold.setEnabled(not enhanced)
        self.compare_button.setEnabled(enhanced)
        self.legacy_panel.hide()
        if not enhanced:
            self.source_view.draw_polygon = False
            self.source_view.polygon_points = []
            self.source_view.draw_roi = True
        self.mask_view.image = None
        self.mask_view.update()
        self.source_view.update()
        self.update_notice()

    def update_notice(self):
        name = self.settings().model_name
        text = f"Using {name}. Record normally at 30 fps; the analyzer fits the selected cage region to the model automatically."
        if self.appearance.currentData() == "white":
            text += " White/light mice use Legacy masking. Check daytime and nighttime previews; sleep estimates are experimental."
        elif self.masking_mode.currentData() == "enhanced":
            text += " Enhanced uses fixed references for light/dark conditions. Draw a cage boundary to exclude reflections."
        else:
            text += " Legacy preserves the original masking and pixel-refinement behavior."
        self.notice.setText(text)

    def set_roi(self, roi):
        if self.source_view.roi != (tuple(roi) if roi else None):
            self.source_view.clear_polygon()
        self.source_view.roi = tuple(roi) if roi else None
        self.source_view.update()
        self.roi_label.setText(f"Cage region (x, y, width, height): {roi}" if roi else "Cage region: whole frame. Drag a rectangle around the cage.")
        self.mask_view.image = None
        self.mask_view.update()

    def set_polygon(self, polygon):
        self.source_view.polygon = tuple(tuple(point) for point in polygon) if polygon else None
        self.source_view.draw_polygon = False
        self.source_view.polygon_points = []
        self.source_view.draw_roi = True
        self.source_view.update()
        self.mask_view.image = None
        self.mask_view.update()
        self.legacy_panel.hide()

    def begin_boundary(self):
        if self.gray is None:
            self.statusChanged.emit("Load a video preview before drawing its cage boundary.")
            return
        self.source_view.begin_polygon()
        self.statusChanged.emit("Click around the complete mouse-accessible area, then Finish boundary or double-click. Exclude reflected images.")

    def finish_boundary(self):
        if self.source_view.finish_polygon():
            self.statusChanged.emit("Cage boundary saved in original image coordinates. Test segmentation to check it.")
        else:
            self.statusChanged.emit("Place at least three boundary points before finishing.")

    def update_background_label(self):
        names = ", ".join(Path(path).name for path in self.background_images)
        self.background_label.setText(f"Empty-cage references: {names}" if names else "Backgrounds: estimated automatically for each lighting condition")

    def choose_backgrounds(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Select up to two matching empty-cage images", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if len(paths) > 2:
            self.failed.emit("Select at most two empty-cage images, one per lighting condition.")
        elif paths:
            self.background_images = tuple(paths)
            self.update_background_label()

    def clear_backgrounds(self):
        self.background_images = ()
        self.update_background_label()

    def choose_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Original FLIR video", self.video.text(), "Video (*.mp4 *.mkv *.avi);;All files (*)")
        if path:
            self.video.setText(path)
            self.timing_csv.clear()
            self.take.setValue(0)
            self.set_roi(None)
            self.source_view.clear_polygon()
            self.clear_backgrounds()
            self.output.setText(str(Path(path).parent / (Path(path).stem + "_sleep")))
            self.load_video()

    def choose_output(self):
        path = QFileDialog.getExistingDirectory(self, "Analysis output folder", self.output.text())
        if path:
            self.output.setText(path)

    def choose_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Recorder timing CSV", "", "CSV (*.csv)")
        if path:
            self.timing_csv.setText(path)
            self.load_video()

    def choose_classifier(self):
        path = QFileDialog.getExistingDirectory(self, "Classifier folder", str(ROOT / "classifier"))
        if path:
            self.classifier.setText(path)

    def load_video(self):
        settings = self.settings()
        def action(progress, cancel):
            progress({"message": "Reading video and acquisition timing"})
            info = inspect_video(settings.video)
            timing = Timing(settings)
            with closing(frames(replace(settings, max_seconds=None), timing, cancel=cancel)) as source:
                frame = next(source)
            return settings.video, info, timing, frame
        self.tasks.start(action, self.video_loaded)

    def video_loaded(self, result):
        self.loaded_video, self.info, self.timing, frame = result
        self.frame_index, self.frame_stamp, self.gray = frame
        self.seek.setRange(0, max(0, int(self.timing.duration(self.info) - 1 / max(self.info.fps, 1))))
        self.seek.setValue(0)
        self.source_view.set_image(self.gray)
        self.statusChanged.emit(f"{self.info.width} × {self.info.height}, {self.info.fps:.3f} fps · {clock_text(self.timing.duration(self.info))} · {self.timing.source}")

    def preview_at_seek(self):
        settings = self.settings()
        second = self.seek.value()
        timing = self.timing if self.loaded_video == settings.video else None
        def action(progress, cancel):
            source_timing = timing or Timing(settings)
            with closing(frames(replace(settings, max_seconds=None), source_timing, second, cancel)) as source:
                return next(source)
        self.tasks.start(action, self.preview_loaded)

    def preview_loaded(self, frame):
        self.frame_index, self.frame_stamp, self.gray = frame
        self.source_view.set_image(self.gray)
        self.mask_view.image = None
        self.mask_view.update()
        self.statusChanged.emit("Preview ready. Test segmentation at several times, including both lighting conditions.")

    def test_segmentation(self, checked=False, *, compare=False):
        if self.gray is None or self.loaded_video != self.video.text().strip():
            self.load_video()
            return
        if self.source_view.draw_polygon:
            self.statusChanged.emit("Finish or reset the cage boundary before testing segmentation.")
            return
        settings, gray = self.settings(), self.gray.copy()
        stamp, index, info, timing = self.frame_stamp, self.frame_index, self.info, self.timing
        def action(progress, cancel):
            from ..segmentation import FramePreparer
            from ..masking import LegacyMasker, EnhancedMasker, render_mask
            from ..backgrounds import prepare_backgrounds
            replace(settings, output=settings.output or ".").validate()
            prepare = FramePreparer(settings.roi)
            prepared = prepare(gray)
            legacy_render = None
            if settings.masking_mode == "enhanced":
                backgrounds, geometry = prepare_backgrounds(settings, info, timing, progress=progress, cancel=cancel)
                detector = EnhancedMasker(settings, backgrounds, geometry)
                progress({"message": f"Testing Enhanced / {settings.model_name}; warming up one second of tracking"})
                result, last_stamp = None, None
                with self.inference.lease(settings.checkpoint) as model:
                    check_cancelled(cancel)
                    batch = []
                    def process():
                        nonlocal result, last_stamp
                        predictions = model.predict([item[2][0] for item in batch])
                        for (frame_index, frame_stamp, frame), prediction in zip(batch, predictions, strict=True):
                            ids = timing.frame_ids
                            gap = ids is not None and frame_index > 0 and ids[frame_index] - ids[frame_index - 1] > 4
                            result = detector.process(prediction, frame, frame_stamp, device_gap=gap)
                            last_stamp = frame_stamp
                        batch.clear()
                    context = replace(settings, max_seconds=stamp + 1e-6)
                    with closing(sampled_frames(context, timing, max(0, stamp - 1), cancel)) as source:
                        for frame_index, frame_stamp, frame_gray in source:
                            check_cancelled(cancel)
                            batch.append((frame_index, frame_stamp, prepare(frame_gray)))
                            if len(batch) >= settings.batch_size:
                                process()
                    if batch:
                        process()
                    if last_stamp is None or last_stamp < stamp - 1e-7:
                        batch.append((index, stamp, prepared))
                        process()
                rendered = render_mask(prepared[1], result.mask)
                area = float(result.mask.mean())
                description = result.reason or result.source
                detail = f"Enhanced: {description.replace('_', ' ')} · tracked area {area:.1%}"
                if result.lighting is not None:
                    detail += f" · lighting reference {result.lighting + 1}"
            if settings.masking_mode == "legacy" or compare:
                legacy = replace(settings, masking_mode="legacy")
                check_cancelled(cancel)
                progress({"message": f"Testing Legacy / {legacy.model_name}"})
                with self.inference.lease(legacy.checkpoint) as model:
                    check_cancelled(cancel)
                    prediction = model.predict([prepared[0]])[0]
                result = LegacyMasker(legacy).process(prediction, prepared, stamp)
                legacy_render = render_mask(prepared[1], result.mask)
                if settings.masking_mode == "legacy":
                    rendered, legacy_render = legacy_render, None
                    detail = f"Legacy: tracked area {float(result.mask.mean()):.1%}. Check that the complete mouse is selected."
                else:
                    detail += f" · compared with Legacy / {legacy.model_name}"
            return rendered, legacy_render, detail
        def done(result):
            self.mask_view.set_image(result[0])
            self.legacy_panel.setVisible(result[1] is not None)
            if result[1] is not None:
                self.legacy_view.set_image(result[1])
            self.statusChanged.emit(result[2])
        self.tasks.start(action, done)

    def save_settings(self):
        if self.source_view.draw_polygon:
            self.statusChanged.emit("Finish or reset the cage boundary before saving settings.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save analysis settings", str(ROOT / "settings.json"), "JSON (*.json)")
        if path:
            try:
                self.settings().save(path)
                self.statusChanged.emit("Settings saved.")
            except Exception as exc:
                self.failed.emit(str(exc))

    def load_settings(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load analysis settings", str(ROOT), "JSON (*.json)")
        if path:
            try:
                self.apply_settings(Settings.load(path))
                if Path(self.video.text()).is_file():
                    self.load_video()
                else:
                    self.gray = self.info = self.timing = None
                    self.loaded_video = ""
                    self.source_view.image = None
                    self.source_view.update()
                    self.statusChanged.emit("Settings loaded. Select the original video and an output folder.")
            except Exception as exc:
                self.failed.emit(str(exc))

    def start_analysis(self):
        if self.source_view.draw_polygon:
            self.statusChanged.emit("Finish or reset the cage boundary before starting analysis.")
            return
        try:
            settings = self.settings().validate()
            settings.save(ROOT / "last_settings.json")
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        resume = self.resume.isChecked()
        def action(progress, cancel):
            from ..pipeline import analyze
            return analyze(settings, resume=resume, progress=progress, cancel=cancel,
                           segmenter_factory=self.inference.lease)
        self.tasks.start(action, lambda result: self.analysis_done(settings.output, result))

    def close_resources(self):
        self.inference.close()

    def analysis_done(self, folder, result):
        self.analysisFinished.emit(folder, result)
        self.statusChanged.emit("Analysis complete." if result["status"] == "complete" else "Cancelled. Select Resume to continue from the saved epoch.")

    def prepare_video(self):
        from ..video import prepare
        try:
            settings = self.settings().validate()
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        directory = QFileDialog.getExistingDirectory(self, "Folder for fitted preview video", str(Path(settings.output).parent))
        if directory:
            self.tasks.start(lambda progress, cancel: prepare(replace(settings, output=directory), progress, cancel),
                             lambda result: self.statusChanged.emit(f"Fitted video saved: {result}. Analyze the original MP4 to preserve acquisition timing."))

