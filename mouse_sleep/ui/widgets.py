from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, QRectF, QPointF, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QWidget, QHBoxLayout, QPushButton

def clock_text(seconds):
    seconds = max(0, int(seconds))
    return f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


class ImageView(QWidget):
    roiPicked = Signal(object)
    polygonPicked = Signal(object)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(260, 230)
        self.image = None
        self.roi = None
        self.draw_roi = False
        self.polygon = None
        self.show_polygon = True
        self.draw_polygon = False
        self.polygon_points = []
        self.anchor = self.cursor_point = None

    def set_image(self, array):
        array = np.ascontiguousarray(array)
        if array.ndim == 2:
            self.image = QImage(array.data, array.shape[1], array.shape[0], array.strides[0], QImage.Format.Format_Grayscale8).copy()
        else:
            self.image = QImage(array.data, array.shape[1], array.shape[0], array.strides[0], QImage.Format.Format_RGB888).copy()
        self.update()

    def display_rect(self):
        if self.image is None:
            return QRectF()
        scale = min(self.width() / self.image.width(), self.height() / self.image.height())
        w, h = self.image.width() * scale, self.image.height() * scale
        return QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)

    def image_point(self, point):
        rect = self.display_rect()
        if rect.isEmpty():
            return None
        return (int(np.clip((point.x() - rect.x()) / rect.width() * self.image.width(), 0, self.image.width() - 1)),
                int(np.clip((point.y() - rect.y()) / rect.height() * self.image.height(), 0, self.image.height() - 1)))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#0a1220"))
        rect = self.display_rect()
        if self.image is None:
            painter.setPen(QColor("#a0aec0"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Select a video to preview")
            return
        painter.drawImage(rect, self.image)
        roi = self.roi
        if self.anchor is not None and self.cursor_point is not None:
            x, y = min(self.anchor[0], self.cursor_point[0]), min(self.anchor[1], self.cursor_point[1])
            roi = (x, y, abs(self.anchor[0] - self.cursor_point[0]), abs(self.anchor[1] - self.cursor_point[1]))
        if roi:
            sx, sy = rect.width() / self.image.width(), rect.height() / self.image.height()
            x, y, w, h = roi
            painter.setPen(QPen(QColor("#48d8b4"), 2))
            painter.drawRect(QRectF(rect.x() + x * sx, rect.y() + y * sy, w * sx, h * sy))
        points = (self.polygon_points if self.draw_polygon else self.polygon) if self.show_polygon else None
        if points:
            sx, sy = rect.width() / self.image.width(), rect.height() / self.image.height()
            mapped = QPolygonF([QPointF(rect.x() + x * sx, rect.y() + y * sy) for x, y in points])
            painter.setPen(QPen(QColor("#ffc864"), 2))
            if self.draw_polygon:
                painter.drawPolyline(mapped)
            else:
                painter.drawPolygon(mapped)
            for point in mapped:
                painter.drawEllipse(point, 3, 3)

    def begin_polygon(self):
        self.draw_polygon, self.draw_roi = True, False
        self.polygon_points = []
        self.anchor = self.cursor_point = None
        self.update()

    def undo_polygon(self):
        if self.draw_polygon and self.polygon_points:
            self.polygon_points.pop()
            self.update()

    def finish_polygon(self):
        if not self.draw_polygon or len(self.polygon_points) < 3:
            return False
        self.polygon = tuple(self.polygon_points)
        self.draw_polygon, self.draw_roi = False, True
        self.polygon_points = []
        self.polygonPicked.emit(self.polygon)
        self.update()
        return True

    def clear_polygon(self):
        self.polygon, self.polygon_points = None, []
        self.draw_polygon, self.draw_roi = False, True
        self.polygonPicked.emit(None)
        self.update()

    def mouseDoubleClickEvent(self, event):
        if self.draw_polygon:
            self.finish_polygon()

    def mousePressEvent(self, event):
        if self.draw_polygon and self.image is not None and self.display_rect().contains(event.position()):
            point = self.image_point(event.position())
            if not self.polygon_points or point != self.polygon_points[-1]:
                self.polygon_points.append(point)
            self.update()
            return
        if self.draw_roi and self.image is not None and self.display_rect().contains(event.position()):
            self.anchor = self.cursor_point = self.image_point(event.position())

    def mouseMoveEvent(self, event):
        if self.anchor is not None:
            self.cursor_point = self.image_point(event.position())
            self.update()

    def mouseReleaseEvent(self, event):
        if self.anchor is not None:
            end = self.image_point(event.position())
            x, y = min(self.anchor[0], end[0]), min(self.anchor[1], end[1])
            w, h = abs(self.anchor[0] - end[0]) + 1, abs(self.anchor[1] - end[1]) + 1
            self.anchor = self.cursor_point = None
            if min(w, h) >= 20:
                self.roi = (x, y, w, h)
                self.roiPicked.emit(self.roi)
            self.update()


def path_row(field, callback, button_text="Browse…"):
    holder = QWidget()
    row = QHBoxLayout(holder)
    row.setContentsMargins(0, 0, 0, 0)
    row.addWidget(field, 1)
    button = QPushButton(button_text)
    button.clicked.connect(callback)
    row.addWidget(button)
    return holder


def pair(first, second):
    panel = QWidget()
    layout = QHBoxLayout(panel)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(first)
    layout.addWidget(second)
    return panel


def style_application(app):
    app.setStyle("Fusion")
    app.setStyleSheet("""
        QWidget { background: #152238; color: #e5edf6; font-family: Segoe UI; font-size: 12px; }
        QLabel#title { font-size: 26px; font-weight: 600; color: #65dec2; }
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QTableWidget {
            background: #0c1727; border: 1px solid #39506c; border-radius: 4px; padding: 5px; }
        QPushButton { background: #29415e; border: 1px solid #496480; border-radius: 5px; padding: 8px; }
        QPushButton:hover { background: #37577a; }
        QPushButton#primary { background: #197e6d; font-weight: 600; }
        QPushButton:disabled { color: #6e8097; background: #233248; }
        QTabBar::tab { padding: 10px 18px; background: #1d304b; }
        QTabBar::tab:selected { background: #294762; }
        QHeaderView::section { background: #263c56; padding: 6px; }
        QProgressBar { border: 1px solid #39506c; border-radius: 4px; text-align: center; }
        QProgressBar::chunk { background: #269c86; }
    """)
