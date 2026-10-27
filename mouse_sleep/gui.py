"""Desktop entry point and compatibility imports."""
import sys

from PySide6.QtWidgets import QApplication

from .ui.tasks import Worker
from .ui.widgets import ImageView, clock_text, style_application
from .ui.window import Window

__all__ = ["ImageView", "Worker", "Window", "clock_text", "style_application", "main"]


def main():
    app = QApplication(sys.argv)
    style_application(app)
    window = Window()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
