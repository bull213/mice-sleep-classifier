"""An OS-owned output lock, automatically released after cancellation or crash."""
from pathlib import Path
import os


class OutputLock:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.handle = None

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        self.handle = (self.directory / ".analysis.lock").open("a+b")
        if self.handle.seek(0, os.SEEK_END) == 0:
            self.handle.write(b"0")
            self.handle.flush()
        self.handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.handle.close()
            self.handle = None
            raise ValueError("Another analysis is using this output folder. Wait for it to stop or select a different folder.") from exc
        return self

    def __exit__(self, *args):
        if self.handle is not None:
            self.handle.close()
            self.handle = None
