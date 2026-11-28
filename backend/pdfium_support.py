"""Share PDFium's process-wide lock with Docling when installed."""

from threading import Lock

try:
    from docling.utils.locks import pypdfium2_lock
except ImportError:
    pypdfium2_lock = Lock()
