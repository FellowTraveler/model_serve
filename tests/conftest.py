"""Shared test setup: keep the proxy's request logs out of the real request_logs/ folder.

Set before any test module imports harmony_proxy, which creates its RequestLogger at import.
"""
import os
import tempfile

os.environ["REQUEST_LOG_DIR"] = tempfile.mkdtemp(prefix="model_serve_test_request_logs_")
