import subprocess
import sys
import webbrowser
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PYTHON_EXE = sys.executable

if __name__ == "__main__":
    print("=" * 60)
    print("  Starting Campus Smart Count Monitor - Live Dashboard")
    print("=" * 60)
    print(f"Directory: {BASE_DIR}")
    print(f"Python:    {PYTHON_EXE}")
    print("URL:       http://127.0.0.1:5000\n")

    # Open browser automatically after 1.5 seconds
    def open_browser():
        time.sleep(1.5)
        webbrowser.open("http://127.0.0.1:5000")

    import threading
    threading.Thread(target=open_browser, daemon=True).start()

    # Launch app.py
    from app import app, start_worker
    start_worker()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
