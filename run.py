"""
GeneCurator launcher.

    python run.py                          # local: http://127.0.0.1:8000, opens the browser
    HOST=0.0.0.0 PORT=8020 python run.py   # server: listens on all interfaces, no browser

Environment variables:
    HOST      interface to listen on (default 127.0.0.1)
    PORT      port (default 8000)
    NO_BROWSER=1   do not open a browser (automatic when HOST is not 127.0.0.1)
"""

import os
import threading
import time
import webbrowser

import uvicorn


HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8000"))
OPEN_BROWSER = HOST in ("127.0.0.1", "localhost") and os.environ.get("NO_BROWSER", "0") != "1"


def _open_browser():
    time.sleep(1.0)
    webbrowser.open(f"http://127.0.0.1:{PORT}/")


if __name__ == "__main__":
    # Relative data paths (data/, kegg_data/, the library .txt files) are
    # resolved from the project folder, wherever the command is run from.
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    if OPEN_BROWSER:
        threading.Thread(target=_open_browser, daemon=True).start()
    uvicorn.run(
        "backend.main:app",
        host=HOST,
        port=PORT,
        reload=False,
        log_level="info",
    )
