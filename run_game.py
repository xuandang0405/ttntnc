"""Start the local, built game and open it when the server is ready."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import threading
import time
from urllib.error import URLError
from urllib.request import urlopen
import webbrowser

ROOT = Path(__file__).resolve().parent


def is_game_running(url: str) -> bool:
    try:
        with urlopen(f'{url}/health', timeout=0.5) as response:
            data = json.load(response)
        return isinstance(data, dict) and data.get('status') == 'ok' and (data.get('app')=='co-vua' or data.get('algorithm')=='Random' and data.get('levels')==[1])
    except (URLError, OSError, ValueError):
        return False


def open_when_ready(url: str) -> None:
    for _ in range(60):
        if is_game_running(url):
            webbrowser.open(url)
            return
        time.sleep(0.25)


def main() -> int:
    parser = argparse.ArgumentParser(description='Co Vua - local chess game')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('--port must be between 1 and 65535')
    os.chdir(ROOT)
    if not (ROOT / 'frontend/dist/index.html').is_file():
        print('Chua co frontend build. Hay chay CAI_DAT.bat truoc.')
        return 1
    url = f'http://127.0.0.1:{args.port}'
    if is_game_running(url):
        print(f'Game dang chay: {url}')
        if not args.no_browser:
            webbrowser.open(url)
        return 0
    try:
        import uvicorn
    except ImportError:
        print('Chua cai thu vien Python. Hay chay CAI_DAT.bat truoc.')
        return 1
    if not args.no_browser:
        threading.Thread(target=open_when_ready, args=(url,), daemon=True).start()
    print(f'Co Vua: {url} | API docs: {url}/docs | Ctrl+C de dung.')
    uvicorn.run('backend.api.server:app', host='127.0.0.1', port=args.port)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
