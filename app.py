#!/usr/bin/env python3
"""
Main application for Render deployment
"""

import os
import threading
import asyncio
from formant_server import app, websocket_main

def run_websocket_server():
    """WebSocketサーバーを別スレッドで実行"""
    asyncio.run(websocket_main())

if __name__ == '__main__':
    # WebSocketサーバーを別スレッドで開始
    websocket_thread = threading.Thread(target=run_websocket_server, daemon=True)
    websocket_thread.start()

    # Flaskサーバーをメインスレッドで実行
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)