#!/usr/bin/env python3
"""
Main application for Render deployment
"""

import os
import threading
import asyncio
import logging
from formant_server import app, websocket_main

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_websocket_server():
    """WebSocketサーバーを別スレッドで実行"""
    try:
        logger.info("Starting WebSocket server thread")
        asyncio.run(websocket_main())
    except Exception as e:
        logger.error(f"WebSocket server error: {e}")
        raise

if __name__ == '__main__':
    logger.info("Starting Praat Formant Extraction Server")
    
    # Render環境では、WebSocketサーバーとFlaskサーバーを同じポートで実行
    # WebSocketサーバーを別スレッドで開始
    websocket_thread = threading.Thread(target=run_websocket_server, daemon=True)
    websocket_thread.start()
    logger.info("WebSocket server thread started")

    # Flaskサーバーをメインスレッドで実行
    port = int(os.environ.get('PORT', 5001))
    logger.info(f"Starting Flask server on port {port}")
    logger.info("Server ready to accept connections")
    
    try:
        app.run(host='0.0.0.0', port=port, debug=False)
    except Exception as e:
        logger.error(f"Flask server error: {e}")
        raise