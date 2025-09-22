#!/usr/bin/env python3
"""
Main application for Render deployment
"""

import os
from formant_server import app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)