#!/usr/bin/env python3
"""
Simple startup script for Render deployment
"""

import os
import sys

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the app
from app import app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Starting AI Career Model API on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
