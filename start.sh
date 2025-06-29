#!/bin/bash

echo "🚀 Starting AI Career Model API"
echo "================================"

# Test model loading first
echo "🧪 Testing model loading..."
python test_model_loading.py

if [ $? -eq 0 ]; then
    echo "✅ Model test passed - starting server"
    echo "🌐 Starting Gunicorn server..."
    exec gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 2 --timeout 120 --preload app:app
else
    echo "❌ Model test failed - check logs above"
    exit 1
fi
