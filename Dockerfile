# Use Python 3.11 slim image for smaller size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Verify model files are present and accessible
RUN ls -la *.pkl && \
    python -c "import os; print('Model files found:'); [print(f) for f in os.listdir('.') if f.endswith('.pkl')]"

# Test model loading before deployment (inline)
RUN python -c "
import pickle
import os
print('🧪 Testing model loading...')
model_files = ['improved_quick_career_model.pkl', 'final_career_model.pkl']
loaded = False
for model_file in model_files:
    if os.path.exists(model_file):
        print(f'✅ Found {model_file}')
        try:
            with open(model_file, 'rb') as f:
                model_data = pickle.load(f)
            print(f'✅ Successfully loaded {model_file}')
            print(f'   Careers: {len(model_data.get(\"career_names\", []))}')
            print(f'   Features: {len(model_data.get(\"feature_names\", []))}')
            loaded = True
            break
        except Exception as e:
            print(f'❌ Failed to load {model_file}: {e}')
    else:
        print(f'❌ Missing {model_file}')
if not loaded:
    print('💥 No models could be loaded!')
    exit(1)
else:
    print('🎉 Model loading test passed!')
"

# Create non-root user for security
RUN adduser --disabled-password --gecos '' appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose port (use PORT env var for cloud platforms)
EXPOSE ${PORT:-5000}

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT:-5000}/ || exit 1

# Run the application directly with Gunicorn
CMD ["sh", "-c", "echo '🚀 Starting AI Career Model API...' && gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 2 --timeout 120 --preload app:app"]
