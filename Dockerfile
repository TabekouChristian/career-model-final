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

# Make startup script executable
RUN chmod +x start.sh

# Verify model files are present and accessible
RUN ls -la *.pkl && \
    python -c "import os; print('Model files found:'); [print(f) for f in os.listdir('.') if f.endswith('.pkl')]"

# Test model loading before deployment
RUN python test_model_loading.py

# Create non-root user for security
RUN adduser --disabled-password --gecos '' appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose port (use PORT env var for cloud platforms)
EXPOSE ${PORT:-5000}

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT:-5000}/ || exit 1

# Run the application with startup script
CMD ["./start.sh"]
