# Use official Python runtime as base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies (if needed, e.g., for pandas)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Train the model during build (optional, or copy the pre-trained ones)
# RUN python train.py

# Expose port
EXPOSE 5000

# Use gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
