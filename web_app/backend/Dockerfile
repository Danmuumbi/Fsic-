# Use Python 3.12
FROM python:3.12-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies your libraries need (libsndfile is for soundfile/librosa)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your code
COPY . .

# Expose the port Render will hit
EXPOSE 10000

# Start Gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:10000"]
