FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy MLflow server script
COPY start.sh .
RUN chmod +x start.sh

# Expose MLflow's default port
EXPOSE 5000

# Start MLflow server
CMD ["./start.sh"] 