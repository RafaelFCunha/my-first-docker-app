# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copying source folder containing model training main.py
COPY src/ .

# Copying requirements.txt
COPY requirements.txt .

# Install any needed dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Run your Python script when the container launches
CMD ["python", "main.py"]
