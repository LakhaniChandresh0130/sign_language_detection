FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    python3-opencv \
    libv4l-dev \
    && pip install numpy tensorflow
# Copy only requirements first to leverage Docker cache
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files
COPY . .

# Mount dataset, checkpoints, and predictions folders
VOLUME ["/app/src", "/app/checkpoints", "/app/predictions"]

# Expose Gradio's default port
EXPOSE 7860

# Define the command to run the Gradio UI
CMD ["python", "app.py"]
