# Setup python base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Prevent Python from writing pyc files and keep stdout unbuffered
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies required for building python packages if necessary
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the workspace (ignores those matched in .dockerignore)
COPY . .

# Let DVC build the models and metrics organically inside the container 
# Since models/ are omitted from source control and .dockerignore, this acts as the dynamic build step for the artifact.
RUN dvc repro

# Expose Streamlit default port
EXPOSE 8501

# Command to run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
