# Use an official Python runtime as a parent image
FROM python:3.11

# Install dependencies required to build pyarrow
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Create app directory
WORKDIR /app

# Copy the rest of the application code
COPY . .


RUN pip install --no-cache-dir -r requirements.txt
# Expose the port the app runs on
EXPOSE 8000

# Command to run the FastAPI application using uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
