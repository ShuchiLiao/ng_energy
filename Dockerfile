# Stage 1: Build stage
FROM python:3.11-slim AS build-stage

# Maintainer info
LABEL maintainer="liaoshuchi123@gmail.com"

# Set the working directory in the container
RUN mkdir -p /ng_eu
WORKDIR /ng_eu

# Copy the current directory contents into the container at /ng_eu
COPY . /ng_eu

# Copy requirements.txt and install dependencies
COPY requirements.txt /ng_eu/

# Install any needed packages specified in requirements.txt
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        && pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu \
        && apt-get purge -y build-essential \
        && apt-get autoremove -y \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*

# Stage 2: Final stage
FROM python:3.11-slim

# Set the working directory in the container
RUN mkdir -p /ng_eu
WORKDIR /ng_eu

# Copy the current directory contents into the container at /ng_eu
COPY . /ng_eu

# Copy installed dependencies from the build stage
COPY --from=build-stage /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=build-stage /usr/local/bin /usr/local/bin

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
