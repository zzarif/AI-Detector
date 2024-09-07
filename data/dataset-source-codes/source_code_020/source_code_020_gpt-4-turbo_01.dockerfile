# Use an official Python runtime as a parent image, using the slim version for better size optimization
FROM python:3.10-slim

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD hello.py /app

# Define the command to run the app using Python
CMD ["python", "hello.py"]
