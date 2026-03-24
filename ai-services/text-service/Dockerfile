# Dockerfile
# This packages your entire NLP service into a box
# that can run anywhere — your laptop, a server, anywhere.

# Start from a Python environment
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy your dependency list first (for faster builds)
COPY requirements.txt .

# Install all dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code files
COPY predict.py .
COPY main.py .

# Copy your trained model files into the container
# (make sure your model files are in a folder called "model")
COPY model/ ./model/

# Tell Docker which port this service listens on
EXPOSE 8001

# The command that runs when the container starts
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
