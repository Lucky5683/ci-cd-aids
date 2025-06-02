# Use an official Python image
FROM python:3.13

# Set the working directory in the container
WORKDIR /app

# Copy local files to the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Command to run the app
CMD ["python", "app.py"]
