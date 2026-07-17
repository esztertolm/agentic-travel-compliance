FROM python:3.12-slim

# Install system dependencies (needed for compiling certain C++ libraries if required by Chroma/pypdf)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Leverage Docker cache for dependency installation
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

ENV PYTHONPATH="/app"

EXPOSE 8501

# Default command (will be overridden in docker-compose for the database test)
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]