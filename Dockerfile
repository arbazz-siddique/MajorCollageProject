FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama inside container
RUN curl -fsSL https://ollama.com/install.sh | sh

# Create app directory
WORKDIR /app

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install -r requirements.txt

# Expose Streamlit + Ollama ports
EXPOSE 8501
EXPOSE 11434

# Start both services together
CMD ollama serve & streamlit run app.py --server.port=8501 --server.address=0.0.0.0
