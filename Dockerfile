FROM python:3.9-slim

# Set environment variables to optimize Python runtime inside containers
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first for build caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Expose Streamlit's default port
EXPOSE 8501

# Run streamlit app
CMD ["streamlit", "run", "ckd_predictor_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
