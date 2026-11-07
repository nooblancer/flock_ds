# ==============================================================================
# Production Dockerfile for NILM Disaggregation Engine & Analytics Dashboard
# Base: Debian-slim Python 3.10 runtime
# ==============================================================================

FROM python:3.10-slim

# Prevent interactive prompts and configure optimal Python streaming
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8 \
    MPLCONFIGDIR=/app/outputs/.matplotlib_cache

# Install system runtime dependencies:
# - libgomp1: required for XGBoost OpenMP parallel execution on Linux
# - ca-certificates: for secure HTTPS certificate verification
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set container working directory
WORKDIR /app

# Copy requirements first for Docker layer caching
COPY requirements.txt /app/requirements.txt

# Upgrade pip and install production-locked dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy repository codebase
COPY . /app/

# Pre-create required directory trees for mounted outputs and caches
RUN mkdir -p /app/data \
    /app/outputs/plots \
    /app/outputs/tables \
    /app/outputs/.matplotlib_cache

# Expose port for optional Streamlit dashboard
EXPOSE 8501

# Default entrypoint and master pipeline argument
ENTRYPOINT ["python", "run_pipeline.py"]
CMD ["--all"]
