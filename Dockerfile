FROM pytorch/pytorch:latest

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-qt5 \
    qt5-default \
    libxcb1 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-xfixes0 \
    libxcb-xinerama0 \
    libxkbcommon-x11-0 \
    x11-utils \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/
WORKDIR /app
RUN pip install -r requirements.txt

# Copy application files
COPY . /app/

# Set display environment variable
ENV DISPLAY=:0

# Set QT platform
ENV QT_QPA_PLATFORM=xcb 