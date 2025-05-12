FROM python:3.12-bullseye

# Install ping and nslookup
RUN apt-get update \
 && apt-get install -y iputils-ping dnsutils \
 && rm -rf /var/lib/apt/lists/*

# Set python alias
RUN ln -s /usr/bin/python3 /usr/bin/python

# Make your app importable
ENV PYTHONPATH=/app
WORKDIR /app

# Copy and install your package (plus dev/docs extras) and ipykernel
COPY pyproject.toml .
RUN pip install --upgrade pip \
 && pip install .[dev,docs] ipykernel

# Install ansible podman plugin
RUN ansible-galaxy collection install containers.podman

# Keep the container running
CMD ["tail", "-f", "/dev/null"]
