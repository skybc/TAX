# Industrial Defect Segmentation System - Deployment Guide

**Version**: 1.0.0  
**Last Updated**: December 23, 2025  
**Target Audience**: DevOps Engineers, System Administrators

---

## Table of Contents

1. [Overview](#overview)
2. [Deployment Options](#deployment-options)
3. [Docker Deployment](#docker-deployment)
4. [Server Deployment](#server-deployment)
5. [Production Configuration](#production-configuration)
6. [Monitoring & Logging](#monitoring--logging)
7. [Scaling](#scaling)
8. [Troubleshooting](#troubleshooting)

---

## Overview

### Deployment Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Load Balancer                     │
│                 (Nginx / HAProxy)                   │
└──────────────┬──────────────────────┬───────────────┘
               │                      │
       ┌───────┴────────┐    ┌───────┴────────┐
       │  Application 1  │    │  Application 2  │
       │   (GPU Server)  │    │   (GPU Server)  │
       └───────┬────────┘    └───────┬────────┘
               │                      │
         ┌─────┴──────────────────────┴─────┐
         │          Shared Storage           │
         │    (NFS / S3 / Cloud Storage)     │
         └───────────────────────────────────┘
```

### System Requirements

#### Minimum Requirements (Development/Testing)
- **OS**: Ubuntu 20.04+ / Windows 10+ / macOS 11+
- **CPU**: Intel Core i5 or equivalent (4 cores)
- **RAM**: 8 GB
- **GPU**: Optional (CUDA-compatible with 4GB VRAM)
- **Storage**: 10 GB free space

#### Recommended Requirements (Production)
- **OS**: Ubuntu 22.04 LTS Server
- **CPU**: Intel Xeon or AMD EPYC (8+ cores)
- **RAM**: 16-32 GB
- **GPU**: NVIDIA RTX 3070+ or Tesla T4+ (8GB+ VRAM)
- **Storage**: 50 GB SSD + additional storage for data

### Network Requirements
- **Inbound**: Port 5000 (application), 22 (SSH), 443 (HTTPS)
- **Outbound**: Internet access for downloading models and dependencies
- **Bandwidth**: 100 Mbps+ recommended for image upload/download

---

## Deployment Options

### Option 1: Docker Deployment (Recommended)

**Advantages**:
- Consistent environment across platforms
- Easy scaling and orchestration
- Isolated dependencies
- Simple rollback and updates

**Use Cases**:
- Production deployments
- Multi-instance deployments
- Cloud deployments (AWS, Azure, GCP)

### Option 2: Direct Server Deployment

**Advantages**:
- Full control over environment
- Direct hardware access
- Lower overhead

**Use Cases**:
- Single-server deployments
- Legacy infrastructure
- Custom hardware configurations

### Option 3: Kubernetes Deployment

**Advantages**:
- Auto-scaling
- High availability
- Load balancing
- Self-healing

**Use Cases**:
- Large-scale deployments
- Multi-region deployments
- Enterprise environments

---

## Docker Deployment

### Prerequisites

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo apt-get install docker-compose

# Install NVIDIA Container Toolkit (for GPU support)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Build Docker Image

**Dockerfile** (located at project root):
```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Download SAM weights (if not included)
RUN mkdir -p models/checkpoints && \
    cd models/checkpoints && \
    wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O sam_vit_h.pth

# Set environment variables
ENV PYTHONPATH=/app
ENV QT_QPA_PLATFORM=offscreen

# Expose port (if web interface is added)
EXPOSE 5000

# Run application
CMD ["python3", "src/main.py"]
```

**Build the image**:
```bash
# Build image
docker build -t industrial-defect-seg:latest .

# Tag for registry (optional)
docker tag industrial-defect-seg:latest your-registry/industrial-defect-seg:latest
```

### Run Container

#### CPU-only Deployment

```bash
docker run -d \
  --name defect-seg-app \
  -p 5000:5000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/outputs:/app/data/outputs \
  -e DEVICE=cpu \
  industrial-defect-seg:latest
```

#### GPU Deployment

```bash
docker run -d \
  --name defect-seg-app \
  --gpus all \
  -p 5000:5000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/outputs:/app/data/outputs \
  -e DEVICE=cuda \
  -e CUDA_VISIBLE_DEVICES=0 \
  industrial-defect-seg:latest
```

#### Production Deployment with Docker Compose

**docker-compose.yml**:
```yaml
version: '3.8'

services:
  app:
    image: industrial-defect-seg:latest
    container_name: defect-seg-app
    restart: unless-stopped
    
    # GPU support
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    
    # Environment variables
    environment:
      - DEVICE=cuda
      - CUDA_VISIBLE_DEVICES=0
      - LOG_LEVEL=INFO
      - MAX_WORKERS=4
    
    # Volumes
    volumes:
      - ./data:/app/data
      - ./outputs:/app/data/outputs
      - ./logs:/app/logs
      - ./config:/app/config
    
    # Ports
    ports:
      - "5000:5000"
    
    # Health check
    healthcheck:
      test: ["CMD", "python3", "-c", "import torch; print(torch.cuda.is_available())"]
      interval: 30s
      timeout: 10s
      retries: 3
    
    # Logging
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
  
  # Optional: Monitoring with Prometheus
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    restart: unless-stopped
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    ports:
      - "9090:9090"
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
  
  # Optional: Visualization with Grafana
  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    restart: unless-stopped
    volumes:
      - grafana-data:/var/lib/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin123

volumes:
  prometheus-data:
  grafana-data:
```

**Start services**:
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f app

# Stop services
docker-compose down

# Update and restart
docker-compose pull
docker-compose up -d
```

---

## Server Deployment

### Prerequisites

```bash
# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install Python 3.10+
sudo apt-get install python3.10 python3.10-venv python3-pip

# Install CUDA (for GPU support)
# Download from https://developer.nvidia.com/cuda-downloads
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda

# Verify CUDA installation
nvidia-smi
```

### Application Setup

```bash
# Create application directory
sudo mkdir -p /opt/industrial-defect-seg
sudo chown -R $USER:$USER /opt/industrial-defect-seg
cd /opt/industrial-defect-seg

# Clone repository
git clone https://github.com/your-org/industrial-defect-seg.git .

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Download SAM weights
mkdir -p models/checkpoints
cd models/checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O sam_vit_h.pth
cd ../..

# Create necessary directories
mkdir -p data/{raw,processed,outputs}/{images,masks,annotations}
mkdir -p logs
```

### Systemd Service

Create service file `/etc/systemd/system/defect-seg.service`:

```ini
[Unit]
Description=Industrial Defect Segmentation System
After=network.target

[Service]
Type=simple
User=www-data
Group=www-data
WorkingDirectory=/opt/industrial-defect-seg
Environment="PATH=/opt/industrial-defect-seg/venv/bin"
Environment="PYTHONPATH=/opt/industrial-defect-seg"
Environment="DEVICE=cuda"
ExecStart=/opt/industrial-defect-seg/venv/bin/python src/main.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=defect-seg

# Security settings
NoNewPrivileges=true
PrivateTmp=true

# Resource limits
LimitNOFILE=65536
MemoryLimit=16G

[Install]
WantedBy=multi-user.target
```

**Enable and start service**:
```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable service
sudo systemctl enable defect-seg

# Start service
sudo systemctl start defect-seg

# Check status
sudo systemctl status defect-seg

# View logs
sudo journalctl -u defect-seg -f
```

### Nginx Reverse Proxy

Install Nginx:
```bash
sudo apt-get install nginx
```

Create Nginx configuration `/etc/nginx/sites-available/defect-seg`:

```nginx
upstream defect_seg_backend {
    server 127.0.0.1:5000;
    # Add more servers for load balancing
    # server 127.0.0.1:5001;
    # server 127.0.0.1:5002;
}

server {
    listen 80;
    server_name your-domain.com;
    
    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    # SSL certificates
    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;
    
    # SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;
    
    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    
    # Client upload size limit
    client_max_body_size 100M;
    
    # Timeout settings
    proxy_connect_timeout 600s;
    proxy_send_timeout 600s;
    proxy_read_timeout 600s;
    
    # Proxy to application
    location / {
        proxy_pass http://defect_seg_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # Static files
    location /static {
        alias /opt/industrial-defect-seg/static;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }
    
    # API endpoint (if implemented)
    location /api {
        proxy_pass http://defect_seg_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    # Health check
    location /health {
        access_log off;
        return 200 "healthy\n";
        add_header Content-Type text/plain;
    }
}
```

**Enable site**:
```bash
# Create symbolic link
sudo ln -s /etc/nginx/sites-available/defect-seg /etc/nginx/sites-enabled/

# Test configuration
sudo nginx -t

# Reload Nginx
sudo systemctl reload nginx
```

### SSL Certificate (Let's Encrypt)

```bash
# Install Certbot
sudo apt-get install certbot python3-certbot-nginx

# Obtain certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal (already configured by certbot)
sudo certbot renew --dry-run
```

---

## Production Configuration

### Configuration Files

**config/production.yaml**:
```yaml
app:
  name: "Industrial Defect Segmentation System"
  version: "1.0.0"
  environment: "production"
  debug: false

device:
  type: "cuda"
  gpu_id: 0

sam:
  model_type: "vit_h"
  checkpoint: "models/checkpoints/sam_vit_h.pth"
  cache_embeddings: true

logging:
  level: "INFO"
  file: "logs/app.log"
  console: false
  max_file_size: "100MB"
  backup_count: 10

performance:
  max_cache_size: 4096
  num_workers: 4
  prefetch_factor: 2
  batch_size: 8

security:
  allowed_origins: ["https://your-domain.com"]
  max_upload_size: 104857600  # 100 MB
  rate_limit: 100  # requests per minute
```

### Environment Variables

Create `.env` file:
```bash
# Application
APP_ENV=production
DEBUG=false

# Device
DEVICE=cuda
CUDA_VISIBLE_DEVICES=0

# Paths
DATA_ROOT=/opt/industrial-defect-seg/data
MODELS_DIR=/opt/industrial-defect-seg/models
LOGS_DIR=/opt/industrial-defect-seg/logs

# Database (if applicable)
DATABASE_URL=postgresql://user:password@localhost:5432/defect_seg

# API Keys (if applicable)
API_KEY=your-api-key-here

# Monitoring
SENTRY_DSN=https://your-sentry-dsn
```

### Security Best Practices

1. **File Permissions**:
```bash
# Set correct ownership
sudo chown -R www-data:www-data /opt/industrial-defect-seg

# Restrict permissions
sudo chmod 750 /opt/industrial-defect-seg
sudo chmod 640 /opt/industrial-defect-seg/.env
sudo chmod 640 /opt/industrial-defect-seg/config/*.yaml
```

2. **Firewall Configuration**:
```bash
# Enable UFW
sudo ufw enable

# Allow SSH
sudo ufw allow 22/tcp

# Allow HTTP/HTTPS
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Check status
sudo ufw status
```

3. **Fail2Ban** (protect against brute force):
```bash
# Install
sudo apt-get install fail2ban

# Configure
sudo cp /etc/fail2ban/jail.conf /etc/fail2ban/jail.local

# Enable and start
sudo systemctl enable fail2ban
sudo systemctl start fail2ban
```

---

## Monitoring & Logging

### Application Logging

**Log Configuration** (in `src/logger.py`):
```python
import logging
from logging.handlers import RotatingFileHandler

def setup_production_logger():
    logger = logging.getLogger('IndustrialDefectSeg')
    logger.setLevel(logging.INFO)
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        'logs/app.log',
        maxBytes=100 * 1024 * 1024,  # 100 MB
        backupCount=10
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(file_handler)
    
    # Error file handler
    error_handler = RotatingFileHandler(
        'logs/error.log',
        maxBytes=50 * 1024 * 1024,  # 50 MB
        backupCount=5
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s\n%(exc_info)s'
    ))
    logger.addHandler(error_handler)
    
    return logger
```

### System Monitoring

#### Prometheus Metrics

Create `monitoring/prometheus.yml`:
```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'defect-seg-app'
    static_configs:
      - targets: ['localhost:5000']
    metrics_path: '/metrics'
  
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['localhost:9100']
```

**Install Node Exporter**:
```bash
# Download
wget https://github.com/prometheus/node_exporter/releases/download/v1.6.1/node_exporter-1.6.1.linux-amd64.tar.gz

# Extract
tar xvfz node_exporter-1.6.1.linux-amd64.tar.gz

# Run as service
sudo cp node_exporter-1.6.1.linux-amd64/node_exporter /usr/local/bin/
sudo useradd -rs /bin/false node_exporter

# Create systemd service
sudo cat > /etc/systemd/system/node_exporter.service <<EOF
[Unit]
Description=Node Exporter
After=network.target

[Service]
User=node_exporter
Group=node_exporter
Type=simple
ExecStart=/usr/local/bin/node_exporter

[Install]
WantedBy=multi-user.target
EOF

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable node_exporter
sudo systemctl start node_exporter
```

#### Grafana Dashboard

1. **Access Grafana**: `http://your-server:3000` (admin/admin123)

2. **Add Prometheus data source**:
   - Configuration → Data Sources → Add data source → Prometheus
   - URL: `http://prometheus:9090`

3. **Import dashboard**:
   - Dashboards → Import → Dashboard ID: 1860 (Node Exporter Full)

### Log Aggregation (ELK Stack)

**Optional: Elasticsearch, Logstash, Kibana**

**docker-compose.elk.yml**:
```yaml
version: '3.8'

services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.10.0
    environment:
      - discovery.type=single-node
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
    ports:
      - "9200:9200"
    volumes:
      - elasticsearch-data:/usr/share/elasticsearch/data
  
  logstash:
    image: docker.elastic.co/logstash/logstash:8.10.0
    volumes:
      - ./monitoring/logstash.conf:/usr/share/logstash/pipeline/logstash.conf
    ports:
      - "5044:5044"
    depends_on:
      - elasticsearch
  
  kibana:
    image: docker.elastic.co/kibana/kibana:8.10.0
    ports:
      - "5601:5601"
    environment:
      ELASTICSEARCH_HOSTS: http://elasticsearch:9200
    depends_on:
      - elasticsearch

volumes:
  elasticsearch-data:
```

---

## Scaling

### Horizontal Scaling

#### Load Balancer Configuration

**HAProxy** (`/etc/haproxy/haproxy.cfg`):
```haproxy
global
    log /dev/log local0
    maxconn 4096
    
defaults
    log global
    mode http
    option httplog
    option dontlognull
    timeout connect 5000
    timeout client  50000
    timeout server  50000

frontend http_front
    bind *:80
    default_backend servers

backend servers
    balance roundrobin
    option httpchk GET /health
    server server1 192.168.1.101:5000 check
    server server2 192.168.1.102:5000 check
    server server3 192.168.1.103:5000 check
```

#### Kubernetes Deployment

**k8s/deployment.yaml**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: defect-seg-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: defect-seg
  template:
    metadata:
      labels:
        app: defect-seg
    spec:
      containers:
      - name: defect-seg
        image: your-registry/industrial-defect-seg:latest
        ports:
        - containerPort: 5000
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "16Gi"
            cpu: "4"
          requests:
            nvidia.com/gpu: 1
            memory: "8Gi"
            cpu: "2"
        env:
        - name: DEVICE
          value: "cuda"
        - name: LOG_LEVEL
          value: "INFO"
        volumeMounts:
        - name: data-volume
          mountPath: /app/data
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: defect-seg-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: defect-seg-service
spec:
  selector:
    app: defect-seg
  ports:
  - protocol: TCP
    port: 80
    targetPort: 5000
  type: LoadBalancer
```

**Deploy to Kubernetes**:
```bash
# Apply deployment
kubectl apply -f k8s/deployment.yaml

# Check status
kubectl get pods
kubectl get services

# Scale replicas
kubectl scale deployment defect-seg-deployment --replicas=5
```

### Vertical Scaling

**Increase resources per instance**:

1. **Memory**:
```yaml
# In docker-compose.yml
services:
  app:
    deploy:
      resources:
        limits:
          memory: 32G
        reservations:
          memory: 16G
```

2. **GPU allocation**:
```bash
# Multiple GPUs
docker run --gpus '"device=0,1"' ...

# Or in docker-compose.yml
environment:
  - CUDA_VISIBLE_DEVICES=0,1
```

---

## Troubleshooting

### Common Issues

#### 1. Out of Memory

**Symptom**: Application crashes with OOM error

**Solutions**:
```bash
# Reduce batch size
# In config/production.yaml
performance:
  batch_size: 4  # Reduce from 8

# Enable swap (temporary)
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Monitor memory
watch -n 1 free -h
```

#### 2. GPU Not Available

**Symptom**: CUDA errors or falling back to CPU

**Solutions**:
```bash
# Check GPU
nvidia-smi

# Check CUDA in container
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Verify NVIDIA container toolkit
dpkg -l | grep nvidia-container-toolkit

# Restart Docker
sudo systemctl restart docker
```

#### 3. High Latency

**Symptom**: Slow inference or processing

**Solutions**:
```bash
# Enable model caching
# In config
sam:
  cache_embeddings: true

# Use smaller model
sam:
  model_type: "vit_b"  # Instead of vit_h

# Optimize Nginx
# In /etc/nginx/nginx.conf
worker_processes auto;
worker_connections 4096;
```

#### 4. Service Won't Start

**Symptom**: Systemd service fails

**Debug**:
```bash
# Check status
sudo systemctl status defect-seg

# View logs
sudo journalctl -u defect-seg -n 100

# Check permissions
ls -la /opt/industrial-defect-seg

# Test manually
sudo -u www-data /opt/industrial-defect-seg/venv/bin/python src/main.py
```

### Health Checks

**Script** (`scripts/health_check.sh`):
```bash
#!/bin/bash

# Check application health
APP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:5000/health)

if [ "$APP_STATUS" != "200" ]; then
    echo "Application unhealthy (HTTP $APP_STATUS)"
    # Restart service
    sudo systemctl restart defect-seg
    # Send alert
    echo "Application restarted at $(date)" | mail -s "Defect Seg Alert" admin@your-domain.com
else
    echo "Application healthy"
fi

# Check GPU
GPU_STATUS=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
if [ "$GPU_STATUS" -gt 95 ]; then
    echo "Warning: GPU utilization at ${GPU_STATUS}%"
fi

# Check disk space
DISK_USAGE=$(df -h /opt/industrial-defect-seg | awk 'NR==2 {print $5}' | sed 's/%//')
if [ "$DISK_USAGE" -gt 90 ]; then
    echo "Warning: Disk usage at ${DISK_USAGE}%"
fi
```

**Add to cron** (`crontab -e`):
```cron
*/5 * * * * /opt/industrial-defect-seg/scripts/health_check.sh >> /var/log/health_check.log 2>&1
```

### Backup & Recovery

**Backup Script** (`scripts/backup.sh`):
```bash
#!/bin/bash

BACKUP_DIR="/backup/defect-seg"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup data
tar -czf $BACKUP_DIR/data_$DATE.tar.gz /opt/industrial-defect-seg/data

# Backup models
tar -czf $BACKUP_DIR/models_$DATE.tar.gz /opt/industrial-defect-seg/models

# Backup configuration
tar -czf $BACKUP_DIR/config_$DATE.tar.gz /opt/industrial-defect-seg/config

# Backup database (if applicable)
pg_dump defect_seg > $BACKUP_DIR/database_$DATE.sql

# Remove old backups (keep last 7 days)
find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete
find $BACKUP_DIR -name "*.sql" -mtime +7 -delete

echo "Backup completed: $DATE"
```

**Daily backup** (`crontab -e`):
```cron
0 2 * * * /opt/industrial-defect-seg/scripts/backup.sh
```

---

## Production Checklist

### Pre-Deployment

- [ ] Hardware requirements met
- [ ] CUDA and GPU drivers installed
- [ ] Application tested in staging environment
- [ ] Configuration files reviewed
- [ ] SSL certificates obtained
- [ ] Firewall configured
- [ ] Backup strategy in place
- [ ] Monitoring configured
- [ ] Documentation updated

### Post-Deployment

- [ ] Application accessible via domain
- [ ] SSL certificate valid
- [ ] Health checks passing
- [ ] Logs being collected
- [ ] Monitoring dashboards created
- [ ] Backup script tested
- [ ] Performance benchmarks recorded
- [ ] Team trained on operations

### Maintenance Schedule

- **Daily**: Check logs for errors
- **Weekly**: Review monitoring metrics
- **Monthly**: Update dependencies
- **Quarterly**: Security audit
- **Yearly**: Hardware upgrade assessment

---

**Document Version**: 1.0.0  
**Last Updated**: December 23, 2025  
**Maintainer**: Industrial AI Team

For support, contact: devops@your-domain.com
