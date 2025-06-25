# 🐳 Docker Containerization Guide
## AI Scene-to-Video Pipeline

This guide covers containerization setup for the AI Scene-to-Video Pipeline using Docker and Docker Compose.

## 📋 Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- Git
- 8GB+ RAM recommended
- API Keys (OpenAI, fal.ai)

## 🚀 Quick Start

### 1. **Environment Setup**
```bash
# Clone and setup environment
git clone https://github.com/KofiRusu/ContentCreator-0.1.git
cd ContentCreator-0.1

# Setup local development environment
./scripts/setup-env.sh

# Edit your API keys
nano .env
```

### 2. **Docker Development**
```bash
# Build the Docker image
./scripts/docker-dev.sh build

# Start development container
./scripts/docker-dev.sh dev

# Open interactive shell
./scripts/docker-dev.sh shell

# Run tests in container
./scripts/docker-dev.sh test
```

## 🔧 Container Configuration

### **Docker Services**

#### **Production Container** (`ai-scene-pipeline`)
- **Purpose**: Production workloads
- **Memory**: 2GB limit, 512MB reserved
- **CPU**: 1.0 limit, 0.5 reserved
- **Volumes**: Read-only source, persistent assets
- **Security**: Non-root user

#### **Development Container** (`ai-scene-pipeline-dev`)
- **Purpose**: Development and testing
- **Memory**: Same as production
- **Features**: Interactive shell, writable source mounts
- **Profile**: `dev` (started with `--profile dev`)

### **Volume Mounts**

| Local Path | Container Path | Mode | Purpose |
|------------|---------------|------|---------|
| `./src` | `/app/src` | ro/rw | Application source code |
| `./config` | `/app/config` | ro | Configuration files |
| `./src/assets` | `/app/src/assets` | rw | Generated assets (persistent) |
| `./stories` | `/app/stories` | ro | Input story files |

### **Environment Variables**

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key for GPT-4o/DALL·E 3 | Required |
| `FAL_API_KEY` | fal.ai API key for Veo 3 | Required |
| `DEFAULT_VIDEO_BUDGET` | Budget limit for video generation | 50.0 |
| `MAX_SCENES` | Maximum scenes to process | 10 |
| `DEBUG` | Enable debug logging | false |
| `LOG_LEVEL` | Logging level | INFO |

## 📁 Directory Structure

```
ContentCreator-0.1/
├── Dockerfile                 # Multi-stage container definition
├── docker-compose.yml        # Service orchestration
├── .dockerignore             # Docker build exclusions
├── .env                      # Environment variables (local)
├── .env.example             # Environment template
├── venv/                    # Python virtual environment
├── scripts/
│   ├── docker-dev.sh        # Docker development helper
│   └── setup-env.sh         # Local environment setup
├── stories/                 # Input story files
│   └── sample_story.txt     # Example story
└── src/assets/              # Generated assets (persistent)
    ├── images/              # Generated images
    ├── videos/              # Generated videos
    └── temp/                # Temporary files
```

## 🛠️ Docker Commands

### **Development Workflow**

```bash
# Full development setup
./scripts/docker-dev.sh build
./scripts/docker-dev.sh dev

# Interactive development
./scripts/docker-dev.sh shell
# Inside container:
python -m src.main generate-media /app/stories/sample_story.txt --dry-run

# Run tests
./scripts/docker-dev.sh test

# View logs
./scripts/docker-dev.sh logs

# Check status
./scripts/docker-dev.sh status
```

### **Production Deployment**

```bash
# Production container
./scripts/docker-dev.sh prod

# Run specific command
docker-compose exec ai-scene-pipeline python -m src.main version

# Generate from story
docker-compose exec ai-scene-pipeline python -m src.main generate-media /app/stories/story.txt
```

### **Container Management**

```bash
# Stop all containers
./scripts/docker-dev.sh stop

# Clean up (removes containers, images, volumes)
./scripts/docker-dev.sh clean

# Environment check
./scripts/docker-dev.sh env-check
```

## 🔐 Security Features

### **Container Security**
- **Non-root user**: Runs as `appuser` (not root)
- **Read-only filesystem**: Source code mounted read-only in production
- **Resource limits**: Memory and CPU constraints
- **Minimal base image**: Python 3.12 slim

### **Environment Security**
- **No secrets in image**: API keys loaded via environment
- **Gitignore protection**: `.env` excluded from version control
- **Docker ignore**: Sensitive files excluded from build context

### **Network Security**
- **Custom network**: Isolated container network
- **No exposed ports**: Unless explicitly configured
- **Internal communication**: Services communicate via Docker network

## 📊 Resource Requirements

### **Minimum System Requirements**
- **RAM**: 4GB available
- **Storage**: 10GB free space
- **CPU**: 2 cores
- **Network**: Stable internet for API calls

### **Recommended System Requirements**
- **RAM**: 8GB+ available
- **Storage**: 20GB+ free space
- **CPU**: 4+ cores
- **Network**: High-speed internet

### **Container Resource Limits**
```yaml
resources:
  limits:
    memory: 2G      # Maximum memory usage
    cpus: '1.0'     # Maximum CPU usage
  reservations:
    memory: 512M    # Guaranteed memory
    cpus: '0.5'     # Guaranteed CPU
```

## 🧪 Testing in Containers

### **Run All Tests**
```bash
./scripts/docker-dev.sh test
```

### **Run Specific Tests**
```bash
# CLI tests
docker-compose exec ai-scene-pipeline-dev python -m pytest src/tests/test_main_cli.py -v

# Image generation tests
docker-compose exec ai-scene-pipeline-dev python -m pytest src/tests/test_image_gen.py -v

# Video generation tests  
docker-compose exec ai-scene-pipeline-dev python -m pytest src/tests/test_video_gen.py -v
```

### **Interactive Testing**
```bash
./scripts/docker-dev.sh shell
# Inside container:
python -m src.main test
python -m src.main generate-media /app/stories/sample_story.txt --dry-run
```

## 🚀 Production Deployment

### **Docker Compose Production**
```bash
# Production environment
cp .env.example .env.prod
# Edit .env.prod with production values

# Deploy with production env
docker-compose --env-file .env.prod up -d ai-scene-pipeline
```

### **Kubernetes Deployment** (Future)
```yaml
# Example Kubernetes deployment (not included yet)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: contentcreator-pipeline
spec:
  replicas: 1
  selector:
    matchLabels:
      app: contentcreator-pipeline
  template:
    metadata:
      labels:
        app: contentcreator-pipeline
    spec:
      containers:
      - name: pipeline
        image: contentcreator-pipeline:latest
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-keys
              key: openai-key
```

## 🔍 Troubleshooting

### **Common Issues**

#### **API Key Problems**
```bash
# Check environment variables
./scripts/docker-dev.sh env-check

# Verify API keys in container
docker-compose exec ai-scene-pipeline-dev env | grep API_KEY
```

#### **Permission Issues**
```bash
# Fix asset directory permissions
sudo chown -R $USER:$USER src/assets/

# Rebuild with clean slate
./scripts/docker-dev.sh clean
./scripts/docker-dev.sh build
```

#### **Memory Issues**
```bash
# Check container resource usage
docker stats contentcreator-pipeline-dev

# Increase Docker memory limits in Docker Desktop
# Settings > Resources > Advanced > Memory
```

#### **Build Issues**
```bash
# Clean build cache
docker builder prune

# Build with no cache
docker build --no-cache -t contentcreator-pipeline:latest .
```

### **Debug Mode**
```bash
# Enable debug logging
echo "DEBUG=true" >> .env
echo "LOG_LEVEL=DEBUG" >> .env

# Restart with debug
./scripts/docker-dev.sh stop
./scripts/docker-dev.sh dev
```

## 📈 Performance Tuning

### **Container Optimization**
- **Multi-stage builds**: Minimize final image size
- **Layer caching**: Optimized Dockerfile layer order
- **Resource limits**: Prevent resource exhaustion
- **Health checks**: Monitor container health

### **Volume Performance**
- **Bind mounts**: Fast development iteration
- **Named volumes**: Persistent data storage
- **tmpfs mounts**: Temporary file performance

### **Network Performance**
- **Custom networks**: Reduced latency
- **DNS caching**: Faster API resolution
- **Connection pooling**: Reuse HTTP connections

## 🆘 Support

### **Get Help**
- **GitHub Issues**: Report bugs and feature requests
- **Documentation**: Check README.md and QUICK_START.md
- **Container Logs**: `./scripts/docker-dev.sh logs`
- **Debug Mode**: Enable debugging for detailed output

### **Useful Commands**
```bash
# Container information
docker inspect contentcreator-pipeline-dev

# Resource usage
docker stats

# Network information
docker network ls
docker network inspect contentcreator-network

# Volume information
docker volume ls
docker volume inspect contentcreator-0.1_assets_data
``` 