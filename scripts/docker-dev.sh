#!/bin/bash
# Docker Development Helper Script for AI Scene-to-Video Pipeline

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project info
PROJECT_NAME="ContentCreator AI Pipeline"
CONTAINER_NAME="contentcreator-pipeline"
DEV_CONTAINER_NAME="contentcreator-pipeline-dev"

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" 
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

show_help() {
    echo -e "${BLUE}$PROJECT_NAME - Docker Development Helper${NC}"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  build       Build the Docker image"
    echo "  dev         Start development container (interactive)"
    echo "  prod        Start production container"
    echo "  stop        Stop all containers"
    echo "  clean       Clean up containers and images"
    echo "  logs        Show container logs" 
    echo "  shell       Open shell in running container"
    echo "  test        Run tests in container"
    echo "  status      Show container status"
    echo "  env-check   Check environment variables"
    echo ""
    echo "Options:"
    echo "  -h, --help  Show this help message"
    echo "  -v          Verbose output"
    echo ""
    echo "Examples:"
    echo "  $0 build              # Build Docker image"
    echo "  $0 dev                # Start development environment"
    echo "  $0 test               # Run tests in container"
    echo "  $0 shell              # Open interactive shell"
}

check_env() {
    log_info "Checking environment setup..."
    
    if [ ! -f ".env" ]; then
        log_warning ".env file not found!"
        log_info "Creating .env from template..."
        if [ -f ".env.example" ]; then
            cp .env.example .env
            log_info "Please edit .env file with your actual API keys"
        else
            log_error ".env.example not found!"
            exit 1
        fi
    fi
    
    # Check for required API keys
    if grep -q "your_.*_api_key_here" .env; then
        log_warning "Please update .env with your actual API keys"
    fi
    
    log_success "Environment check complete"
}

build_image() {
    log_info "Building Docker image..."
    docker build -t contentcreator-pipeline:latest .
    log_success "Docker image built successfully"
}

start_dev() {
    log_info "Starting development container..."
    check_env
    docker-compose --profile dev up -d ai-scene-pipeline-dev
    log_success "Development container started"
    log_info "Use '$0 shell' to access the container"
}

start_prod() {
    log_info "Starting production container..."
    check_env
    docker-compose up -d ai-scene-pipeline
    log_success "Production container started"
}

stop_containers() {
    log_info "Stopping all containers..."
    docker-compose down
    log_success "All containers stopped"
}

clean_up() {
    log_info "Cleaning up containers and images..."
    docker-compose down --rmi all --volumes --remove-orphans
    log_success "Cleanup complete"
}

show_logs() {
    log_info "Showing container logs..."
    if docker ps | grep -q $DEV_CONTAINER_NAME; then
        docker logs -f $DEV_CONTAINER_NAME
    elif docker ps | grep -q $CONTAINER_NAME; then
        docker logs -f $CONTAINER_NAME
    else
        log_error "No containers running"
        exit 1
    fi
}

open_shell() {
    log_info "Opening shell in container..."
    if docker ps | grep -q $DEV_CONTAINER_NAME; then
        docker exec -it $DEV_CONTAINER_NAME /bin/bash
    elif docker ps | grep -q $CONTAINER_NAME; then
        docker exec -it $CONTAINER_NAME /bin/bash
    else
        log_error "No containers running. Start a container first with '$0 dev' or '$0 prod'"
        exit 1
    fi
}

run_tests() {
    log_info "Running tests in container..."
    if docker ps | grep -q $DEV_CONTAINER_NAME; then
        docker exec $DEV_CONTAINER_NAME python -m pytest src/tests/ -v
    elif docker ps | grep -q $CONTAINER_NAME; then
        docker exec $CONTAINER_NAME python -m pytest src/tests/ -v
    else
        log_error "No containers running. Start a container first."
        exit 1
    fi
}

show_status() {
    log_info "Container status:"
    docker ps --filter "name=contentcreator" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
}

env_check() {
    log_info "Environment variables check:"
    if [ -f ".env" ]; then
        echo "Environment file exists: ✅"
        echo "Variables defined:"
        grep -E "^[A-Z_]+" .env | sed 's/=.*/=***/' | while read line; do
            echo "  $line"
        done
    else
        echo "Environment file exists: ❌"
    fi
}

# Main script logic
case "$1" in
    build)
        build_image
        ;;
    dev)
        start_dev
        ;;
    prod)
        start_prod
        ;;
    stop)
        stop_containers
        ;;
    clean)
        clean_up
        ;;
    logs)
        show_logs
        ;;
    shell)
        open_shell
        ;;
    test)
        run_tests
        ;;
    status)
        show_status
        ;;
    env-check)
        env_check
        ;;
    -h|--help|help)
        show_help
        ;;
    "")
        show_help
        ;;
    *)
        log_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac 