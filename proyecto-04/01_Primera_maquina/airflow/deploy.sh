#!/bin/bash

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

check_docker() {
    if ! docker info >/dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker first."
        exit 1
    fi
    print_success "Docker is running"
}

set_env_vars() {
    if [ -f .env ]; then
        export $(cat .env | grep -v '^#' | xargs)
    fi
    
    export DOCKER_USERNAME=${DOCKER_USERNAME:-luisfrontuso10}
    export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-http://10.43.101.175:30500}
    export MLFLOW_S3_ENDPOINT_URL=${MLFLOW_S3_ENDPOINT_URL:-http://10.43.101.175:30382}
    export AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID:-adminuser}
    export AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY:-securepassword123}
    export HOST_IP=${HOST_IP:-10.43.101.175}
    export AIRFLOW_PROJ_DIR=.
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        export AIRFLOW_UID=$(id -u)
        print_success "AIRFLOW_UID set to $AIRFLOW_UID"
    fi
}

pull_images() {
    print_info "Pulling latest images..."
    print_info "Pulling Airflow image: $DOCKER_USERNAME/airflow-houses:latest"
    docker pull $DOCKER_USERNAME/airflow-houses:latest || print_warning "Could not pull custom Airflow image"
    print_success "Images pulled"
}

start_services() {
    print_info "Starting Airflow services..."
    docker-compose up -d
    print_success "Services started"
}

stop_services() {
    print_info "Stopping Airflow services..."
    docker-compose down
    print_success "Services stopped"
}

show_status() {
    print_info "Service status:"
    docker-compose ps
}

show_logs() {
    local service=${1:-}
    if [ -z "$service" ]; then
        docker-compose logs -f
    else
        docker-compose logs -f "$service"
    fi
}

cleanup() {
    print_info "Cleaning up..."
    docker-compose down -v
    docker system prune -f
    print_success "Cleanup completed"
}

wait_for_services() {
    print_info "Waiting for services to be healthy..."
    
    print_info "Waiting for PostgreSQL..."
    until docker-compose exec postgres pg_isready -U airflow >/dev/null 2>&1; do
        sleep 2
    done
    print_success "PostgreSQL is ready"
    
    print_info "Waiting for Airflow webserver..."
    until curl -f http://localhost:8080/health >/dev/null 2>&1; do
        sleep 5
    done
    print_success "Airflow webserver is ready"
}

show_urls() {
    print_info "Services are available at:"
    echo "🌐 Airflow WebUI: http://localhost:8080 (airflow/airflow)"
    echo "🐘 PgAdmin: http://localhost:5050 (admin@example.com/admin)"
}

main() {
    echo -e "${BLUE}🚀 Airflow MLOps Deployment${NC}"
    
    case "${1:-start}" in
        "start"|"up")
            check_docker
            set_env_vars
            create_directories
            pull_images
            start_services
            wait_for_services
            show_urls
            ;;
        "stop"|"down")
            stop_services
            ;;
        "restart")
            stop_services
            sleep 5
            set_env_vars
            start_services
            wait_for_services
            show_urls
            ;;
        "status")
            show_status
            ;;
        "logs")
            show_logs "${2:-}"
            ;;
        "pull")
            set_env_vars
            pull_images
            ;;
        "cleanup")
            cleanup
            ;;
        *)
            echo "Usage: $0 {start|stop|restart|status|logs [service]|pull|cleanup}"
            exit 1
            ;;
    esac
}

main "$@"