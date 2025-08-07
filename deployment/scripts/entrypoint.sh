#!/bin/sh
set -e

# Quantum-Inspired Task Scheduler Production Entrypoint
# Handles initialization, configuration validation, and service startup

# Default values
DEFAULT_CONFIG_PATH="/app/config/production.json"
DEFAULT_LOG_LEVEL="INFO"
DEFAULT_PORT="8080"
DEFAULT_WORKERS="4"

# Environment variables with defaults
export QUANTUM_CONFIG_PATH="${QUANTUM_CONFIG_PATH:-$DEFAULT_CONFIG_PATH}"
export QUANTUM_LOG_LEVEL="${QUANTUM_LOG_LEVEL:-$DEFAULT_LOG_LEVEL}"
export QUANTUM_PORT="${QUANTUM_PORT:-$DEFAULT_PORT}"
export QUANTUM_WORKERS="${QUANTUM_WORKERS:-$DEFAULT_WORKERS}"

# Color codes for logging
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1" >&2
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1" >&2
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1" >&2
}

log_debug() {
    if [ "$QUANTUM_DEBUG" = "1" ]; then
        echo -e "${BLUE}[DEBUG]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1" >&2
    fi
}

# Trap signals for graceful shutdown
shutdown() {
    log_info "Received shutdown signal, stopping services..."
    
    # Stop background processes if any
    if [ -n "$MONITOR_PID" ]; then
        kill "$MONITOR_PID" 2>/dev/null || true
    fi
    
    # Stop main process
    if [ -n "$MAIN_PID" ]; then
        kill "$MAIN_PID" 2>/dev/null || true
        wait "$MAIN_PID" 2>/dev/null || true
    fi
    
    log_info "Shutdown complete"
    exit 0
}

trap shutdown TERM INT QUIT

# Validate environment
validate_environment() {
    log_info "Validating environment..."
    
    # Check required directories
    for dir in "$QUANTUM_CACHE_DIR" "$QUANTUM_LOG_DIR" "$QUANTUM_DATA_DIR"; do
        if [ ! -d "$dir" ]; then
            log_info "Creating directory: $dir"
            mkdir -p "$dir"
        fi
        
        if [ ! -w "$dir" ]; then
            log_error "Directory not writable: $dir"
            exit 1
        fi
    done
    
    # Check configuration file
    if [ ! -f "$QUANTUM_CONFIG_PATH" ]; then
        log_warn "Configuration file not found: $QUANTUM_CONFIG_PATH"
        log_info "Creating default configuration..."
        create_default_config
    fi
    
    # Validate Python environment
    if ! python -c "import photon_mlir" 2>/dev/null; then
        log_error "Failed to import photon_mlir package"
        exit 1
    fi
    
    # Validate port
    if ! echo "$QUANTUM_PORT" | grep -E '^[0-9]+$' > /dev/null; then
        log_error "Invalid port number: $QUANTUM_PORT"
        exit 1
    fi
    
    if [ "$QUANTUM_PORT" -lt 1024 ] && [ "$(id -u)" != "0" ]; then
        log_warn "Port $QUANTUM_PORT requires root privileges"
    fi
    
    log_info "Environment validation complete"
}

# Create default configuration
create_default_config() {
    cat > "$QUANTUM_CONFIG_PATH" << EOF
{
  "quantum_scheduler": {
    "optimization_level": "balanced",
    "cache_strategy": "hybrid", 
    "max_workers": ${QUANTUM_WORKERS},
    "enable_validation": true,
    "enable_monitoring": true
  },
  "server": {
    "host": "0.0.0.0",
    "port": ${QUANTUM_PORT},
    "workers": ${QUANTUM_WORKERS},
    "timeout": 300,
    "keepalive": 2,
    "max_requests": 1000,
    "max_requests_jitter": 100
  },
  "logging": {
    "level": "${QUANTUM_LOG_LEVEL}",
    "format": "json",
    "file": "${QUANTUM_LOG_DIR}/quantum-scheduler.log",
    "max_size": "100MB",
    "backup_count": 5,
    "rotation": "time",
    "interval": "D",
    "when": "midnight"
  },
  "security": {
    "enable_authentication": false,
    "enable_tls": false,
    "allowed_origins": ["*"],
    "rate_limiting": {
      "enabled": true,
      "requests_per_minute": 60,
      "burst_size": 10
    }
  },
  "monitoring": {
    "enabled": true,
    "metrics_endpoint": "/metrics",
    "health_endpoint": "/health",
    "interval": 30
  },
  "cache": {
    "directory": "${QUANTUM_CACHE_DIR}",
    "max_size": "1GB",
    "ttl": 3600,
    "cleanup_interval": 300
  }
}
EOF
    
    log_info "Default configuration created at $QUANTUM_CONFIG_PATH"
}

# Run security checks
run_security_checks() {
    log_info "Running security checks..."
    
    # Check file permissions
    check_permissions() {
        file="$1"
        expected_perms="$2"
        actual_perms=$(stat -c '%a' "$file" 2>/dev/null || echo "000")
        
        if [ "$actual_perms" != "$expected_perms" ]; then
            log_warn "Incorrect permissions on $file: $actual_perms (expected: $expected_perms)"
        fi
    }
    
    check_permissions "$QUANTUM_CONFIG_PATH" "644"
    check_permissions "$QUANTUM_LOG_DIR" "755"
    check_permissions "$QUANTUM_CACHE_DIR" "755"
    
    # Run basic security validation
    if command -v python >/dev/null; then
        python -c "
import photon_mlir
from photon_mlir.quantum_validation import QuantumValidator, ValidationLevel

try:
    validator = QuantumValidator(ValidationLevel.STRICT)
    print('✓ Security validation initialized successfully')
except Exception as e:
    print(f'✗ Security validation failed: {e}')
    exit(1)
" || {
            log_error "Security validation failed"
            exit 1
        }
    fi
    
    log_info "Security checks complete"
}

# Initialize monitoring
start_monitoring() {
    if [ "$QUANTUM_ENABLE_MONITORING" = "true" ]; then
        log_info "Starting performance monitoring..."
        
        # Start monitoring in background
        python -c "
import time
import signal
import sys
from photon_mlir.quantum_validation import QuantumMonitor

def signal_handler(sig, frame):
    print('Monitoring shutdown requested')
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

monitor = QuantumMonitor()
monitor.start_monitoring()

try:
    while True:
        time.sleep(10)
        summary = monitor.get_performance_summary()
        if summary.get('total_measurements', 0) > 0:
            print(f'Monitor: {summary[\"total_measurements\"]} measurements, '
                  f'{summary.get(\"active_alerts\", 0)} alerts')
except KeyboardInterrupt:
    pass
finally:
    monitor.stop_monitoring()
    print('Monitoring stopped')
" &
        
        MONITOR_PID=$!
        log_info "Monitoring started (PID: $MONITOR_PID)"
    fi
}

# Start the quantum scheduler server
start_server() {
    log_info "Starting Quantum-Inspired Task Scheduler server..."
    log_info "Configuration: $QUANTUM_CONFIG_PATH"
    log_info "Listen address: 0.0.0.0:$QUANTUM_PORT"
    log_info "Workers: $QUANTUM_WORKERS"
    log_info "Log level: $QUANTUM_LOG_LEVEL"
    
    # Use gunicorn for production WSGI server
    exec python -m photon_mlir.server \
        --config "$QUANTUM_CONFIG_PATH" \
        --host "0.0.0.0" \
        --port "$QUANTUM_PORT" \
        --workers "$QUANTUM_WORKERS" \
        --log-level "$QUANTUM_LOG_LEVEL" \
        --access-logfile "$QUANTUM_LOG_DIR/access.log" \
        --error-logfile "$QUANTUM_LOG_DIR/error.log" \
        --timeout 300 \
        --keepalive 2 \
        --max-requests 1000 \
        --max-requests-jitter 100 \
        --preload \
        --worker-class "sync" \
        --worker-connections 1000
}

# Start the CLI interface  
start_cli() {
    log_info "Starting Quantum Task Scheduler CLI..."
    exec python -m photon_mlir.cli "$@"
}

# Run benchmarks
run_benchmarks() {
    log_info "Running quantum scheduler benchmarks..."
    cd /app
    exec python -m pytest tests/benchmarks/performance/test_quantum_benchmarks.py -v --benchmark
}

# Run tests
run_tests() {
    log_info "Running quantum scheduler tests..."
    cd /app
    exec python -m pytest tests/ -v --cov=photon_mlir --cov-report=term-missing
}

# Show version information
show_version() {
    python -c "
import photon_mlir
print(f'Photon MLIR Quantum Scheduler v{photon_mlir.__version__}')
print(f'Author: {photon_mlir.__author__}')
print(f'Email: {photon_mlir.__email__}')
"
}

# Show help
show_help() {
    cat << EOF
Quantum-Inspired Task Scheduler - Production Container

USAGE:
    entrypoint.sh [COMMAND] [OPTIONS]

COMMANDS:
    server          Start the HTTP API server (default)
    cli             Start the CLI interface
    benchmark       Run performance benchmarks
    test            Run test suite
    version         Show version information
    help            Show this help message

ENVIRONMENT VARIABLES:
    QUANTUM_CONFIG_PATH     Configuration file path (default: $DEFAULT_CONFIG_PATH)
    QUANTUM_LOG_LEVEL      Logging level (default: $DEFAULT_LOG_LEVEL)
    QUANTUM_PORT           Server port (default: $DEFAULT_PORT)
    QUANTUM_WORKERS        Number of workers (default: $DEFAULT_WORKERS)
    QUANTUM_DEBUG          Enable debug mode (default: 0)
    QUANTUM_CACHE_DIR      Cache directory
    QUANTUM_LOG_DIR        Log directory  
    QUANTUM_DATA_DIR       Data directory

EXAMPLES:
    # Start server with custom port
    QUANTUM_PORT=9000 entrypoint.sh server
    
    # Run CLI with debug output
    QUANTUM_DEBUG=1 entrypoint.sh cli --help
    
    # Run benchmarks
    entrypoint.sh benchmark

EOF
}

# Main execution logic
main() {
    log_info "Quantum-Inspired Task Scheduler starting up..."
    log_info "Container image: photon-mlir-quantum:$(cat /app/VERSION 2>/dev/null || echo 'unknown')"
    log_info "User: $(id)"
    log_info "Working directory: $(pwd)"
    
    # Parse command
    COMMAND="${1:-server}"
    shift 2>/dev/null || true
    
    case "$COMMAND" in
        server)
            validate_environment
            run_security_checks
            start_monitoring
            start_server
            ;;
        cli)
            validate_environment
            start_cli "$@"
            ;;
        benchmark)
            validate_environment
            run_benchmarks
            ;;
        test)
            validate_environment
            run_tests
            ;;
        version)
            show_version
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log_error "Unknown command: $COMMAND"
            show_help
            exit 1
            ;;
    esac
}

# Execute main function
main "$@"