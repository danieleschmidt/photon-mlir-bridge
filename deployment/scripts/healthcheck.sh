#!/bin/sh
set -e

# Health check script for Quantum-Inspired Task Scheduler
# Performs comprehensive health checks for production deployment

# Configuration
HEALTH_ENDPOINT="${QUANTUM_HEALTH_ENDPOINT:-http://localhost:8080/health}"
TIMEOUT="${QUANTUM_HEALTH_TIMEOUT:-10}"
MAX_RESPONSE_TIME="${QUANTUM_MAX_RESPONSE_TIME:-5000}"  # milliseconds
REQUIRED_MEMORY_MB="${QUANTUM_MIN_MEMORY_MB:-100}"

# Exit codes
EXIT_OK=0
EXIT_WARNING=1
EXIT_CRITICAL=2
EXIT_UNKNOWN=3

# Logging
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [HEALTH] $1" >&2
}

# Check if service is responsive
check_http_endpoint() {
    log "Checking HTTP endpoint: $HEALTH_ENDPOINT"
    
    # Use wget or curl, whichever is available
    if command -v wget >/dev/null; then
        response=$(wget -q -O - --timeout="$TIMEOUT" "$HEALTH_ENDPOINT" 2>&1) || {
            log "ERROR: HTTP endpoint not responding"
            return $EXIT_CRITICAL
        }
    elif command -v curl >/dev/null; then
        response=$(curl -s --max-time "$TIMEOUT" "$HEALTH_ENDPOINT" 2>&1) || {
            log "ERROR: HTTP endpoint not responding" 
            return $EXIT_CRITICAL
        }
    else
        # Fallback to Python
        response=$(python3 -c "
import urllib.request
import socket
socket.setdefaulttimeout($TIMEOUT)
try:
    with urllib.request.urlopen('$HEALTH_ENDPOINT') as resp:
        print(resp.read().decode())
except Exception as e:
    print(f'ERROR: {e}')
    exit($EXIT_CRITICAL)
" 2>&1) || return $EXIT_CRITICAL
    fi
    
    # Check response content
    if echo "$response" | grep -q "healthy\|ok\|ready"; then
        log "✓ HTTP endpoint responding correctly"
        return $EXIT_OK
    else
        log "✗ HTTP endpoint returned unexpected response: $response"
        return $EXIT_WARNING
    fi
}

# Check quantum scheduler specific health
check_quantum_scheduler() {
    log "Checking quantum scheduler health..."
    
    python3 -c "
import sys
import time
import traceback

try:
    # Test basic import
    import photon_mlir
    from photon_mlir.quantum_scheduler import QuantumTaskPlanner
    from photon_mlir.quantum_validation import QuantumValidator, ValidationLevel
    
    # Test basic functionality
    start_time = time.time()
    
    # Create planner
    planner = QuantumTaskPlanner()
    
    # Create simple test tasks
    config = {'model_type': 'test', 'layers': 2}
    tasks = planner.create_compilation_plan(config)
    
    # Validate tasks
    validator = QuantumValidator(ValidationLevel.BASIC)
    validation_result = validator.validate_tasks(tasks)
    
    if not validation_result.is_valid:
        print('ERROR: Task validation failed')
        sys.exit($EXIT_WARNING)
    
    # Quick optimization test
    result = planner.optimize_schedule(tasks)
    
    if result.makespan <= 0:
        print('ERROR: Invalid scheduling result')
        sys.exit($EXIT_WARNING)
    
    elapsed = (time.time() - start_time) * 1000  # ms
    
    if elapsed > $MAX_RESPONSE_TIME:
        print(f'WARNING: Health check took {elapsed:.1f}ms (limit: ${MAX_RESPONSE_TIME}ms)')
        sys.exit($EXIT_WARNING)
    
    print(f'✓ Quantum scheduler healthy (response time: {elapsed:.1f}ms)')
    sys.exit($EXIT_OK)
    
except ImportError as e:
    print(f'ERROR: Failed to import quantum scheduler: {e}')
    sys.exit($EXIT_CRITICAL)
    
except Exception as e:
    print(f'ERROR: Quantum scheduler health check failed: {e}')
    traceback.print_exc()
    sys.exit($EXIT_CRITICAL)
" 2>&1 || return $?
    
    return $EXIT_OK
}

# Check system resources
check_system_resources() {
    log "Checking system resources..."
    
    # Memory check
    if [ -f /proc/meminfo ]; then
        available_mb=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo 2>/dev/null || echo "0")
        
        if [ "$available_mb" -lt "$REQUIRED_MEMORY_MB" ]; then
            log "WARNING: Low memory: ${available_mb}MB available (required: ${REQUIRED_MEMORY_MB}MB)"
            return $EXIT_WARNING
        else
            log "✓ Memory OK: ${available_mb}MB available"
        fi
    fi
    
    # Disk space check (cache directory)
    if [ -d "$QUANTUM_CACHE_DIR" ]; then
        cache_usage=$(df "$QUANTUM_CACHE_DIR" 2>/dev/null | awk 'NR==2 {print $5}' | sed 's/%//' || echo "0")
        
        if [ "$cache_usage" -gt 90 ]; then
            log "WARNING: Cache disk usage high: ${cache_usage}%"
            return $EXIT_WARNING
        else
            log "✓ Cache disk OK: ${cache_usage}% used"
        fi
    fi
    
    # Process check
    if ! pgrep -f "photon_mlir" > /dev/null; then
        log "WARNING: No quantum scheduler processes found"
        return $EXIT_WARNING
    else
        process_count=$(pgrep -f "photon_mlir" | wc -l)
        log "✓ Processes OK: $process_count quantum scheduler processes running"
    fi
    
    return $EXIT_OK
}

# Check file system health
check_filesystem() {
    log "Checking filesystem health..."
    
    # Check required directories
    for dir in "$QUANTUM_CACHE_DIR" "$QUANTUM_LOG_DIR" "$QUANTUM_DATA_DIR"; do
        if [ ! -d "$dir" ]; then
            log "ERROR: Required directory missing: $dir"
            return $EXIT_CRITICAL
        fi
        
        if [ ! -w "$dir" ]; then
            log "ERROR: Directory not writable: $dir"
            return $EXIT_CRITICAL
        fi
    done
    
    # Check configuration file
    if [ ! -f "$QUANTUM_CONFIG_PATH" ]; then
        log "ERROR: Configuration file missing: $QUANTUM_CONFIG_PATH"
        return $EXIT_CRITICAL
    fi
    
    if [ ! -r "$QUANTUM_CONFIG_PATH" ]; then
        log "ERROR: Configuration file not readable: $QUANTUM_CONFIG_PATH"
        return $EXIT_CRITICAL
    fi
    
    # Test configuration validity
    python3 -c "
import json
import sys

try:
    with open('$QUANTUM_CONFIG_PATH', 'r') as f:
        config = json.load(f)
    
    # Check required sections
    required_sections = ['quantum_scheduler', 'server', 'logging']
    for section in required_sections:
        if section not in config:
            print(f'ERROR: Missing configuration section: {section}')
            sys.exit($EXIT_CRITICAL)
    
    print('✓ Configuration file valid')
    sys.exit($EXIT_OK)
    
except json.JSONDecodeError as e:
    print(f'ERROR: Invalid JSON in configuration: {e}')
    sys.exit($EXIT_CRITICAL)
    
except Exception as e:
    print(f'ERROR: Configuration check failed: {e}')
    sys.exit($EXIT_CRITICAL)
" 2>&1 || return $?
    
    log "✓ Filesystem OK"
    return $EXIT_OK
}

# Check network connectivity (if required)
check_network() {
    if [ "$QUANTUM_NETWORK_CHECK" = "true" ]; then
        log "Checking network connectivity..."
        
        # Test DNS resolution
        if ! nslookup google.com >/dev/null 2>&1; then
            log "WARNING: DNS resolution failed"
            return $EXIT_WARNING
        fi
        
        # Test HTTP connectivity
        if command -v wget >/dev/null; then
            if ! wget -q --spider --timeout=5 https://httpbin.org/status/200 2>/dev/null; then
                log "WARNING: External HTTP connectivity failed"
                return $EXIT_WARNING
            fi
        fi
        
        log "✓ Network OK"
    fi
    
    return $EXIT_OK
}

# Comprehensive health check
run_health_check() {
    log "Starting comprehensive health check..."
    
    overall_status=$EXIT_OK
    
    # Run all checks
    checks=(
        "check_filesystem"
        "check_system_resources" 
        "check_quantum_scheduler"
        "check_network"
    )
    
    # Only check HTTP endpoint if server mode
    if [ "$1" != "--skip-http" ]; then
        checks=("check_http_endpoint" "${checks[@]}")
    fi
    
    for check in "${checks[@]}"; do
        $check
        status=$?
        
        if [ $status -gt $overall_status ]; then
            overall_status=$status
        fi
        
        # Stop on critical errors (unless in debug mode)
        if [ $status -eq $EXIT_CRITICAL ] && [ "$QUANTUM_DEBUG" != "1" ]; then
            break
        fi
    done
    
    case $overall_status in
        $EXIT_OK)
            log "✅ All health checks passed"
            ;;
        $EXIT_WARNING)
            log "⚠️  Health check completed with warnings"
            ;;
        $EXIT_CRITICAL)
            log "❌ Critical health check failures detected"
            ;;
        *)
            log "❓ Unknown health check status: $overall_status"
            overall_status=$EXIT_UNKNOWN
            ;;
    esac
    
    return $overall_status
}

# Quick health check (for frequent monitoring)
run_quick_check() {
    log "Running quick health check..."
    
    # Just check if the service is responding
    if check_http_endpoint; then
        log "✅ Quick health check passed"
        return $EXIT_OK
    else
        log "❌ Quick health check failed"
        return $EXIT_CRITICAL
    fi
}

# Show usage
show_usage() {
    cat << EOF
Quantum Scheduler Health Check

USAGE:
    healthcheck.sh [OPTIONS]

OPTIONS:
    --quick         Run quick health check only
    --skip-http     Skip HTTP endpoint check
    --help          Show this help

ENVIRONMENT VARIABLES:
    QUANTUM_HEALTH_ENDPOINT     Health check endpoint (default: http://localhost:8080/health)
    QUANTUM_HEALTH_TIMEOUT      Request timeout in seconds (default: 10)
    QUANTUM_MAX_RESPONSE_TIME   Max acceptable response time in ms (default: 5000)
    QUANTUM_MIN_MEMORY_MB       Minimum required memory in MB (default: 100)
    QUANTUM_NETWORK_CHECK       Enable network connectivity check (default: false)
    QUANTUM_DEBUG               Enable debug output (default: false)

EXIT CODES:
    0    OK - All checks passed
    1    WARNING - Non-critical issues found
    2    CRITICAL - Service not healthy
    3    UNKNOWN - Check status unclear

EOF
}

# Main execution
main() {
    case "$1" in
        --quick)
            run_quick_check
            ;;
        --skip-http)
            run_health_check --skip-http
            ;;
        --help|-h)
            show_usage
            exit $EXIT_OK
            ;;
        "")
            run_health_check
            ;;
        *)
            log "ERROR: Unknown option: $1"
            show_usage
            exit $EXIT_UNKNOWN
            ;;
    esac
}

# Execute main function
main "$@"