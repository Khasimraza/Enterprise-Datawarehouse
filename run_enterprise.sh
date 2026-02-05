#!/bin/bash

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DEFAULT_ENV="dev"
PROJECT_DIR=$(pwd)
LOG_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIR/setup_$(date +%Y%m%d_%H%M%S).log"

# Create logs directory
mkdir -p "$LOG_DIR"

# Logging function
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

# Usage function
usage() {
    cat << EOF
Enterprise Data Warehouse Setup Script

Usage: $0 [OPTIONS]

OPTIONS:
    -e, --env ENVIRONMENT    Target environment (dev|test|prod) [default: $DEFAULT_ENV]
    -s, --skip-generation   Skip model generation
    -d, --dry-run           Show what would be done without executing
    -f, --full-setup        Run complete setup including Snowflake deployment
    -h, --help              Show this help message

EXAMPLES:
    $0                              # Setup dev environment
    $0 --env prod --full-setup      # Full production setup
    $0 --dry-run                    # Preview setup steps
    $0 --skip-generation            # Setup without generating models

REQUIREMENTS:
    - Python 3.8+
    - DBT CLI installed
    - Snowflake credentials configured
    - Git (for version control)

EOF
}

# Parse command line arguments
ENVIRONMENT="$DEFAULT_ENV"
SKIP_GENERATION=false
DRY_RUN=false
FULL_SETUP=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -s|--skip-generation)
            SKIP_GENERATION=true
            shift
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -f|--full-setup)
            FULL_SETUP=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate environment
if [[ ! "$ENVIRONMENT" =~ ^(dev|test|prod)$ ]]; then
    log_error "Invalid environment: $ENVIRONMENT. Must be dev, test, or prod."
    exit 1
fi

# Header
echo -e "${BLUE}"
cat << 'EOF'
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║        Enterprise Data Warehouse Platform Setup              ║
║                                                               ║
║  • 32 Fact Tables                                            ║
║  • 128 Dimension Tables                                      ║
║  • Advanced DBT Transformations                              ║
║  • Comprehensive Data Governance                             ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

log "Starting Enterprise Data Warehouse setup for $ENVIRONMENT environment"
log "Project directory: $PROJECT_DIR"
log "Log file: $LOG_FILE"

# Step 1: Validate Prerequisites
log "Step 1: Validating prerequisites..."

# Check Python
if ! command -v python3 &> /dev/null; then
    log_error "Python 3 is not installed"
    exit 1
fi
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
log_success "Python $PYTHON_VERSION found"

# Check if we're in a virtual environment (recommended)
if [[ -z "$VIRTUAL_ENV" ]]; then
    log_warning "Not running in a virtual environment. Consider using 'python -m venv venv && source venv/bin/activate'"
fi

# Check for required files
REQUIRED_FILES=(
    "dbt_project.yml"
    "requirements.txt"
    "governance/data_catalog.yml"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "$PROJECT_DIR/$file" ]]; then
        log_error "Required file not found: $file"
        exit 1
    fi
done
log_success "All required project files found"

# Step 2: Install Python Dependencies
log "Step 2: Installing Python dependencies..."

if [[ "$DRY_RUN" == "true" ]]; then
    log "[DRY-RUN] Would install: pip install -r requirements.txt"
else
    pip install -r requirements.txt
    log_success "Python dependencies installed"
fi

# Step 3: Environment Configuration
log "Step 3: Setting up environment configuration..."

# Check for .env file
if [[ ! -f "$PROJECT_DIR/.env" ]]; then
    if [[ -f "$PROJECT_DIR/.env.example" ]]; then
        log_warning ".env file not found. Please copy .env.example to .env and configure your credentials"
        if [[ "$DRY_RUN" == "false" ]]; then
            cp .env.example .env
            log "Created .env file from .env.example template"
        fi
    else
        log_error ".env.example file not found. Cannot create environment configuration"
        exit 1
    fi
fi

# Load environment variables if not in dry-run mode
if [[ "$DRY_RUN" == "false" && -f ".env" ]]; then
    source .env
    log_success "Environment variables loaded"
fi

# Step 4: DBT Environment Setup
log "Step 4: Setting up DBT environment..."

if [[ "$DRY_RUN" == "true" ]]; then
    log "[DRY-RUN] Would setup DBT environment"
else
    SETUP_ARGS="--env $ENVIRONMENT"
    if [[ "$SKIP_GENERATION" == "true" ]]; then
        SETUP_ARGS="$SETUP_ARGS --skip-generation"
    fi
    
    python scripts/deployment/setup_dbt_environment.py $SETUP_ARGS
    log_success "DBT environment setup completed"
fi

# Step 5: Snowflake Deployment (if full setup requested)
if [[ "$FULL_SETUP" == "true" ]]; then
    log "Step 5: Deploying Snowflake objects..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "[DRY-RUN] Would deploy Snowflake objects"
    else
        python scripts/deployment/deploy_snowflake_objects.py --env "$ENVIRONMENT"
        log_success "Snowflake objects deployed"
    fi
fi

# Step 6: Model Generation (if not skipped)
if [[ "$SKIP_GENERATION" == "false" ]]; then
    log "Step 6: Generating fact and dimension models..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "[DRY-RUN] Would generate models"
    else
        python scripts/utilities/generate_fact_dimension_models.py
        log_success "Models generated"
    fi
fi

# Step 7: DBT Operations
log "Step 7: Running initial DBT operations..."

if [[ "$DRY_RUN" == "true" ]]; then
    log "[DRY-RUN] Would run DBT operations:"
    log "  - dbt debug"
    log "  - dbt deps" 
    log "  - dbt seed"
    log "  - dbt compile"
    log "  - dbt docs generate"
else
    # DBT debug check
    log "Running DBT debug check..."
    if dbt debug --target "$ENVIRONMENT"; then
        log_success "DBT debug check passed"
    else
        log_error "DBT debug check failed. Please check your profiles.yml and credentials"
        exit 1
    fi
    
    # Install DBT packages
    log "Installing DBT packages..."
    dbt deps
    log_success "DBT packages installed"
    
    # Load seed data
    log "Loading seed data..."
    if dbt seed --target "$ENVIRONMENT"; then
        log_success "Seed data loaded"
    else
        log_warning "Seed data loading had issues, continuing..."
    fi
    
    # Compile models
    log "Compiling DBT models..."
    if dbt compile --target "$ENVIRONMENT"; then
        log_success "Models compiled successfully"
    else
        log_error "Model compilation failed"
        exit 1
    fi
    
    # Generate documentation
    log "Generating DBT documentation..."
    if dbt docs generate --target "$ENVIRONMENT"; then
        log_success "Documentation generated"
    else
        log_warning "Documentation generation had issues, continuing..."
    fi
fi

# Step 8: Validation and Testing
log "Step 8: Running validation tests..."

if [[ "$DRY_RUN" == "true" ]]; then
    log "[DRY-RUN] Would run validation tests"
else
    # Check if staging models can run
    log "Testing staging models..."
    if dbt run --select staging --target "$ENVIRONMENT" --limit 10; then
        log_success "Staging models test passed"
    else
        log_error "Staging models test failed"
        exit 1
    fi
fi

# Step 9: Generate Summary Report
log "Step 9: Generating setup summary..."

SUMMARY_FILE="$LOG_DIR/setup_summary_$(date +%Y%m%d_%H%M%S).txt"

cat > "$SUMMARY_FILE" << EOF
Enterprise Data Warehouse Setup Summary
=====================================

Environment: $ENVIRONMENT
Setup Date: $(date)
Project Directory: $PROJECT_DIR

Configuration:
- Skip Generation: $SKIP_GENERATION
- Dry Run: $DRY_RUN  
- Full Setup: $FULL_SETUP

Files Created/Modified:
- .env (environment configuration)
- ~/.dbt/profiles.yml (DBT profiles)
- models/ (generated models)
- dbt_packages/ (installed packages)

Next Steps:
1. Review and update .env file with your credentials
2. Test DBT connection: dbt debug --target $ENVIRONMENT
3. Run full DBT pipeline: dbt run --target $ENVIRONMENT
4. Run data quality tests: dbt test --target $ENVIRONMENT
5. View documentation: dbt docs serve

Useful Commands:
- dbt run --select marts.facts          # Run fact tables only
- dbt run --select marts.dimensions     # Run dimension tables only
- dbt test --select marts               # Test marts layer
- dbt run --full-refresh                # Full refresh of incremental models

Support:
- Documentation: docs/
- Issues: Check logs in $LOG_DIR
- Architecture: documentation/architecture_design.md
EOF

log_success "Setup summary saved to: $SUMMARY_FILE"

# Final success message
echo ""
log_success "🎉 Enterprise Data Warehouse setup completed successfully!"
echo ""
echo -e "${GREEN}Next Steps:${NC}"
echo "1. 📝 Configure your Snowflake credentials in .env file"
echo "2. 🧪 Test your setup: dbt debug --target $ENVIRONMENT"
echo "3. 🚀 Run your first transformation: dbt run --target $ENVIRONMENT"
echo "4. 📊 View documentation: dbt docs serve"
echo ""
echo -e "${BLUE}Logs and summary:${NC}"
echo "- Setup log: $LOG_FILE"
echo "- Summary report: $SUMMARY_FILE"
echo ""
echo -e "${YELLOW}Support:${NC}"
echo "- Documentation: documentation/"
echo "- Data Catalog: governance/data_catalog.yml"
echo "- Architecture Guide: documentation/architecture_design.md"
echo ""
echo -e "${GREEN}Happy Data Engineering! 🚀${NC}"
