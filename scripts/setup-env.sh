#!/bin/bash
# Environment Setup Script for AI Scene-to-Video Pipeline

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Check if we're in the project root
if [ ! -f "pyproject.toml" ]; then
    log_error "Please run this script from the project root directory"
    exit 1
fi

log_info "Setting up AI Scene-to-Video Pipeline development environment..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    log_info "Creating Python virtual environment..."
    python3 -m venv venv
    log_success "Virtual environment created"
else
    log_info "Virtual environment already exists"
fi

# Activate virtual environment
log_info "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
log_info "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
log_info "Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    log_success "Dependencies installed from requirements.txt"
fi

# Install development dependencies if available
if [ -f "requirements-dev.txt" ]; then
    pip install -r requirements-dev.txt
    log_success "Development dependencies installed"
fi

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        log_info "Creating .env file from template..."
        cp .env.example .env
        log_warning "Please edit .env file with your actual API keys!"
        log_info "Required API keys:"
        log_info "  - OPENAI_API_KEY: Get from https://platform.openai.com/api-keys"
        log_info "  - FAL_API_KEY: Get from https://fal.ai/dashboard"
    else
        log_error ".env.example not found!"
        exit 1
    fi
else
    log_info ".env file already exists"
fi

# Create stories directory for test content
if [ ! -d "stories" ]; then
    log_info "Creating stories directory..."
    mkdir -p stories
    
    # Create a sample story file
    cat > stories/sample_story.txt << 'EOF'
The Detective's Discovery

Detective Sarah Chen had been investigating the mysterious disappearances for weeks. The trail had led her to an abandoned warehouse on the outskirts of the city, where shadows danced in the dim moonlight filtering through broken windows.

As she stepped carefully through the debris-strewn floor, her flashlight beam caught something glinting in the corner. A locket, tarnished with age, lay among the dust and forgotten memories. The moment she picked it up, she heard footsteps echoing from the floor above.

With her heart racing, Sarah climbed the creaking stairs, each step bringing her closer to the truth she had been seeking. At the top, a door stood slightly ajar, revealing a room filled with old photographs and documents scattered across makeshift tables.

The pieces of the puzzle were finally coming together, but Sarah realized she wasn't alone in this discovery. A shadow moved behind her, and she spun around to face whatever danger awaited in the darkness.
EOF
    
    log_success "Sample story created in stories/sample_story.txt"
fi

# Create assets directory structure
log_info "Setting up assets directory structure..."
mkdir -p src/assets/{images,videos,temp}
touch src/assets/.gitkeep

# Display activation instructions
log_success "Environment setup complete!"
echo ""
log_info "To activate the virtual environment in future sessions:"
echo "  source venv/bin/activate"
echo ""
log_info "To test the installation:"
echo "  python -m src.main version"
echo ""
log_info "To run a test generation:"
echo "  python -m src.main test"
echo ""
log_info "To generate from the sample story:"
echo "  python -m src.main generate-media stories/sample_story.txt --dry-run"
echo ""
log_warning "Remember to configure your API keys in the .env file!"

# Check if API keys are configured
if grep -q "your_.*_api_key_here" .env; then
    log_warning "API keys not yet configured. Please edit .env file."
else
    log_success "API keys appear to be configured ✅"
fi 