#!/bin/bash

# ContentCreator Auto Git Pipeline
# This script automates git operations for the ContentCreator-0.1 repository

set -e  # Exit immediately if a command exits with a non-zero status

# Configuration
REPO_NAME="ContentCreator-0.1"
GITHUB_USERNAME=""  # Set this to your GitHub username
REMOTE_URL=""       # Will be constructed from username and repo name
BRANCH="main"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if git is initialized
check_git_init() {
    if [ ! -d ".git" ]; then
        print_status "Initializing git repository..."
        git init
        print_success "Git repository initialized"
    fi
}

# Function to configure git user (if not already set)
configure_git_user() {
    if [ -z "$(git config --global user.name)" ] || [ -z "$(git config --global user.email)" ]; then
        print_warning "Git user not configured globally. Please set up:"
        echo "git config --global user.name 'Your Name'"
        echo "git config --global user.email 'your.email@example.com'"
        return 1
    fi
}

# Function to set up remote repository
setup_remote() {
    if [ -z "$GITHUB_USERNAME" ]; then
        print_error "Please set GITHUB_USERNAME in this script"
        return 1
    fi
    
    REMOTE_URL="https://github.com/${GITHUB_USERNAME}/${REPO_NAME}.git"
    
    # Check if remote already exists
    if git remote get-url origin >/dev/null 2>&1; then
        print_status "Remote origin already exists: $(git remote get-url origin)"
    else
        print_status "Adding remote origin: $REMOTE_URL"
        git remote add origin "$REMOTE_URL"
        print_success "Remote origin added"
    fi
}

# Function to generate commit message based on changes
generate_commit_message() {
    local changes=$(git diff --cached --name-status)
    local added=$(echo "$changes" | grep "^A" | wc -l | tr -d ' ')
    local modified=$(echo "$changes" | grep "^M" | wc -l | tr -d ' ')
    local deleted=$(echo "$changes" | grep "^D" | wc -l | tr -d ' ')
    
    local message="Auto-commit: "
    local parts=()
    
    [ "$added" -gt 0 ] && parts+=("$added added")
    [ "$modified" -gt 0 ] && parts+=("$modified modified")
    [ "$deleted" -gt 0 ] && parts+=("$deleted deleted")
    
    if [ ${#parts[@]} -eq 0 ]; then
        message="Auto-commit: Minor updates"
    else
        message="$message$(IFS=', '; echo "${parts[*]}")"
    fi
    
    echo "$message"
}

# Function to add and commit changes
commit_changes() {
    local custom_message="$1"
    
    # Check if there are any changes
    if git diff --quiet && git diff --cached --quiet; then
        print_warning "No changes to commit"
        return 0
    fi
    
    # Add all changes
    print_status "Adding all changes..."
    git add .
    
    # Generate or use custom commit message
    if [ -n "$custom_message" ]; then
        local message="$custom_message"
    else
        local message=$(generate_commit_message)
    fi
    
    print_status "Committing with message: $message"
    git commit -m "$message"
    print_success "Changes committed"
}

# Function to push to remote
push_to_remote() {
    local force_push="$1"
    
    print_status "Pushing to remote repository..."
    
    if [ "$force_push" = "force" ]; then
        git push -u origin "$BRANCH" --force
        print_warning "Force pushed to origin/$BRANCH"
    else
        if git push -u origin "$BRANCH" 2>/dev/null; then
            print_success "Pushed to origin/$BRANCH"
        else
            print_warning "Push failed. You may need to pull first or force push."
            print_status "Attempting to pull and merge..."
            if git pull origin "$BRANCH" --no-edit; then
                git push -u origin "$BRANCH"
                print_success "Pulled, merged, and pushed successfully"
            else
                print_error "Pull failed. Manual intervention required."
                return 1
            fi
        fi
    fi
}

# Function to run tests (if they exist)
run_tests() {
    if [ -f "pyproject.toml" ] && command -v poetry >/dev/null; then
        print_status "Running tests with Poetry..."
        poetry run pytest
    elif [ -f "requirements.txt" ] && command -v python >/dev/null; then
        print_status "Running tests with Python..."
        python -m pytest
    else
        print_warning "No test runner found or configured"
    fi
}

# Function to display help
show_help() {
    echo "ContentCreator Auto Git Pipeline"
    echo "Usage: $0 [OPTIONS] [COMMIT_MESSAGE]"
    echo ""
    echo "Options:"
    echo "  -h, --help          Show this help message"
    echo "  -f, --force         Force push to remote"
    echo "  -t, --test          Run tests before committing"
    echo "  -s, --setup         Setup remote repository only"
    echo "  -q, --quick         Quick commit and push (skip tests)"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Auto commit and push"
    echo "  $0 'Add new feature'                 # Commit with custom message"
    echo "  $0 --test 'Fix bug in parser'        # Run tests before commit"
    echo "  $0 --force                           # Force push changes"
    echo "  $0 --setup                           # Setup remote only"
}

# Main function
main() {
    local force_push=""
    local run_tests_flag=""
    local setup_only=""
    local quick_mode=""
    local custom_message=""
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -f|--force)
                force_push="force"
                shift
                ;;
            -t|--test)
                run_tests_flag="true"
                shift
                ;;
            -s|--setup)
                setup_only="true"
                shift
                ;;
            -q|--quick)
                quick_mode="true"
                shift
                ;;
            -*)
                print_error "Unknown option: $1"
                show_help
                exit 1
                ;;
            *)
                custom_message="$1"
                shift
                ;;
        esac
    done
    
    print_status "Starting ContentCreator Auto Git Pipeline..."
    
    # Check git initialization
    check_git_init
    
    # Configure git user
    if ! configure_git_user; then
        exit 1
    fi
    
    # Setup remote
    if ! setup_remote; then
        exit 1
    fi
    
    # If setup only, exit here
    if [ "$setup_only" = "true" ]; then
        print_success "Remote setup completed"
        exit 0
    fi
    
    # Run tests if requested and not in quick mode
    if [ "$run_tests_flag" = "true" ] && [ "$quick_mode" != "true" ]; then
        run_tests
    fi
    
    # Commit changes
    commit_changes "$custom_message"
    
    # Push to remote
    push_to_remote "$force_push"
    
    print_success "Auto Git Pipeline completed successfully!"
}

# Run main function with all arguments
main "$@" 