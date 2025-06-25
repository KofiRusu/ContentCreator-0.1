# ContentCreator Git Pipeline - Quick Start

This guide will get you up and running with the automated git pipeline in under 5 minutes.

## Prerequisites

- Git installed and configured
- Python 3.8+ installed
- GitHub account with ContentCreator repository created

## Step-by-Step Setup

### 1. Configure Git (if not already done)
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### 2. Setup Development Environment
```bash
make setup
```

### 3. Configure Your GitHub Username
Edit one of these files to set your GitHub username:

**Option A: Edit the Python script directly**
```bash
# Edit scripts/git_auto.py, line 26:
username: str = "your-github-username"  # Replace with your username
```

**Option B: Edit the config file**
```bash
# Edit config/git_config.yml:
git:
  repository:
    username: "your-github-username"  # Replace with your username
```

### 4. Setup Remote Repository
```bash
# Replace 'yourusername' with your actual GitHub username
make git-setup USERNAME=yourusername
```

## Quick Usage Examples

### Basic Automated Commit and Push
```bash
make git-auto USERNAME=yourusername
```

### Commit with Custom Message
```bash
make git-auto USERNAME=yourusername MESSAGE="Add new feature"
```

### Run Tests Before Committing
```bash
make git-test USERNAME=yourusername
```

### Force Push (use carefully!)
```bash
make git-force USERNAME=yourusername
```

### Full Development Pipeline (format, test, commit, push)
```bash
make full-pipeline USERNAME=yourusername MESSAGE="Complete feature implementation"
```

## Alternative Usage Methods

### Using Python Script Directly
```bash
# Basic usage
python scripts/git_auto.py --username yourusername

# With custom message
python scripts/git_auto.py --username yourusername -m "Add new feature"

# With tests
python scripts/git_auto.py --username yourusername --test
```

### Using Bash Script Directly
```bash
# First, edit the script to set your username:
# Edit scripts/git_pipeline.sh, line 10:
GITHUB_USERNAME="your-github-username"

# Then use:
./scripts/git_pipeline.sh
./scripts/git_pipeline.sh "Custom commit message"
./scripts/git_pipeline.sh --test "Run tests first"
```

## What the Pipeline Does

1. **Checks** if git is properly configured
2. **Sets up** remote repository connection
3. **Adds** all changed files to git
4. **Generates** intelligent commit messages based on changes
5. **Commits** changes with timestamp
6. **Pushes** to GitHub repository
7. **Handles** merge conflicts automatically
8. **Runs tests** (if requested)
9. **Formats code** (if using Makefile targets)

## Example Commit Messages Generated

- `Auto-commit: 5 added, 2 modified [2024-01-15 14:30:22]`
- `Auto-commit: 1 deleted, 3 modified [2024-01-15 14:30:22]`
- `Custom message [2024-01-15 14:30:22]`

## Troubleshooting

### Common Error: "USERNAME not provided"
```bash
# Always include USERNAME parameter:
make git-auto USERNAME=yourusername MESSAGE="Your message"
```

### Common Error: "Remote repository not found"
1. Make sure you've created the `ContentCreator` repository on GitHub
2. Check that your username is correct
3. Verify the repository is public or you have access

### Common Error: "Git user not configured"
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

## Next Steps

After completing the quick start:

1. **Read** the full documentation in `scripts/README.md`
2. **Customize** the configuration in `config/git_config.yml`
3. **Set up** GitHub Actions by pushing to your repository
4. **Explore** advanced features like custom hooks and notifications

## GitHub Actions

Once you push to GitHub, the CI/CD pipeline will automatically:
- Run tests on multiple Python versions
- Check code formatting
- Run security scans
- Create releases
- Auto-format pull requests

## Help

For detailed help on any command:
```bash
make help
python scripts/git_auto.py --help
./scripts/git_pipeline.sh --help
```

Happy coding! 🚀 