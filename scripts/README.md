# ContentCreator Git Automation Pipeline

This directory contains automated git pipeline scripts for the ContentCreator repository.

## Overview

The git automation pipeline provides:
- **Automated commits** with intelligent commit messages
- **Automated pushes** to GitHub repository
- **Test integration** before commits
- **Code formatting** and linting
- **GitHub Actions** CI/CD pipeline
- **Flexible configuration** options

## Scripts

### 1. Bash Script (`git_pipeline.sh`)
Full-featured bash script with colored output and comprehensive error handling.

```bash
# Basic usage
./scripts/git_pipeline.sh

# With custom commit message
./scripts/git_pipeline.sh "Add new feature"

# Run tests before commit
./scripts/git_pipeline.sh --test "Fix bug in parser"

# Force push
./scripts/git_pipeline.sh --force

# Setup remote only
./scripts/git_pipeline.sh --setup

# Quick mode (skip tests)
./scripts/git_pipeline.sh --quick "Quick fix"
```

### 2. Python Script (`git_auto.py`)
Object-oriented Python script with better integration into Python projects.

```bash
# Basic usage
python scripts/git_auto.py --username yourusername

# With custom commit message
python scripts/git_auto.py --username yourusername -m "Add new feature"

# Run tests before commit
python scripts/git_auto.py --username yourusername --test

# Force push
python scripts/git_auto.py --username yourusername --force

# Setup remote only
python scripts/git_auto.py --setup --username yourusername
```

## Makefile Integration

Use the Makefile for convenient command shortcuts:

```bash
# Setup development environment
make setup

# Complete automated pipeline
make git-auto USERNAME=yourusername MESSAGE="Add new feature"

# Run tests before git operations
make git-test USERNAME=yourusername

# Force push changes
make git-force USERNAME=yourusername

# Quick commit with formatting
make quick-commit MESSAGE="Fix typo"

# Full development pipeline
make full-pipeline USERNAME=yourusername MESSAGE="Complete feature"
```

## Configuration

### 1. Script Configuration
Edit the scripts directly to set your default values:

**Bash Script (`git_pipeline.sh`):**
```bash
GITHUB_USERNAME="your-github-username"
REPO_NAME="ContentCreator"
BRANCH="main"
```

**Python Script (`git_auto.py`):**
```python
pipeline = GitPipeline(repo_name="ContentCreator", username="your-username")
```

### 2. YAML Configuration
Use `config/git_config.yml` for advanced configuration:

```yaml
git:
  repository:
    username: "your-github-username"
    name: "ContentCreator"
```

## Initial Setup

1. **Configure Git User** (if not already done):
   ```bash
   git config --global user.name "Your Name"
   git config --global user.email "your.email@example.com"
   ```

2. **Create GitHub Repository**:
   - Go to GitHub and create a new repository named `ContentCreator`
   - Make it public or private as needed

3. **Set Up Remote**:
   ```bash
   # Using bash script
   ./scripts/git_pipeline.sh --setup

   # Using python script
   python scripts/git_auto.py --setup --username yourusername

   # Using Makefile
   make git-setup USERNAME=yourusername
   ```

4. **Make Scripts Executable**:
   ```bash
   chmod +x scripts/git_pipeline.sh
   chmod +x scripts/git_auto.py
   ```

## Features

### Intelligent Commit Messages
The scripts automatically generate descriptive commit messages based on file changes:
- `Auto-commit: 3 added, 2 modified [2024-01-15 14:30:22]`
- `Auto-commit: 1 deleted, 5 modified [2024-01-15 14:30:22]`

### Test Integration
Run tests before committing to ensure code quality:
```bash
# Bash
./scripts/git_pipeline.sh --test

# Python
python scripts/git_auto.py --test --username yourusername

# Makefile
make git-test USERNAME=yourusername
```

### Force Push Handling
Safely handle force pushes with warnings:
```bash
# Bash
./scripts/git_pipeline.sh --force

# Python
python scripts/git_auto.py --force --username yourusername

# Makefile
make git-force USERNAME=yourusername
```

### Auto Pull and Merge
Automatically handle merge conflicts by pulling remote changes:
- Attempts to pull and merge if push fails
- Provides clear error messages for manual intervention

## GitHub Actions Integration

The pipeline includes a comprehensive GitHub Actions workflow (`.github/workflows/ci-cd.yml`) that:

- **Tests** on multiple Python versions (3.8, 3.9, 3.10, 3.11)
- **Lints** code with flake8
- **Formats** code with Black and isort
- **Runs security scans** with bandit and safety
- **Auto-formats** pull requests
- **Creates releases** automatically
- **Caches dependencies** for faster builds

## Troubleshooting

### Common Issues

1. **"Git user not configured"**
   ```bash
   git config --global user.name "Your Name"
   git config --global user.email "your.email@example.com"
   ```

2. **"Remote repository not found"**
   - Ensure the GitHub repository exists
   - Check the repository name and username
   - Verify repository permissions

3. **"Push failed"**
   - Check internet connection
   - Verify GitHub credentials
   - Try force push if safe: `--force`

4. **"Tests failed"**
   - Fix failing tests before committing
   - Use `--quick` mode to skip tests temporarily
   - Check test dependencies

### Debug Mode

For detailed debugging, run commands with verbose output:
```bash
# Bash script debug
bash -x scripts/git_pipeline.sh

# Python script debug
python scripts/git_auto.py --username yourusername -v
```

## Security Considerations

- **Never commit sensitive data** (API keys, passwords, etc.)
- **Review changes** before using automation
- **Use `.gitignore`** to exclude sensitive files
- **Enable branch protection** on GitHub for important branches

## Advanced Usage

### Custom Hooks
Add custom pre-commit and pre-push hooks by modifying the scripts:

```bash
# In git_pipeline.sh, add before commit:
echo "Running custom checks..."
./scripts/custom_check.sh
```

### Notification Integration
Integrate with external services:
- Slack notifications
- Email alerts
- Discord webhooks

### Multiple Repositories
Use the scripts for multiple repositories by changing the configuration:
```bash
python scripts/git_auto.py --repo MyOtherRepo --username yourusername
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Use the automation pipeline to commit:
   ```bash
   make git-auto USERNAME=yourusername MESSAGE="Add awesome feature"
   ```

## License

This automation pipeline is part of the ContentCreator project and follows the same license terms. 