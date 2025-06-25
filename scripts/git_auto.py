#!/usr/bin/env python3
"""
ContentCreator Auto Git Pipeline (Python Version)
Automated git operations for the ContentCreator-0.1 repository
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple


class Colors:
    """ANSI color codes for terminal output"""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    RESET = '\033[0m'


class GitPipeline:
    """Automated Git Pipeline for ContentCreator"""
    
    def __init__(self, repo_name: str = "ContentCreator", username: str = ""):
        self.repo_name = repo_name
        self.username = username
        self.branch = "main"
        self.project_root = Path.cwd()
        
    def print_status(self, message: str) -> None:
        """Print info message with color"""
        print(f"{Colors.BLUE}[INFO]{Colors.RESET} {message}")
        
    def print_success(self, message: str) -> None:
        """Print success message with color"""
        print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} {message}")
        
    def print_warning(self, message: str) -> None:
        """Print warning message with color"""
        print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} {message}")
        
    def print_error(self, message: str) -> None:
        """Print error message with color"""
        print(f"{Colors.RED}[ERROR]{Colors.RESET} {message}")
        
    def run_command(self, command: List[str], capture_output: bool = False) -> Tuple[int, str, str]:
        """Run a shell command and return exit code, stdout, stderr"""
        try:
            result = subprocess.run(
                command,
                capture_output=capture_output,
                text=True,
                cwd=self.project_root
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.SubprocessError as e:
            return 1, "", str(e)
    
    def is_git_repo(self) -> bool:
        """Check if current directory is a git repository"""
        return (self.project_root / ".git").exists()
    
    def init_git_repo(self) -> bool:
        """Initialize git repository if not already initialized"""
        if not self.is_git_repo():
            self.print_status("Initializing git repository...")
            exit_code, _, _ = self.run_command(["git", "init"])
            if exit_code == 0:
                self.print_success("Git repository initialized")
                return True
            else:
                self.print_error("Failed to initialize git repository")
                return False
        return True
    
    def check_git_config(self) -> bool:
        """Check if git user is configured"""
        name_code, name_out, _ = self.run_command(["git", "config", "--global", "user.name"], capture_output=True)
        email_code, email_out, _ = self.run_command(["git", "config", "--global", "user.email"], capture_output=True)
        
        if name_code != 0 or email_code != 0 or not name_out.strip() or not email_out.strip():
            self.print_warning("Git user not configured globally. Please run:")
            print("git config --global user.name 'Your Name'")
            print("git config --global user.email 'your.email@example.com'")
            return False
        return True
    
    def get_remote_url(self) -> Optional[str]:
        """Get the current remote origin URL"""
        exit_code, output, _ = self.run_command(["git", "remote", "get-url", "origin"], capture_output=True)
        if exit_code == 0:
            return output.strip()
        return None
    
    def setup_remote(self) -> bool:
        """Setup remote repository"""
        if not self.username:
            self.print_error("GitHub username not provided. Use --username or set in config")
            return False
        
        remote_url = f"https://github.com/{self.username}/{self.repo_name}.git"
        current_remote = self.get_remote_url()
        
        if current_remote:
            self.print_status(f"Remote origin already exists: {current_remote}")
            return True
        else:
            self.print_status(f"Adding remote origin: {remote_url}")
            exit_code, _, _ = self.run_command(["git", "remote", "add", "origin", remote_url])
            if exit_code == 0:
                self.print_success("Remote origin added")
                return True
            else:
                self.print_error("Failed to add remote origin")
                return False
    
    def get_git_status(self) -> Tuple[List[str], List[str], List[str]]:
        """Get git status - returns (added, modified, deleted) files"""
        exit_code, output, _ = self.run_command(["git", "status", "--porcelain"], capture_output=True)
        if exit_code != 0:
            return [], [], []
        
        added, modified, deleted = [], [], []
        for line in output.strip().split('\n'):
            if not line:
                continue
            status = line[:2]
            filename = line[3:]
            
            if 'A' in status or '?' in status:
                added.append(filename)
            elif 'M' in status:
                modified.append(filename)
            elif 'D' in status:
                deleted.append(filename)
                
        return added, modified, deleted
    
    def has_changes(self) -> bool:
        """Check if there are any changes to commit"""
        added, modified, deleted = self.get_git_status()
        return bool(added or modified or deleted)
    
    def generate_commit_message(self) -> str:
        """Generate automatic commit message based on changes"""
        added, modified, deleted = self.get_git_status()
        
        parts = []
        if added:
            parts.append(f"{len(added)} added")
        if modified:
            parts.append(f"{len(modified)} modified")
        if deleted:
            parts.append(f"{len(deleted)} deleted")
        
        if not parts:
            return "Auto-commit: Minor updates"
        
        return f"Auto-commit: {', '.join(parts)}"
    
    def add_and_commit(self, message: Optional[str] = None) -> bool:
        """Add all changes and commit with given message"""
        if not self.has_changes():
            self.print_warning("No changes to commit")
            return True
        
        # Add all changes
        self.print_status("Adding all changes...")
        exit_code, _, _ = self.run_command(["git", "add", "."])
        if exit_code != 0:
            self.print_error("Failed to add changes")
            return False
        
        # Generate commit message if not provided
        if not message:
            message = self.generate_commit_message()
        
        # Add timestamp to commit message
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_message = f"{message} [{timestamp}]"
        
        self.print_status(f"Committing with message: {full_message}")
        exit_code, _, _ = self.run_command(["git", "commit", "-m", full_message])
        
        if exit_code == 0:
            self.print_success("Changes committed")
            return True
        else:
            self.print_error("Failed to commit changes")
            return False
    
    def push_to_remote(self, force: bool = False) -> bool:
        """Push changes to remote repository"""
        self.print_status("Pushing to remote repository...")
        
        push_args = ["git", "push", "-u", "origin", self.branch]
        if force:
            push_args.append("--force")
        
        exit_code, _, stderr = self.run_command(push_args)
        
        if exit_code == 0:
            self.print_success(f"Pushed to origin/{self.branch}")
            return True
        elif not force:
            self.print_warning("Push failed. Attempting to pull and merge...")
            
            # Try to pull first
            pull_exit, _, _ = self.run_command(["git", "pull", "origin", self.branch, "--no-edit"])
            if pull_exit == 0:
                # Try push again
                push_exit, _, _ = self.run_command(push_args)
                if push_exit == 0:
                    self.print_success("Pulled, merged, and pushed successfully")
                    return True
            
            self.print_error("Push failed. Manual intervention required.")
            self.print_error(f"Error: {stderr}")
            return False
        else:
            self.print_error(f"Force push failed: {stderr}")
            return False
    
    def run_tests(self) -> bool:
        """Run tests if available"""
        if (self.project_root / "pyproject.toml").exists():
            # Try poetry first
            poetry_exit, _, _ = self.run_command(["poetry", "--version"], capture_output=True)
            if poetry_exit == 0:
                self.print_status("Running tests with Poetry...")
                exit_code, _, _ = self.run_command(["poetry", "run", "pytest"])
                return exit_code == 0
        
        # Fall back to direct pytest
        if (self.project_root / "src" / "tests").exists():
            self.print_status("Running tests with pytest...")
            exit_code, _, _ = self.run_command(["python", "-m", "pytest", "src/tests/"])
            return exit_code == 0
        
        self.print_warning("No tests found or test runner not available")
        return True
    
    def auto_pipeline(self, message: Optional[str] = None, force: bool = False, 
                     run_tests: bool = False, setup_only: bool = False) -> bool:
        """Run the complete automated git pipeline"""
        self.print_status("Starting ContentCreator Auto Git Pipeline...")
        
        # Initialize git if needed
        if not self.init_git_repo():
            return False
        
        # Check git configuration
        if not self.check_git_config():
            return False
        
        # Setup remote
        if not self.setup_remote():
            return False
        
        if setup_only:
            self.print_success("Remote setup completed")
            return True
        
        # Run tests if requested
        if run_tests:
            if not self.run_tests():
                self.print_error("Tests failed. Aborting commit.")
                return False
        
        # Commit changes
        if not self.add_and_commit(message):
            return False
        
        # Push to remote
        if not self.push_to_remote(force):
            return False
        
        self.print_success("Auto Git Pipeline completed successfully!")
        return True


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="ContentCreator Auto Git Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python git_auto.py                              # Auto commit and push
  python git_auto.py -m "Add new feature"        # Custom commit message
  python git_auto.py --test                      # Run tests before commit
  python git_auto.py --force                     # Force push changes
  python git_auto.py --setup --username myuser   # Setup remote only
        """
    )
    
    parser.add_argument("-m", "--message", help="Custom commit message")
    parser.add_argument("-f", "--force", action="store_true", help="Force push to remote")
    parser.add_argument("-t", "--test", action="store_true", help="Run tests before committing")
    parser.add_argument("-s", "--setup", action="store_true", help="Setup remote repository only")
    parser.add_argument("-u", "--username", help="GitHub username")
    parser.add_argument("-r", "--repo", default="ContentCreator-0.1", help="Repository name")
    
    args = parser.parse_args()
    
    # Create pipeline instance
    pipeline = GitPipeline(repo_name=args.repo, username=args.username or "")
    
    # Run the pipeline
    success = pipeline.auto_pipeline(
        message=args.message,
        force=args.force,
        run_tests=args.test,
        setup_only=args.setup
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main() 