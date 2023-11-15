#!/bin/bash

# Exit on error
set -e

# Define your repository name and description
REPO_NAME="scalexi"
REPO_DESC="The scalexi package is a robust toolkit designed for the development and fine-tuning of large language models."

# Initialize the local directory as a Git repository
git init

# Adds the files in the local repository and stages them for commit
git add .

# Commit the tracked changes and prepares them to be pushed to a remote repository
git commit -m "Initial commit"

git branch -M main

# Adds the URL for the remote repository where your local repository will be pushed
git remote add origin https://github.com/scalexi/scalexi.git

# Verifies the new remote URL
git remote -v

git pull --rebase origin main

# Pushes the changes in your local repository up to the GitHub repository
git push origin main

# Tagging the version
# You should replace '1.0.0' with your version number
git tag 0.1.0

# Pushes the tags in your local repository to GitHub
git push origin --tags

# To automate the creation of a GitHub repository, you would need to use the GitHub CLI or API
# This requires that you have the GitHub CLI installed and that you are logged in

# Create a new repository on GitHub
#gh repo create $REPO_NAME --public --description "$REPO_DESC"

# Push to the GitHub remote
#git push --set-upstream origin main



