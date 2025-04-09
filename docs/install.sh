#!/bin/bash

# Mintlify Documentation Installation Script

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "npm is not installed. Please install Node.js and npm first."
    echo "Visit https://nodejs.org for installation instructions."
    exit 1
fi

# Install Mintlify CLI
echo "Installing Mintlify CLI..."
npm install -g @mintlify/cli

# Success message
echo "✅ Mintlify CLI installed successfully!"
echo ""
echo "To preview your documentation locally, run:"
echo "  cd docs"
echo "  mintlify dev"
echo ""
echo "Your documentation will be available at http://localhost:3000"
echo ""
echo "For more information, visit https://mintlify.com/docs/development" 