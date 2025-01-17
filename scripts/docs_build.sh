#!/usr/bin/env bash

set -e
set -x

install_graphviz_mac() {
    if command -v brew >/dev/null 2>&1; then
        echo "Installing Graphviz using Homebrew..."
        brew install graphviz
    else
        echo "Error: Homebrew is not installed. Please install Homebrew first."
        echo "You can install Homebrew by running:"
        echo '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
        return 1
    fi
}

install_graphviz_ubuntu() {
    if command -v apt-get >/dev/null 2>&1; then
        echo "Installing Graphviz on Ubuntu..."
        sudo apt-get update
        sudo apt-get install -y graphviz graphviz-dev
    else
        echo "Error: apt-get not found. Are you sure this is a Debian/Ubuntu system?"
        return 1
    fi
}

# Detect OS and run appropriate installation function
install_graphviz() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        install_graphviz_mac
    elif [[ -f /etc/os-release ]] && grep -q -i "ubuntu\|debian" /etc/os-release; then
        install_graphviz_ubuntu
    else
        echo "Unsupported operating system"
        return 1
    fi
}

# Function to build documentation
docs_build() {
    install_graphviz
    pip install -e ".[docs]"
    pip install falkordb graphrag_sdk neo4j-graphrag pypdf && \
        cd website && \
        python ./process_api_reference.py && \
        python ./process_notebooks.py render
}

# Execute the function only if the script is run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    docs_build
fi
