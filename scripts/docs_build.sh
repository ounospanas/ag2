#!/usr/bin/env bash

set -e
set -x

# Source the docs_install_deps.sh script from the same directory
source "$(dirname "$0")/docs_install_deps.sh"

install_graphviz_mac() {
    if command -v brew >/dev/null 2>&1; then
        echo "Installing Graphviz using Homebrew..."
        brew install graphviz
    else
        echo "Error: Homebrew is not installed. Please install Homebrew first."
        echo "You can install Homebrew by running:"
        echo '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
        exit 1
    fi
}

install_graphviz_ubuntu() {
    if command -v apt-get >/dev/null 2>&1; then
        echo "Installing Graphviz on Ubuntu..."
        sudo apt-get update && \
        sudo apt-get install -y graphviz graphviz-dev
    else
        echo "Error: apt-get not found. Are you sure this is a Debian/Ubuntu system?"
        exit 1
    fi
}

install_graphviz() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        install_graphviz_mac
    elif [[ -f /etc/os-release ]] && grep -q -i "ubuntu\|debian" /etc/os-release; then
        install_graphviz_ubuntu
    else
        echo "Unsupported operating system"
        exit 1
    fi
}

docs_generate() {
    cd website && \
        python ./process_api_reference.py && \
        python ./process_notebooks.py render
}

docs_build() {
    install_graphviz && \
        install_packages && \
        docs_generate
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    docs_build
fi
