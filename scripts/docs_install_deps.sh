#!/usr/bin/env bash

set -e
set -x

install_packages() {
    pip install -e ".[docs]"
    pip install "falkordb" "graphrag_sdk" "neo4j-graphrag" "pypdf" \
        "pdoc3>=0.11.5" \
        "pyyaml>=6.0.2" \
        "termcolor>=2.5.0" \
        "nbclient>=0.10.2" \
        "arxiv>=2.1.3" \
        "flaml[automl]" \
        "pygraphviz>=1.14" \
        "replicate"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    install_packages
fi
