#!/usr/bin/env bash
# Install open-data-sci from TestPyPI, falling back to PyPI for dependencies.
#
# Usage:
#   ./dev/scripts/test_install_open_data_sci.sh -e aws,gemini
#   ./dev/scripts/test_install_open_data_sci.sh -v 0.1.0 -e aws
set -euo pipefail

PACKAGE="open-data-sci"
VERSION=""
EXTRAS=""

usage() {
    echo "Usage: $0 [-v VERSION] [-e EXTRA1,EXTRA2,...]"
    echo "  -v VERSION   Specific version to install (defaults to latest on TestPyPI)"
    echo "  -e EXTRAS    Comma-separated list of extras (e.g. aws,gemini,gcp,azure,ollama,jax,dev)"
    exit 1
}

parse_args() {
    while getopts "v:e:h" opt; do
        case "$opt" in
            v) VERSION="$OPTARG" ;;
            e) EXTRAS="$OPTARG" ;;
            h) usage ;;
            *) usage ;;
        esac
    done
}

check_requirements() {
    if ! command -v pip >/dev/null 2>&1; then
        echo "Error: 'pip' is required but was not found on PATH." >&2
        exit 1
    fi
}

build_package_spec() {
    local spec="$PACKAGE"
    if [ -n "$EXTRAS" ]; then
        spec="${spec}[${EXTRAS}]"
    fi
    if [ -n "$VERSION" ]; then
        spec="${spec}==${VERSION}"
    fi
    echo "$spec"
}

install_package() {
    local spec
    spec="$(build_package_spec)"

    # TestPyPI rarely mirrors dependencies, so fall back to the real
    # PyPI index for anything that isn't open-data-sci itself.
    pip install \
        --index-url https://test.pypi.org/simple/ \
        --extra-index-url https://pypi.org/simple/ \
        "$spec"
}

main() {
    parse_args "$@"
    check_requirements
    install_package
}

main "$@"
