#!/usr/bin/env bash
# Build open-data-sci and publish it to TestPyPI.
#
# Usage:
#   ./dev/scripts/test_publish_open_data_sci.sh -t <token>
#   TESTPYPI_TOKEN=<token> ./dev/scripts/test_publish_open_data_sci.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

TOKEN="${TESTPYPI_TOKEN:-}"

usage() {
    echo "Usage: $0 [-t TOKEN]"
    echo "  -t TOKEN   TestPyPI API token (or set TESTPYPI_TOKEN env var)"
    exit 1
}

parse_args() {
    while getopts "t:h" opt; do
        case "$opt" in
            t) TOKEN="$OPTARG" ;;
            h) usage ;;
            *) usage ;;
        esac
    done
}

check_requirements() {
    if ! command -v uv >/dev/null 2>&1; then
        echo "Error: 'uv' is required but was not found on PATH. Install it from https://docs.astral.sh/uv/" >&2
        exit 1
    fi

    if [ -z "$TOKEN" ]; then
        echo "Error: no token provided. Pass -t TOKEN or set TESTPYPI_TOKEN." >&2
        exit 1
    fi
}

build_package() {
    rm -rf dist/
    uv build

    if ! compgen -G "dist/*" >/dev/null; then
        echo "Error: 'uv build' did not produce any files in dist/." >&2
        exit 1
    fi
}

publish_package() {
    uv publish \
        --publish-url https://test.pypi.org/legacy/ \
        --token "$TOKEN" \
        dist/*
}

main() {
    parse_args "$@"
    check_requirements
    build_package
    publish_package
}

main "$@"
