#!/usr/bin/env bash

# This script creates an archive containing the sources for the C++ part of
# metatensor-torch, and copy it to the path given as argument

set -eux

OUTPUT_DIR="$1"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR=$(cd "$OUTPUT_DIR" 2>/dev/null && pwd)

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)

VERSION=$(cat "$ROOT_DIR/metatensor-torch/VERSION")
ARCHIVE_NAME="metatensor-torch-cxx-$VERSION"

TMP_DIR=$(mktemp -d)
mkdir "$TMP_DIR/$ARCHIVE_NAME"

cp -r "$ROOT_DIR"/metatensor-torch/* "$TMP_DIR/$ARCHIVE_NAME/"
cp "$ROOT_DIR/LICENSE" "$TMP_DIR/$ARCHIVE_NAME"
cp "$ROOT_DIR/AUTHORS" "$TMP_DIR/$ARCHIVE_NAME"

# Get the number of commits since last tag, this is used when building the
# code to change the version for development builds
cd "$ROOT_DIR"
./scripts/n-commits-since-last-tag.py "metatensor-torch-v" > "$TMP_DIR/$ARCHIVE_NAME/cmake/n_commits_since_last_tag"

cd "$TMP_DIR"
tar cf "$ARCHIVE_NAME".tar "$ARCHIVE_NAME"

gzip -9 "$ARCHIVE_NAME".tar

rm -f "$ROOT_DIR"/python/metatensor-torch/metatensor-torch-cxx-*.tar.gz
cp "$ARCHIVE_NAME".tar.gz "$OUTPUT_DIR/"
