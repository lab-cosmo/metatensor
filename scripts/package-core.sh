#!/usr/bin/env bash

# This script creates an archive containing the sources for the metatensor-core
# Rust crate, and copy it to the path given as argument

set -eux

OUTPUT_DIR="$1"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR=$(cd "$OUTPUT_DIR" 2>/dev/null && pwd)

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)

rm -rf "$ROOT_DIR/target/package"
cd "$ROOT_DIR/metatensor-core"

# Package metatensor-core using cargo tools, and add a file for
# `n_commits_since_last_tag`
cargo package --allow-dirty --no-verify

TMP_DIR=$(mktemp -d)

cd "$TMP_DIR"
tar xf "$ROOT_DIR"/target/package/metatensor-core-*.crate
ARCHIVE_NAME=$(ls)

# extract the version part of the package from the .crate file name
VERSION=${ARCHIVE_NAME:16}
ARCHIVE_NAME="metatensor-core-cxx-$VERSION"

mv metatensor-core-* "$ARCHIVE_NAME"
cp "$ROOT_DIR/LICENSE" "$TMP_DIR/$ARCHIVE_NAME"
cp "$ROOT_DIR/AUTHORS" "$TMP_DIR/$ARCHIVE_NAME"
cp "$ROOT_DIR/README.md" "$TMP_DIR/$ARCHIVE_NAME"

# Get the number of commits since last tag, this is used when building the
# code to change the version for development builds
cd "$ROOT_DIR"
./scripts/n-commits-since-last-tag.py "metatensor-core-v" > "$TMP_DIR/$ARCHIVE_NAME/cmake/n_commits_since_last_tag"

cd "$TMP_DIR"
# Compile metatensor-core as it's own Cargo workspace (otherwise we can not the
# use metatensor rust crate in a project using workspaces).
echo "[workspace]" >> "$ARCHIVE_NAME/Cargo.toml"

cargo generate-lockfile --manifest-path "$ARCHIVE_NAME/Cargo.toml"

tar cf "$ARCHIVE_NAME.tar" "$ARCHIVE_NAME"
gzip -9 "$ARCHIVE_NAME.tar"

rm -f "$ROOT_DIR"/python/metatensor-core/metatensor-core-cxx-*.tar.gz
cp "$TMP_DIR/$ARCHIVE_NAME.tar.gz" "$OUTPUT_DIR/"
