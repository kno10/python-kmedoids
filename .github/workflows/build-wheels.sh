#!/bin/bash
set -e -x

# update rust
curl https://sh.rustup.rs -sSf | sh -s -- -y
source $HOME/.cargo/env
rustup default stable

# Workaround for OOM issues, https://github.com/rust-lang/cargo/issues/10583
echo "[net]" >> "$HOME/.cargo/config.toml"
echo "git-fetch-with-cli = true" >> "$HOME/.cargo/config.toml"

# build wheels
for PYBIN in /opt/python/cp3[891]*/bin; do
    "${PYBIN}/pip" install -r requirements.txt -r requirements-dev.txt
    "${PYBIN}/maturin" build -i "${PYBIN}/python" --release --strip
done

# check/repair wheels
for wheel in target/wheels/*.whl; do
    auditwheel repair "${wheel}"
done

# test wheels
for PYBIN in /opt/python/cp3[891]*/bin; do
    "${PYBIN}/pip" install kmedoids --no-index --find-links wheelhouse
    cd tests && "${PYBIN}/python" -m unittest discover && cd ..
done
