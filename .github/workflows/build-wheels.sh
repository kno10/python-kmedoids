#!/bin/bash
set -e -x

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
