#!/bin/bash
# Decide whether the wheels in dist/ (or wheelhouse/) should go to PyPI: only if
# this wheel file is not on PyPI yet (adapted from blit's check-pypi-upload.sh).
# Writes perform_pypi_upload=0|1 to $GITHUB_OUTPUT.

PACKAGE=trussnet
WHEEL_FILE=$(ls dist/*.whl wheelhouse/*.whl 2>/dev/null | head -1)

if [ -z "$WHEEL_FILE" ]; then
    echo "No wheel file found"
    echo "perform_pypi_upload=0" >> "$GITHUB_OUTPUT"
    exit 0
fi

WHEEL_NAME=$(basename "$WHEEL_FILE")
echo "Built wheel: $WHEEL_NAME (version $(echo "$WHEEL_NAME" | awk -F- '{ print $2 }'))"

UPLOAD_TO_PYPI=1
PYPI_FILES=$(curl -sL "https://pypi.org/simple/${PACKAGE}/" 2>/dev/null | grep -oP '(?<=href=")[^"]+\.whl(?=")' | xargs -I {} basename {})

if [ -n "$PYPI_FILES" ] && echo "$PYPI_FILES" | grep -qF "$WHEEL_NAME"; then
    echo "$WHEEL_NAME is already on PyPI: skipping the upload"
    UPLOAD_TO_PYPI=0
else
    echo "$WHEEL_NAME is not on PyPI: will upload"
fi

echo "perform_pypi_upload=$UPLOAD_TO_PYPI" >> "$GITHUB_OUTPUT"
