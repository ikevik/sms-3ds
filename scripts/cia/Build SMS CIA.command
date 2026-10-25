#!/bin/bash
# macOS: double-click to build the HOME Menu .cia from your Super Mario Sunshine disc image.
cd "$(dirname "$0")"
if ! command -v python3 >/dev/null 2>&1; then
	echo "Python 3 is needed: https://www.python.org/downloads/"
else
	python3 -I build_cia.py "$@"
fi
echo
read -r -p "Press Return to close this window."
