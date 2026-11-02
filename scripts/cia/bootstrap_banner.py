"""Prepare isolated, pinned tools for the relief banner (no game data or AI model)."""
import hashlib
import http.client
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import urllib.request
import venv
import zipfile

HERE = Path(__file__).resolve().parent
COMMIT = '1f78850086f3a77c41e07162e842f97a5bf3c18a'
URL = 'https://codeload.github.com/skyfloogle/pycgfx/zip/' + COMMIT
SHA256 = 'a822521761dbb428ee14df8d68340bd54595435a869fe835962f5b4d1c62c774'


def download_converter():
    for attempt in range(3):
        try:
            request = urllib.request.Request(URL, headers={'User-Agent': 'sms3ds-banner-builder'})
            with urllib.request.urlopen(request, timeout=30) as response:
                data = response.read()
            if hashlib.sha256(data).hexdigest() != SHA256:
                raise RuntimeError('Banner converter download checksum mismatch')
            return data
        except (OSError, http.client.HTTPException, RuntimeError) as exc:
            if attempt == 2:
                raise RuntimeError('Could not download the verified banner converter: %s' % exc) from exc
            print('Banner converter download interrupted; retrying...', flush=True)


def safe_destination(root, name):
    target = (root / name).resolve()
    if os.path.commonpath([str(root.resolve()), str(target)]) != str(root.resolve()):
        raise ValueError('Archive member escapes the banner tools directory')
    return target


def ensure_tools(cache):
    if sys.version_info < (3, 11):
        raise RuntimeError('The relief banner needs Python 3.11 or newer. '
                           'Use --banner flat with Python 3.8–3.10.')
    cache = Path(cache).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    source = cache / 'pycgfx'
    marker = source / '.source.json'
    expected = {'commit': COMMIT, 'archive_sha256': SHA256}
    if not marker.is_file() or json.loads(marker.read_text()) != expected:
        if source.exists():
            raise RuntimeError('Unrecognized pycgfx cache. Select a new --banner-tools directory.')
        archive = cache / 'pycgfx.zip'
        if not archive.is_file() or hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
            print('Downloading the pinned banner converter...', flush=True)
            archive.write_bytes(download_converter())
        # Verify before extracting, then publish only a complete source directory.
        with tempfile.TemporaryDirectory(dir=str(cache)) as tmp:
            staging = Path(tmp) / 'pycgfx'
            staging.mkdir()
            with zipfile.ZipFile(str(archive)) as zipped:
                for entry in zipped.infolist():
                    parts = Path(entry.filename).parts[1:]
                    if not parts or entry.is_dir():
                        continue
                    if (entry.external_attr >> 16) & 0o170000 == 0o120000:
                        raise ValueError('Symlink in banner converter archive')
                    target = safe_destination(staging, Path(*parts))
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(zipped.read(entry))
            if not (staging / 'main.py').is_file():
                raise RuntimeError('Incomplete banner converter archive')
            (staging / '.source.json').write_text(json.dumps(expected) + '\n')
            staging.rename(source)
    environment = cache / 'venv'
    python = environment / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if not python.is_file():
        print('Creating an isolated banner Python environment...', flush=True)
        venv.EnvBuilder(with_pip=True).create(str(environment))
    requirements = HERE / 'requirements-banner.txt'
    digest = hashlib.sha256(requirements.read_bytes()).hexdigest()
    stamp = environment / '.requirements.sha256'
    probe = subprocess.run([str(python), '-I', '-c', 'import PIL, gltflib'],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if probe.returncode or not stamp.is_file() or stamp.read_text().strip() != digest:
        print('Installing the banner image/converter packages locally...', flush=True)
        subprocess.run([str(python), '-m', 'pip', 'install', '--disable-pip-version-check',
                        '--cache-dir', str(cache / 'pip-cache'), '-r', str(requirements)], check=True)
        stamp.write_text(digest + '\n')
    return python, source


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=HERE / '.banner-tools')
    args = parser.parse_args()
    try:
        python, source = ensure_tools(args.cache)
        print('Banner tools ready: ' + str(python))
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        parser.exit(1, str(exc) + '\nUse --banner flat for the dependency-free banner.\n')
