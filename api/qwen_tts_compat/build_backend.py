"""Build the pinned official Qwen package with Kadan's reviewed compatibility patch.

This PEP 517 backend runs only during normal pip installation, never at API
startup. It fetches a fixed upstream commit into a disposable directory, verifies
Git's content identity, applies the visible patch once, and delegates wheel
creation to setuptools. The wheel retains upstream code, license and attribution;
its metadata records the actual compatible runtime dependencies.
"""
from pathlib import Path
import subprocess
import tempfile
import os

from setuptools import build_meta

SOURCE = 'https://github.com/QwenLM/Qwen3-TTS.git'
REVISION = '022e286b98fbec7e1e916cb940cdf532cd9f488e'
PATCH = Path(__file__).with_name('compatibility.patch')


def _git(directory, *args):
    return subprocess.check_output(['git', '-C', str(directory), *args], text=True).strip()


def _prepare_source(directory):
    """Fail closed on a different revision or a patch that no longer applies."""
    _git(directory, 'init', '--quiet')
    _git(directory, 'remote', 'add', 'origin', SOURCE)
    _git(directory, 'fetch', '--quiet', '--depth=1', 'origin', REVISION)
    _git(directory, 'checkout', '--quiet', '--detach', 'FETCH_HEAD')
    if _git(directory, 'rev-parse', 'HEAD') != REVISION:
        raise RuntimeError('Unexpected Qwen source revision')
    _git(directory, 'fsck', '--no-reflogs', '--strict')
    _git(directory, 'apply', '--check', str(PATCH))
    _git(directory, 'apply', str(PATCH))


def get_requires_for_build_wheel(config_settings=None):
    return []


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    """pip's metadata fallback reuses this wheel, so patch/build happens once."""
    destination = str(Path(wheel_directory).resolve())
    previous = Path.cwd()
    with tempfile.TemporaryDirectory(prefix='kadan-qwen-build-') as directory:
        _prepare_source(directory)
        try:
            os.chdir(directory)
            return build_meta.build_wheel(destination, config_settings)
        finally:
            os.chdir(previous)
