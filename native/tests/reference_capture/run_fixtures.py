"""Persist CPU synthetic evidence; exit nonzero on ANY exact parity failure."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from .artifacts import compare, read, require, sha256
from .fixtures import create

CASES = {'tiny': (False, False), 'representative': (True, False), 'tie': (False, True)}


def run_case(base, name):
    root = base / name
    root.mkdir()
    representative, tied = CASES[name]
    create(root, representative, tied)
    capture, diagnostic = base / (name + '.capture'), base / (name + '.json')
    result = subprocess.run(
        [sys.executable, '-B', '-m', 'native.tests.reference_capture.capture',
         '--synthetic-root', str(root), '--token', '2', '--output', str(capture),
         '--diagnostic', str(diagnostic)], capture_output=True, text=True, timeout=90)
    (base / (name + '.stdout')).write_text(result.stdout)
    (base / (name + '.stderr')).write_text(result.stderr)
    require(result.returncode == 0, 'capture_exit:' + str(result.returncode))
    report = json.loads(diagnostic.read_text())
    require(report['capture_sha256'] == sha256(capture.read_bytes()), 'capture_hash')
    require(report['zero_reservations'] and report['cache_length'] == 1, 'cleanup_or_cache')
    require(report['calls'] == {'model': 1, 'chunk': 3, 'recurrent': 0, 'conv': 3, 'conv_update': 0}, 'calls')
    require(len(report['layers']) == len(report['routes']) == len(report['cache']) == 4, 'layers')
    require(report['effective_threads'] == 2 and report['effective_interop_threads'] == 1, 'threads')
    for route in report['routes']:
        require(len(route['ids']) == (8 if representative else 2), 'route_count')
        if tied:
            require(route['logits'] == [0.] * 4 and route['boundary_tie'], 'tie_fixture')
            require(route['weights'] == [.5, .5], 'tie_weights')
    expected, actual = read(root / 'one-token-equations.capture'), read(capture)
    comparison = compare(expected, actual)
    return dict(comparison, capture_exit=result.returncode,
                expected_selected=expected['selected'], actual_selected=actual['selected'],
                expected_sha256=sha256((root / 'one-token-equations.capture').read_bytes()),
                actual_sha256=report['capture_sha256'],
                router_ids=[r['ids'] for r in report['routes']],
                router_boundary_ties=[r['boundary_tie'] for r in report['routes']])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True, help='New evidence directory; never overwritten')
    parser.add_argument('--cases', nargs='+', choices=CASES, default=list(CASES))
    args = parser.parse_args()
    args.out.mkdir()  # Refuse to mix with previous run artifacts.
    results = {}
    for name in args.cases:
        try:
            results[name] = run_case(args.out, name)
        except Exception as error:
            results[name] = {'accepted': False, 'error': str(error)}
        print(json.dumps({name: results[name]}, allow_nan=False), flush=True)
    (args.out / 'results.json').write_text(json.dumps(results, indent=2, allow_nan=False) + '\n')
    return 0 if all(r['accepted'] for r in results.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
