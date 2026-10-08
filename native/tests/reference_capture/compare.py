"""Exact one-token comparison. Diagnostics cannot turn a mismatch into a pass."""
import argparse
import json
from .artifacts import compare, read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('expected')
    parser.add_argument('actual')
    args = parser.parse_args()
    try:
        report = compare(read(args.expected), read(args.actual))
        print(json.dumps(report, allow_nan=False, sort_keys=True))
        return 0 if report['accepted'] else 1
    except (ValueError, OSError) as error:
        print(json.dumps({'accepted': False, 'error': str(error)}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
