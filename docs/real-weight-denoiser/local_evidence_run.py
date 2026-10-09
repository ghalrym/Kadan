"""Launch an explicitly reviewed operator under the local user service manager.

No shell, background workload, automatic retry or GPU work on import. The command
must perform its own exact-source/review/admission checks. This only decouples its
lifetime from the requesting terminal/remote exec session. Inspect the unit and
receipts after a disconnect; never submit a second workload to recover logging.
"""
import argparse
from pathlib import Path
import re
import subprocess


def command(unit, output, argv):
    if not re.fullmatch(r'kadan-reviewed-[a-z0-9-]{1,64}',unit):
        raise ValueError('invalid_unit')
    output=Path(output).resolve(strict=True)
    if not output.is_dir() or not argv or not Path(argv[0]).is_absolute():
        raise ValueError('absolute_command_and_existing_output_required')
    # 1800s is the existing operator budget; 60s allows its SIGTERM cleanup.
    # File retention is bounded by the operator, not by service stdout.
    return ['systemd-run','--user','--unit='+unit,'--service-type=exec',
        '--property=RuntimeMaxSec=1800','--property=TimeoutStopSec=60',
        '--property=KillMode=mixed','--property=Restart=no',
        '--property=StandardOutput=null','--property=StandardError=journal',
        '--property=WorkingDirectory='+str(output),'--',*argv]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--unit',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('argv',nargs=argparse.REMAINDER)
    args=parser.parse_args();argv=args.argv
    if argv and argv[0]=='--':argv=argv[1:]
    subprocess.run(command(args.unit,args.output,argv),check=True,timeout=15)


if __name__=='__main__':main()
