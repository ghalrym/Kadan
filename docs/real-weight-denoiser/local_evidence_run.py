"""Run a reviewed operator under one absolute preflight/work/restoration budget."""
import argparse
from pathlib import Path
import re
import shlex
import subprocess
import time
from uuid import uuid4

from window_supervisor import context, write_once, STOP_GRACE


def command(unit,output,argv,*,started=None):
    if not re.fullmatch(r'kadan-reviewed-[a-z0-9-]{1,64}',unit):raise ValueError('invalid_unit')
    output=Path(output).resolve(strict=True)
    if not output.is_dir() or not argv or not Path(argv[0]).is_absolute():raise ValueError('absolute_command_and_existing_output_required')
    started=time.monotonic() if started is None else started
    ctx=context(started,uuid4().hex,Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    write_once(output/'window.json',ctx)
    helper=str(Path(__file__).with_name('window_supervisor.py').resolve())
    recovery=shlex.join(['/usr/bin/python3',helper,'--directory',str(output),'--recover']).replace('%','%%').replace('$','$$')
    return ['systemd-run','--user','--unit='+unit,'--service-type=exec',
        '--property=RuntimeMaxSec=900','--property=TimeoutStopSec='+str(STOP_GRACE),
        '--property=KillMode=mixed','--property=Restart=no',
        '--property=ExecStopPost='+recovery,
        '--property=StandardOutput=null','--property=StandardError=journal',
        '--property=WorkingDirectory='+str(output),'--','/usr/bin/python3',helper,
        '--directory',str(output),'--',*argv]


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--unit',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('argv',nargs=argparse.REMAINDER);args=parser.parse_args()
    argv=args.argv[1:] if args.argv and args.argv[0]=='--' else args.argv
    subprocess.run(command(args.unit,args.output,argv),check=True,timeout=15)


if __name__=='__main__':main()
