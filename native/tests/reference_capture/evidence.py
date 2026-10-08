"""Bounded regular-file evidence validation; also runs as an isolated helper."""
import argparse
import json
import os
from pathlib import Path
import stat

from .artifacts import decode, require, sha256

MAX_DIAGNOSTIC = 8 * 1024**2
MAX_MANIFEST = 1024**2


def read_regular(path, maximum, exact=None):
    """Open every component without following symlinks; FIFOs never block open."""
    path=Path(path).absolute()
    require('..' not in path.parts, 'evidence_parent_path')
    directory=os.open('/',os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    fd=None
    try:
        for part in path.parts[1:-1]:
            child=os.open(part,os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,dir_fd=directory)
            os.close(directory)
            directory=child
        fd=os.open(path.name,os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,dir_fd=directory)
        before=os.fstat(fd)
        require(stat.S_ISREG(before.st_mode), 'evidence_regular_file')
        require(0 <= before.st_size <= maximum and (exact is None or before.st_size==exact), 'evidence_size')
        chunks=[]
        remaining=maximum+1
        while remaining:
            chunk=os.read(fd,min(65536,remaining))
            if not chunk:break
            chunks.append(chunk)
            remaining-=len(chunk)
        data=b''.join(chunks)
        after=os.fstat(fd)
        identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
        require(len(data)==before.st_size and len(data)<=maximum
                and identity(before)==identity(after), 'evidence_changed')
        current=os.stat(path.name,dir_fd=directory,follow_symlinks=False)
        require(identity(current)==identity(after) and stat.S_ISREG(current.st_mode), 'evidence_replaced')
        return data
    finally:
        if fd is not None:os.close(fd)
        os.close(directory)


def validate(folder, manifest_digest):
    folder=Path(folder)
    data=read_regular(folder/'reference.capture',993316,exact=993316)
    record=decode(data)
    report=json.loads(read_regular(folder/'diagnostic.json',MAX_DIAGNOSTIC))
    require(record['input']==248044 and record['vocab']==248320
            and report['status']=='complete' and report['zero_reservations'] is True
            and report['calls']['model']==1 and report['capture_sha256']==sha256(data)
            and report['manifest_sha256']==manifest_digest, 'reference_artifacts')
    return True


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('operation',choices=('artifacts','manifest','smoke'))
    parser.add_argument('path')
    parser.add_argument('digest')
    args=parser.parse_args()
    try:
        if args.operation=='artifacts':validate(args.path,args.digest)
        elif args.operation=='smoke':
            require(read_regular(Path(args.path)/'stdout.raw',1024)==b'SMOKE_DONE 16777216\n','smoke_output')
            require(read_regular(Path(args.path)/'stderr.raw',1024)==b'','smoke_stderr')
        else:require(sha256(read_regular(args.path,MAX_MANIFEST))==args.digest,'manifest_changed')
        return 0
    except Exception:
        return 1


if __name__=='__main__':
    raise SystemExit(main())
