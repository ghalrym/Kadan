"""Bounded local evidence for reviewed operators; no workload on import.

Use a dedicated host evidence bind, never the temporary container writable layer.
A completed append is fsynced. A killed append may leave one incomplete last line;
readers must report that truncation, not infer a completed event from it.
"""
import fcntl
import json
import logging
import os
from pathlib import Path
import stat
import time


def append(path, record, *, cap=4*1024*1024, record_cap=1024*1024+8192):
    data=(json.dumps(record,allow_nan=False,separators=(',',':'))+'\n').encode()
    if len(data)>record_cap:
        raise ValueError('evidence_record_limit')
    path=Path(path)
    fd=os.open(path,os.O_WRONLY|os.O_APPEND|os.O_CREAT|os.O_NOFOLLOW|os.O_CLOEXEC,0o600)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX)
        info=os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size+len(data)>cap:
            raise ValueError('evidence_file_limit_or_type')
        remaining=memoryview(data)
        while remaining:
            count=os.write(fd,remaining)
            if count<=0:raise OSError('evidence_short_write')
            remaining=remaining[count:]
        os.fsync(fd)
        directory=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_CLOEXEC)
        try:os.fsync(directory)
        finally:os.close(directory)
    finally:os.close(fd)


class ImageEvents(logging.Handler):
    """One API producer; only small structured progress/timing/publication events.

    Errors propagate: missing evidence is not silently accepted. The independent
    host watchdog still owns bounded physical cleanup if producer I/O stalls.
    """
    def __init__(self, path):
        super().__init__();self.path=path

    def emit(self, record):
        message=record.getMessage()
        prefix=message.split(' ',1)[0]
        if prefix not in ('image_step_returned','dual_image_timing','dual_image_published'):
            return
        append(self.path,dict(unix_time=time.time(),monotonic=time.monotonic(),
            logger=record.name,message=message),cap=256*1024,record_cap=8192)
