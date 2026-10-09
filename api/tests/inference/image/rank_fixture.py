"""CPU-only protocol fixture. Never imported by the production rank worker."""
import json
import os
import socket
import sys
import time

from api.inference.image.rank_transport import encode, receive_blocking


def main():
    config = json.loads(sys.argv[2])
    channel = socket.socket(fileno=int(sys.argv[1]))
    while True:
        command = receive_blocking(channel)
        marker = command.get('payload', {}).get('prompt', '')
        if marker == 'stderr-exit' and config['rank'] == 1:
            sys.stderr.write('x' * 100000 + 'fixture peer failure detail')
            sys.stderr.flush()
            os._exit(2)
        if marker == 'peer-exit' and config['rank'] == 1:
            os._exit(2)
        if marker == 'wait':
            time.sleep(60)
        reply = dict(command, rank=config['rank'], device=config['device'], status='ok',
            resident_bytes=0 if command['operation']=='park' else 1)
        if marker == 'oom' and config['rank'] == 1:
            reply.update(status='error',error='OutOfMemoryError: synthetic allocation failure')
        if marker == 'stale' and config['rank'] == 1:
            reply['sequence'] -= 1
        channel.sendall(encode(reply))


if __name__ == '__main__':
    main()
