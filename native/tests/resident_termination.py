"""Supervised termination reaches a leaf blocked inside step, not its command FIFO."""
import os
import selectors
import signal
import subprocess
import sys

nonce = '0123456789abcdef0123456789abcdef'
child = subprocess.Popen([sys.argv[1], '--block'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
try:
    def line():
        selector = selectors.DefaultSelector()
        selector.register(child.stdout, selectors.EVENT_READ)
        try:
            assert selector.select(5), 'worker output deadline'
            return child.stdout.readline()
        finally:
            selector.close()
    assert line().startswith(b'ready 2 ')
    for command in (f'submit {nonce} 0\n', f'start {nonce} 1\n', f'step {nonce} 1 2 0\n'):
        child.stdin.write(command.encode())
        child.stdin.flush()
        reply = line()
    assert reply == b'running\n'
    # The owned Popen child is still unreaped, preventing PID reuse here.
    os.kill(child.pid, signal.SIGTERM)
    assert child.wait(timeout=5) == -signal.SIGTERM
    assert b'token ' not in child.stdout.read()
finally:
    if child.poll() is None:
        child.kill()
        child.wait(timeout=5)
