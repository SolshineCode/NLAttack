#!/usr/bin/env python3
"""Drive `hermes auth add nous` through a PTY so the TTY-only [Y/n] prompt can be
answered non-interactively. Declines importing the stale creds, forcing a fresh
OAuth device flow, prints the verification URL, and keeps polling (up to 10 min)
so the auth completes once the user approves the URL in a browser."""
import pty, os, select, time, sys

CMD = ["hermes", "auth", "add", "nous", "--type", "oauth", "--no-browser"]

pid, master = pty.fork()
if pid == 0:
    os.execvp(CMD[0], CMD)
    os._exit(1)

buf = b""
answered = False
start = time.time()
while time.time() - start < 600:
    r, _, _ = select.select([master], [], [], 1.0)
    if master in r:
        try:
            data = os.read(master, 4096)
        except OSError:
            break
        if not data:
            break
        buf += data
        sys.stdout.buffer.write(data)
        sys.stdout.flush()
        if not answered and b"[Y/n]" in buf:
            os.write(master, b"n\n")
            answered = True
    # exit early once the flow reports success
    if b"uccess" in buf or b"authenticated" in buf.lower() or b"logged in" in buf.lower():
        time.sleep(2)
        try:
            data = os.read(master, 4096)
            sys.stdout.buffer.write(data); sys.stdout.flush()
        except OSError:
            pass
        break
try:
    os.waitpid(pid, os.WNOHANG)
except OSError:
    pass
print("\n=== nous_login.py done ===")
