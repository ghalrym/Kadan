"""Owner-scoped HTTP over Unix sockets; no network listener or Docker proxy.

Public socket accepts status, pair, install and cancel only. The separately mounted
private admin socket issues pairing codes/recovery locally. Browser Origin is an
explicit setup value, never derived from Host or forwarded headers.
"""
import hashlib
import hmac
from http.cookies import SimpleCookie, CookieError
from http.server import BaseHTTPRequestHandler
import json
import secrets
import socketserver
import threading
import time
from urllib.parse import urlsplit

from updater.docker import atomic_json


class Access:
    def __init__(self, root, origin, clock=time.time):
        parsed = urlsplit(origin)
        local = parsed.hostname in ('localhost', '127.0.0.1', '::1')
        if (parsed.scheme not in ('http', 'https') or (parsed.scheme == 'http' and not local)
                or parsed.path or parsed.query or parsed.fragment or parsed.username or not parsed.netloc):
            raise ValueError('Set an exact HTTPS browser origin, or a localhost HTTP origin')
        self.root, self.origin, self.clock = root, origin, clock
        self.lock = threading.Lock()
        self.code = None
        self.code_until = 0
        self.attempts = []
        path = root / 'access.json'
        self.sessions = json.loads(path.read_text()) if path.exists() else {}

    @staticmethod
    def digest(value):
        return hashlib.sha256(value.encode()).hexdigest()

    def issue_code(self):
        with self.lock:
            code = secrets.token_urlsafe(32)
            self.code, self.code_until = self.digest(code), self.clock() + 300
            self.attempts = []
            return code

    def origin_ok(self, headers):
        return (headers.get('Origin') == self.origin and headers.get('X-Kadan-Update') == '1'
                and headers.get('Sec-Fetch-Site', 'same-origin') == 'same-origin')

    def authorized(self, headers):
        cookie = SimpleCookie()
        try:
            cookie.load(headers.get('Cookie', ''))
            token = cookie['kadan_updater'].value
        except (KeyError, ValueError, CookieError):
            return False
        with self.lock:
            return self.sessions.get(self.digest(token), 0) > self.clock()

    def pair(self, code):
        with self.lock:
            now = self.clock()
            self.attempts = [item for item in self.attempts if now - item < 300]
            if len(self.attempts) >= 8:
                raise ValueError('Pairing is temporarily locked; issue a new code locally')
            self.attempts.append(now)
            if (not isinstance(code, str) or not self.code or now >= self.code_until
                    or not hmac.compare_digest(self.digest(code), self.code)):
                raise ValueError('Invalid or expired pairing code')
            self.code = None
            token = secrets.token_urlsafe(32)
            # A fresh local pairing revokes the previous browser authorization.
            self.sessions = {self.digest(token): now + 12 * 3600}
            atomic_json(self.root / 'access.json', self.sessions)
            secure = '; Secure' if self.origin.startswith('https:') else ''
            return f'kadan_updater={token}; HttpOnly; SameSite=Strict; Path=/v1/updates; Max-Age=43200{secure}'


class UnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


def handler(engine, access, admin=False):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # Never log cookies, pairing codes or request bodies.

        def handle_one_request(self):
            self.connection.settimeout(5)
            super().handle_one_request()

        def reply(self, status, body, cookie=None):
            raw = json.dumps(body).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(raw)))
            self.send_header('Cache-Control', 'no-store')
            if cookie:
                self.send_header('Set-Cookie', cookie)
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path != '/status' or admin:
                return self.reply(404, {'detail': 'Unknown operation'})
            self.reply(200, engine.status(access.authorized(self.headers)))

        def do_POST(self):
            if admin:
                if self.path == '/pair':
                    return self.reply(200, {'code': access.issue_code(), 'expires_in': 300})
                if self.path == '/recover':
                    # Recovery remains explicit and local. Return acceptance so
                    # the CLI can poll status instead of timing out a long load.
                    threading.Thread(target=engine.recover, daemon=True).start()
                    return self.reply(202, {'message': 'Recovery requested; inspect status.'})
                return self.reply(404, {'detail': 'Unknown local operation'})
            if not access.origin_ok(self.headers):
                return self.reply(403, {'detail': 'Use the explicitly configured Kadan browser origin'})
            if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                return self.reply(415, {'detail': 'JSON is required'})
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 <= length <= 1024:
                    raise ValueError('Update request is too large')
                body = json.loads(self.rfile.read(length)) if length else None
                if self.path == '/pair':
                    if not isinstance(body, dict) or set(body) != {'code'}:
                        raise ValueError('Pairing requires a code')
                    cookie = access.pair(body['code'])
                    return self.reply(200, engine.status(True), cookie)
                if not access.authorized(self.headers):
                    return self.reply(401, {'detail': 'Pair this browser using a local server code'})
                if self.path == '/install':
                    if not isinstance(body, dict) or set(body) != {'commit'}:
                        raise ValueError('Choose the displayed release')
                    engine.start(body['commit'])
                    return self.reply(202, engine.status(True))
                if self.path == '/cancel' and body is None:
                    engine.request_cancel()
                    return self.reply(200, engine.status(True))
                return self.reply(404, {'detail': 'Unknown operation'})
            except (ValueError, TypeError) as exc:
                return self.reply(409, {'detail': str(exc)})
    return Handler
