# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""The release installer's Hub download with the real huggingface_hub client.

A local HTTP server answers the requests huggingface_hub makes for a
repository lookup (``/api/<type>s/<repo>/revision/<rev>``) and for each file
(``HEAD`` and ``GET`` ``[datasets/]<repo>/resolve/<commit>/<path>``). This
runs in the complete uv.lock environment, which holds huggingface_hub; the
minimal distribution-contract environment does not.
"""

import hashlib
import http.server
import json
import shutil
import socket
import ssl
import struct
import subprocess
import threading
import types
from urllib.parse import unquote

import pytest

pytest.importorskip("huggingface_hub")

from real2sim import release_assets  # noqa: E402

POLICIES = "airoa-org/yubi-real2sim-policies"
ASSETS = "airoa-org/yubi-real2sim-assets"
COMMIT = "3" * 40
FILES = {
    "pi05_fixture/params/weights.bin": b"weights\n" * 4096,
    "pi05_fixture/assets/norm_stats.json": b"{}\n",
    "pi05_other/params/weights.bin": b"not requested\n",
}


class _Hub(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self):
        super().__init__(("127.0.0.1", 0), _Handler)
        self.repo, self.prefix = POLICIES, ""
        self.files = dict(FILES)
        self.refs = {"main": COMMIT, COMMIT: COMMIT}
        self.refusal: str | None = None
        # Listed by the repository lookup but not served: a 404 EntryNotFound.
        self.unserved: list[str] = []
        # Statuses answered, in order, before serving "api" or "resolve" requests.
        self.statuses: dict[str, list[int]] = {"api": [], "resolve": []}
        self.sleeps: list[float] = []
        self.requests: list[tuple[str, str]] = []

    @property
    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"


class _Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _send(self, status, body=b"", headers=None, *, head=False):
        self.send_response(status)
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if not head:
            self.wfile.write(body)

    def _error(self, status, code, *, head=False):
        body = json.dumps({"error": code}).encode()
        self._send(status, body, {"X-Error-Code": code, "Content-Type": "application/json"}, head=head)

    def _send_status(self, status, head):
        body = json.dumps({"error": f"status {status}"}).encode()
        self._send(status, body, {"Content-Type": "application/json"}, head=head)

    def _route(self, head):
        hub = self.server
        path = unquote(self.path.split("?")[0])
        hub.requests.append((self.command, path))
        if hub.refusal:
            return self._error(403 if hub.refusal == "GatedRepo" else 404, hub.refusal, head=head)
        kind = "datasets" if hub.prefix else "models"
        api = f"/api/{kind}/{hub.repo}/revision/"
        if path.startswith(api):
            if hub.statuses["api"]:
                return self._send_status(hub.statuses["api"].pop(0), head)
            commit = hub.refs.get(path[len(api):])
            if commit is None:
                return self._error(404, "RevisionNotFound", head=head)
            body = json.dumps({"id": hub.repo, "sha": commit, "private": False,
                               "siblings": [{"rfilename": name}
                                            for name in [*hub.files, *hub.unserved]]}).encode()
            return self._send(200, body, {"Content-Type": "application/json"}, head=head)
        resolve = f"/{hub.prefix}{hub.repo}/resolve/"
        if not path.startswith(resolve):
            return self._error(404, "RepoNotFound", head=head)
        if hub.statuses["resolve"]:
            return self._send_status(hub.statuses["resolve"].pop(0), head)
        revision, _, name = path[len(resolve):].partition("/")
        if hub.refs.get(revision) is None:
            return self._error(404, "RevisionNotFound", head=head)
        data = hub.files.get(name)
        if data is None:
            return self._error(404, "EntryNotFound", head=head)
        headers = {"X-Repo-Commit": hub.refs[revision],
                   "ETag": f'"{hashlib.sha256(data).hexdigest()}"',
                   "Content-Type": "application/octet-stream"}
        start = 0
        if self.headers.get("Range"):
            start = int(self.headers["Range"].removeprefix("bytes=").split("-")[0])
        self._send(206 if start else 200, data[start:], headers, head=head)

    def do_GET(self):
        self._route(head=False)

    def do_HEAD(self):
        self._route(head=True)


@pytest.fixture
def hub(monkeypatch, tmp_path):
    server = _Hub()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("HF_ENDPOINT", server.endpoint)
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "1")
    # Only the installer's retry delays; huggingface_hub keeps its own short waits.
    monkeypatch.setattr(release_assets, "time", types.SimpleNamespace(sleep=server.sleeps.append))
    yield server
    server.shutdown()
    server.server_close()


def _repo(revision=COMMIT):
    return release_assets.HubRepo("policies", POLICIES, "model", revision,
                                  (("{inputs_root}/checkpoints/", ""),))


def test_only_the_requested_folder_is_downloaded_at_the_pinned_commit(hub, tmp_path):
    tree = release_assets.download_tree(_repo(), "pi05_fixture", tmp_path / "policies", timeout=10)
    assert tree == tmp_path / "policies/pi05_fixture"
    got = {path.relative_to(tmp_path / "policies").as_posix(): path.read_bytes()
           for path in tree.rglob("*") if path.is_file()}
    assert got == {name: data for name, data in FILES.items() if name.startswith("pi05_fixture/")}
    files = [path for method, path in hub.requests if "/resolve/" in path]
    assert files and all(f"/resolve/{COMMIT}/pi05_fixture/" in path for path in files)
    assert ("GET", f"/api/models/{POLICIES}/revision/{COMMIT}") in hub.requests


def test_a_dataset_repository_is_addressed_as_one(hub, tmp_path):
    hub.repo, hub.prefix = ASSETS, "datasets/"
    hub.files = {"scenes/flat_a_room/background.json": b"{}\n"}
    repo = release_assets.HubRepo("assets", ASSETS, "dataset", COMMIT, ())
    tree = release_assets.download_tree(repo, "scenes/flat_a_room", tmp_path / "assets", timeout=10)
    assert (tree / "background.json").read_bytes() == b"{}\n"
    assert ("GET", f"/api/datasets/{ASSETS}/revision/{COMMIT}") in hub.requests


def test_a_gated_repository_asks_for_login_and_the_terms(hub, tmp_path):
    hub.refusal = "GatedRepo"
    (tmp_path / "policies/pi05_other").mkdir(parents=True)
    (tmp_path / "policies/pi05_other/kept.bin").write_bytes(b"x")
    with pytest.raises(release_assets.ReleaseAssetError) as caught:
        release_assets.download_tree(_repo(), "pi05_fixture", tmp_path / "policies", timeout=10)
    message = str(caught.value)
    assert ".venv/bin/hf auth login" in message and f"https://huggingface.co/{POLICIES}" in message
    assert len(hub.requests) == 1


def test_an_unknown_revision_is_named(hub, tmp_path):
    with pytest.raises(release_assets.ReleaseAssetError, match=f"has no revision '{'4' * 40}'"):
        release_assets.download_tree(_repo("4" * 40), "pi05_fixture", tmp_path / "policies", timeout=10)


def test_a_missing_folder_is_reported(hub, tmp_path):
    with pytest.raises(release_assets.ReleaseAssetError, match="holds no files"):
        release_assets.download_tree(_repo(), "pi05_absent", tmp_path / "policies", timeout=10)


@pytest.mark.parametrize("where", ["api", "resolve"])
def test_a_bad_request_fails_at_once_with_its_status(hub, tmp_path, where):
    # huggingface_hub's HTTP errors are OSErrors; a 400 must still not be retried.
    hub.statuses[where] = [400] * release_assets.MAX_ATTEMPTS
    with pytest.raises(release_assets.ReleaseAssetError, match="returned HTTP 400: ") as caught:
        release_assets.download_tree(_repo(), "pi05_fixture", tmp_path / "policies", timeout=10)
    assert "attempts" not in str(caught.value)
    assert hub.sleeps == []


def test_a_listed_file_the_hub_does_not_serve_fails_at_once(hub, tmp_path):
    hub.unserved = ["pi05_fixture/params/missing.bin"]
    with pytest.raises(release_assets.ReleaseAssetError, match="returned HTTP 404: ") as caught:
        release_assets.download_tree(_repo(), "pi05_fixture", tmp_path / "policies", timeout=10)
    assert "Entry Not Found" in str(caught.value)
    assert hub.sleeps == []


@pytest.mark.parametrize("where", ["api", "resolve"])
def test_too_many_requests_is_retried_and_the_download_completes(hub, tmp_path, where):
    hub.statuses[where] = [429]
    tree = release_assets.download_tree(_repo(), "pi05_fixture", tmp_path / "policies", timeout=10)
    assert (tree / "params/weights.bin").read_bytes() == FILES["pi05_fixture/params/weights.bin"]
    assert hub.sleeps == [1]


def _closed_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def test_a_refused_connection_is_retried_without_a_wait_after_the_last_attempt(
        hub, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_ENDPOINT", f"http://127.0.0.1:{_closed_port()}")
    with pytest.raises(release_assets.ReleaseAssetError,
                       match=f"failed after {release_assets.MAX_ATTEMPTS} attempts"):
        release_assets.download_tree(_repo(), "pi05_fixture", tmp_path / "policies", timeout=10)
    assert hub.sleeps == [min(2**attempt, 30) for attempt in range(release_assets.MAX_ATTEMPTS - 1)]


RETRY_WAITS = [min(2**attempt, 30) for attempt in range(release_assets.MAX_ATTEMPTS - 1)]
SETUP_HINT = ("check the proxy variables (HTTPS_PROXY, NO_PROXY), the CA bundle "
              "(REQUESTS_CA_BUNDLE) and the system clock")


class _Socket:
    """A local TCP server that passes each connection to ``answer`` and closes it."""

    def __init__(self, answer):
        self.listener = socket.create_server(("127.0.0.1", 0))
        self.listener.settimeout(0.2)
        self.port = self.listener.getsockname()[1]
        self.answer = answer
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self):
        while not self.stopped.is_set():
            try:
                connection, _ = self.listener.accept()
            except TimeoutError:
                continue
            with connection:
                connection.settimeout(10)
                try:
                    self.answer(connection)
                except OSError:
                    pass

    def close(self):
        self.stopped.set()
        self.thread.join(5)
        self.listener.close()


@pytest.fixture
def serve():
    servers = []

    def start(answer):
        servers.append(_Socket(answer))
        return servers[-1].port

    yield start
    for server in servers:
        server.close()


@pytest.fixture(scope="module")
def untrusted_tls(tmp_path_factory):
    """A TLS server context with a self-signed certificate for 127.0.0.1."""
    openssl = shutil.which("openssl")
    if openssl is None:
        pytest.skip("the openssl command is needed to make a self-signed certificate")
    folder = tmp_path_factory.mktemp("tls")
    subprocess.run(
        [openssl, "req", "-x509", "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:prime256v1",
         "-nodes", "-keyout", str(folder / "key.pem"), "-out", str(folder / "cert.pem"),
         "-days", "2", "-subj", "/CN=127.0.0.1", "-addext", "subjectAltName=IP:127.0.0.1"],
        check=True, capture_output=True)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(folder / "cert.pem", folder / "key.pem")
    return context


@pytest.fixture
def default_trust(monkeypatch):
    for name in ("REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "SSL_CERT_FILE", "SSL_CERT_DIR"):
        monkeypatch.delenv(name, raising=False)


def _proxy_only(monkeypatch, scheme, proxy):
    for name in ("NO_PROXY", "no_proxy", "ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy",
                 "HTTPS_PROXY", "https_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(f"{scheme.upper()}_PROXY", proxy)
    monkeypatch.setenv(f"{scheme}_proxy", proxy)


def _complete_handshake(context):
    def answer(connection):
        with context.wrap_socket(connection, server_side=True) as tls:
            tls.recv(1)
    return answer


def test_an_untrusted_certificate_fails_at_once_and_names_the_setup(
        hub, tmp_path, monkeypatch, serve, untrusted_tls, default_trust):
    monkeypatch.setenv("HF_ENDPOINT", f"https://127.0.0.1:{serve(_complete_handshake(untrusted_tls))}")
    with pytest.raises(release_assets.ReleaseAssetError,
                       match="could not be reached because its TLS certificate could not be "
                             "verified") as caught:
        release_assets.download_tree(_repo(), "pi05_fixture", tmp_path / "policies", timeout=10)
    assert "CERTIFICATE_VERIFY_FAILED" in str(caught.value) and SETUP_HINT in str(caught.value)
    assert "attempts" not in str(caught.value)
    assert type(caught.value.__cause__).__name__ == "SSLError"
    assert hub.sleeps == []


def _cut_after_client_hello(connection):
    connection.recv(4096)


def _reset_after_client_hello(connection):
    connection.recv(4096)
    connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))


def _plain_http(connection):
    connection.recv(4096)
    connection.sendall(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n")


# requests raises SSLError for the first two and ConnectionError for the reset.
@pytest.mark.parametrize("answer,raised,tls_error", [
    (_cut_after_client_hello, "SSLError(SSLEOFError", True),
    (_plain_http, "SSLError(SSLError(1, '[SSL: WRONG_VERSION_NUMBER]", True),
    (_reset_after_client_hello, "ConnectionResetError(104, 'Connection reset by peer')", False),
], ids=["closed-mid-handshake", "plain-http", "reset-mid-handshake"])
def test_a_failed_tls_handshake_is_retried(
        hub, tmp_path, monkeypatch, serve, default_trust, answer, raised, tls_error):
    monkeypatch.setenv("HF_ENDPOINT", f"https://127.0.0.1:{serve(answer)}")
    with pytest.raises(release_assets.ReleaseAssetError,
                       match=f"failed after {release_assets.MAX_ATTEMPTS} attempts") as caught:
        release_assets.download_tree(_repo(), "pi05_fixture", tmp_path / "policies", timeout=10)
    assert raised in str(caught.value)
    assert (SETUP_HINT in str(caught.value)) is tls_error
    assert hub.sleeps == RETRY_WAITS


def _proxy_answers(status):
    challenge = 'Proxy-Authenticate: Basic realm="proxy"\r\n' if status.startswith("407") else ""
    def answer(connection):
        connection.recv(4096)
        connection.sendall(f"HTTP/1.1 {status}\r\n{challenge}Content-Length: 0\r\n\r\n".encode())
    return answer


def _hub_endpoint(hub, scheme):
    # Over https the request needs a CONNECT tunnel, which these proxies never open.
    return hub.endpoint.replace("http://", f"{scheme}://")


@pytest.mark.parametrize("scheme", ["https", "http"])
def test_a_proxy_that_asks_for_authentication_fails_at_once(
        hub, tmp_path, monkeypatch, serve, scheme):
    # Over https the proxy refuses the CONNECT tunnel; over http it answers the request.
    _proxy_only(monkeypatch, scheme,
                f"http://127.0.0.1:{serve(_proxy_answers('407 Proxy Authentication Required'))}")
    monkeypatch.setenv("HF_ENDPOINT", _hub_endpoint(hub, scheme))
    with pytest.raises(release_assets.ReleaseAssetError,
                       match=r"because the proxy requires authentication \(HTTP 407\)") as caught:
        release_assets.download_tree(_repo(), "pi05_fixture", tmp_path / "policies", timeout=10)
    assert SETUP_HINT in str(caught.value) and "attempts" not in str(caught.value)
    assert hub.requests == [] and hub.sleeps == []


@pytest.mark.parametrize("scheme,refusal", [
    ("https", "503 Service Unavailable"), ("https", None), ("http", None),
], ids=["tunnel-503", "https-refused", "http-refused"])
def test_an_unavailable_proxy_is_retried_and_named_after_the_last_attempt(
        hub, tmp_path, monkeypatch, serve, scheme, refusal):
    port = serve(_proxy_answers(refusal)) if refusal else _closed_port()
    _proxy_only(monkeypatch, scheme, f"http://127.0.0.1:{port}")
    monkeypatch.setenv("HF_ENDPOINT", _hub_endpoint(hub, scheme))
    with pytest.raises(release_assets.ReleaseAssetError,
                       match=f"failed after {release_assets.MAX_ATTEMPTS} attempts") as caught:
        release_assets.download_tree(_repo(), "pi05_fixture", tmp_path / "policies", timeout=10)
    assert "ProxyError" in str(caught.value) and SETUP_HINT in str(caught.value)
    assert hub.requests == [] and hub.sleeps == RETRY_WAITS
