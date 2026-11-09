# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Loopback GS traffic must not be routed through an environment HTTP proxy."""

from __future__ import annotations

import contextlib
import http.server
import json
import os
import subprocess
import sys
import threading
import urllib.request
from collections.abc import Iterator
from unittest import mock

import pytest

import robosuite_env.tools.rerender_third_person as rerender
from robosuite_env import gs_launch


_PROXY_NAMES = ("http_proxy", "HTTP_PROXY", "no_proxy", "NO_PROXY")


@contextlib.contextmanager
def _serve(handler: type[http.server.BaseHTTPRequestHandler]) -> Iterator[int]:
    server = http.server.HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield int(server.server_address[1])
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture
def health_server() -> Iterator[tuple[int, list[str]]]:
    tokens: list[str] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 -- http.server API
            tokens.append(self.headers.get(gs_launch.GS_HEALTH_TOKEN_HEADER, ""))
            body = json.dumps({"status": "ok"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args) -> None:
            pass

    with _serve(Handler) as port:
        yield port, tokens


@pytest.fixture
def dead_proxy(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[str, list[str]]]:
    """Export a proxy that records every request and answers 502.

    A listening trap, rather than a closed port, makes a proxied request fail
    the same way every time and shows which URL reached the proxy.
    """

    reached: list[str] = []

    class Trap(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 -- http.server API
            reached.append(self.path)
            self.send_response(502)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *_args) -> None:
            pass

    for name in _PROXY_NAMES:
        monkeypatch.delenv(name, raising=False)
    with _serve(Trap) as port:
        proxy = f"http://127.0.0.1:{port}"
        monkeypatch.setenv("http_proxy", proxy)
        monkeypatch.setenv("HTTP_PROXY", proxy)
        yield proxy, reached


def test_gs_health_probe_reaches_loopback_server_despite_proxy(
    health_server: tuple[int, list[str]],
    dead_proxy: tuple[str, list[str]],
) -> None:
    port, tokens = health_server
    _proxy, reached = dead_proxy

    identity = gs_launch.fetch_gs_health(port, ready_token="secret", timeout=5.0)

    assert identity == {"status": "ok"}
    # The token went to the local server, not to the proxy.
    assert tokens == ["secret"]
    assert reached == []


def test_replay_worker_reaches_loopback_gs_despite_inherited_proxy(
    health_server: tuple[int, list[str]],
    dead_proxy: tuple[str, list[str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    port, tokens = health_server
    proxy, reached = dead_proxy
    monkeypatch.setenv("no_proxy", "corp.example")
    binding = rerender.GpuBinding(
        allocation_slot=0,
        cuda_token="0",
        egl_physical_index=0,
    )

    environment = rerender.replay_worker_environment(binding)

    # Other hosts keep their proxy configuration.
    assert environment["http_proxy"] == proxy
    assert environment["no_proxy"].split(",")[0] == "corp.example"
    # A urllib client in the worker environment, like the GS client, reaches
    # the loopback server directly.
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import sys, urllib.request\n"
            "request = urllib.request.Request(sys.argv[1], "
            "headers={'X-Real2Sim-GS-Token': 'worker'})\n"
            "with urllib.request.urlopen(request, timeout=5) as response:\n"
            "    print(response.read().decode())\n",
            f"http://127.0.0.1:{port}/health",
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {"status": "ok"}
    assert tokens == ["worker"]
    assert reached == []


_PROXY = "http://proxy.example:3128"
_LOOPBACK_HOSTS = ("127.0.0.1", "127.0.0.1:8321", "localhost", "localhost:8321", "::1")
_OTHER_HOSTS = (
    "corp.example",
    "build.corp.example",
    "a.example",
    "b.example",
    "other.example",
    "10.0.0.5",
    "10.0.0.5:3128",
)

#: Environments a user can plausibly export, named for the case each covers.
_PROXY_ENVIRONMENTS = {
    "no-exclusions": {"http_proxy": _PROXY, "HTTP_PROXY": _PROXY},
    "uppercase-only-domain": {"http_proxy": _PROXY, "NO_PROXY": "corp.example"},
    "lowercase-only-domain": {"HTTP_PROXY": _PROXY, "no_proxy": ".corp.example"},
    "lowercase-wildcard": {"http_proxy": _PROXY, "no_proxy": "*"},
    "uppercase-wildcard": {"HTTP_PROXY": _PROXY, "NO_PROXY": "*"},
    "both-spellings-differ": {
        "http_proxy": _PROXY,
        "no_proxy": "a.example",
        "NO_PROXY": "b.example,corp.example",
    },
    "lowercase-wildcard-uppercase-domain": {
        "http_proxy": _PROXY,
        "no_proxy": "*",
        "NO_PROXY": "corp.example",
    },
    "empty-lowercase-hides-uppercase": {
        "http_proxy": _PROXY,
        "no_proxy": "",
        "NO_PROXY": "corp.example",
    },
    "mixed-case-name": {"http_proxy": _PROXY, "No_Proxy": "corp.example"},
}


def _urllib_view(environment: dict[str, str]) -> tuple[dict[str, str], dict[str, bool]]:
    """Return what urllib does with ``environment``: proxies and bypass per host."""

    with mock.patch.dict(os.environ, environment, clear=True):
        proxies = urllib.request.getproxies_environment()
        bypass = {
            host: bool(urllib.request.proxy_bypass_environment(host, proxies))
            for host in _LOOPBACK_HOSTS + _OTHER_HOSTS
        }
    return {k: v for k, v in proxies.items() if k != "no"}, bypass


@pytest.mark.parametrize(
    "environment",
    list(_PROXY_ENVIRONMENTS.values()),
    ids=list(_PROXY_ENVIRONMENTS),
)
def test_loopback_exclusion_keeps_every_other_host_as_it_was(
    environment: dict[str, str],
) -> None:
    before_proxies, before = _urllib_view(environment)

    after_proxies, after = _urllib_view(gs_launch.with_loopback_no_proxy(environment))

    assert after_proxies == before_proxies
    assert {host: after[host] for host in _OTHER_HOSTS} == {
        host: before[host] for host in _OTHER_HOSTS
    }
    assert all(after[host] for host in _LOOPBACK_HOSTS), after


def test_loopback_exclusion_keeps_each_spelling_and_wildcard() -> None:
    uppercase_only = gs_launch.with_loopback_no_proxy(
        _PROXY_ENVIRONMENTS["uppercase-only-domain"]
    )
    # A new lowercase value would take precedence and hide NO_PROXY.
    assert "no_proxy" not in uppercase_only
    assert uppercase_only["NO_PROXY"] == "corp.example,127.0.0.1,localhost,::1"

    wildcard = gs_launch.with_loopback_no_proxy(
        _PROXY_ENVIRONMENTS["lowercase-wildcard-uppercase-domain"]
    )
    assert wildcard["no_proxy"] == "*"
    assert wildcard["NO_PROXY"] == "corp.example,127.0.0.1,localhost,::1"

    both = gs_launch.with_loopback_no_proxy(
        _PROXY_ENVIRONMENTS["both-spellings-differ"]
    )
    assert both["no_proxy"] == "a.example,127.0.0.1,localhost,::1"
    assert both["NO_PROXY"] == "b.example,corp.example,127.0.0.1,localhost,::1"


def test_with_loopback_no_proxy_is_idempotent() -> None:
    once = gs_launch.with_loopback_no_proxy({"NO_PROXY": "a.example,127.0.0.1"})
    twice = gs_launch.with_loopback_no_proxy(once)

    assert twice == once
    assert once == {"NO_PROXY": "a.example,127.0.0.1,localhost,::1"}
