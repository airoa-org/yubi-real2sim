# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Release downloads from the two Hugging Face repositories at pinned revisions.

``huggingface_hub.snapshot_download`` is replaced by a fake that serves
extracted folders from memory, so these tests need neither the network nor
huggingface_hub. tests/unit/provenance/test_release_hub_client.py runs the
real client against a local server.
"""

import fnmatch
import json
import re
from pathlib import Path
import shutil
import ssl

import pytest
import yaml

from real2sim import release_assets

POLICIES = "airoa-org/yubi-real2sim-policies"
ASSETS = "airoa-org/yubi-real2sim-assets"
POLICY_REV = "1" * 40
ASSET_REV = "2" * 40
UNSET = "UNSET_BEFORE_PUBLICATION"
COMMIT_HASH = re.compile(r"[0-9a-f]{40}")
SCENE = "flat_fixture_room"
CHECKPOINT = "pi05_fixture"
COMMON = ("fr3_collision", "trellis_cup_plate")


class GatedRepoError(Exception):
    """Named like huggingface_hub.errors.GatedRepoError, which the installer recognises."""

    def __init__(self, message, status):
        super().__init__(message)
        self.response = type("Response", (), {"status_code": status})()


class HTTPError(OSError):
    """requests.HTTPError: a RequestException, which is an OSError."""

    def __init__(self, message, status):
        super().__init__(message)
        self.response = type("Response", (), {"status_code": status})()


class HfHubHTTPError(HTTPError):
    """Named and derived like huggingface_hub.errors.HfHubHTTPError."""


class EntryNotFoundError(HfHubHTTPError):
    """Named and derived like huggingface_hub.errors.EntryNotFoundError."""


class LocalEntryNotFoundError(EntryNotFoundError, FileNotFoundError, ValueError):
    """huggingface_hub raises this, from the failed file lookup, when nothing is cached."""

    def __init__(self, cause):
        super().__init__("An error happened while trying to locate the file on the Hub", None)
        self.__cause__ = cause


# Named and derived like requests.exceptions: its ConnectionError is a
# RequestException (an OSError), and SSLError, ProxyError and ConnectTimeout
# derive from that ConnectionError.
RequestsConnectionError = type("ConnectionError", (OSError,), {})
RequestsSSLError = type("SSLError", (RequestsConnectionError,), {})
RequestsProxyError = type("ProxyError", (RequestsConnectionError,), {})
RequestsTimeout = type("Timeout", (OSError,), {})
RequestsConnectTimeout = type("ConnectTimeout", (RequestsConnectionError, RequestsTimeout), {})
# Named like urllib3.exceptions. requests raises its SSLError or ProxyError with
# urllib3's MaxRetryError as the argument; that error keeps its cause in
# ``reason`` and is raised from it.
MaxRetryError = type("MaxRetryError", (Exception,), {})
UrllibSSLError = type("SSLError", (Exception,), {})
UrllibProxyError = type("ProxyError", (Exception,), {})
CertificateError = type("CertificateError", (ValueError,), {})


def _requests_error(kind, reason):
    retry = MaxRetryError(f"Max retries exceeded with url: / (Caused by {reason!r})")
    retry.reason, retry.__cause__ = reason, reason
    return kind(retry)


def _tls(error):
    return _requests_error(RequestsSSLError, UrllibSSLError(error))


def _proxy(error):
    return _requests_error(RequestsProxyError, UrllibProxyError("Unable to connect to proxy", error))


def _raised_while_handling(error, cause):
    error.__context__ = cause
    return error


CERTIFICATE = "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: self-signed certificate"


class FakeHub:
    def __init__(self):
        self.repos: dict[tuple[str, str], dict[str, bytes]] = {}
        self.calls: list[dict] = []
        self.failures: list[Exception] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.failures:
            raise self.failures.pop(0)
        files = self.repos.get((kwargs["repo_id"], kwargs["revision"]))
        if files is None:
            raise GatedRepoError(f"{kwargs['repo_id']}@{kwargs['revision']} not found", 404)
        local = Path(kwargs["local_dir"])
        for path, data in files.items():
            if any(fnmatch.fnmatch(path, pattern) for pattern in kwargs["allow_patterns"]):
                (local / path).parent.mkdir(parents=True, exist_ok=True)
                (local / path).write_bytes(data)
        return str(local)


@pytest.fixture
def hub(monkeypatch):
    fake = FakeHub()
    monkeypatch.setattr(release_assets, "_hub_snapshot", fake)
    monkeypatch.setattr(release_assets.time, "sleep", lambda seconds: None)
    for name in ("REAL2SIM_HF_POLICIES_REVISION", "REAL2SIM_HF_ASSETS_REVISION",
                 "REAL2SIM_INPUTS_ROOT", "HF_ENDPOINT"):
        monkeypatch.delenv(name, raising=False)
    return fake


def _tree(tmp_path: Path, name: str) -> dict[str, bytes]:
    files = {"payload.bin": f"{name} payload\n".encode() * 64, "nested/meta.json": b"{}\n"}
    source = tmp_path / "sources" / name
    for relative, data in files.items():
        (source / relative).parent.mkdir(parents=True, exist_ok=True)
        (source / relative).write_bytes(data)
    return files


@pytest.fixture
def checkout(tmp_path):
    """A checkout with one profile and the four inputs it selects."""
    repo = tmp_path / "checkout"
    destinations = {SCENE: f"{{inputs_root}}/scenes/{SCENE}",
                    CHECKPOINT: f"{{inputs_root}}/checkpoints/{CHECKPOINT}",
                    **{name: f"{{repo_root}}/robosuite_env/assets/{name}" for name in COMMON}}
    folders = {SCENE: f"scenes/{SCENE}", CHECKPOINT: CHECKPOINT,
               **{name: f"assets/{name}" for name in COMMON}}
    archives, served = {}, {POLICIES: {}, ASSETS: {}}
    for name, destination in destinations.items():
        files = _tree(tmp_path, name)
        digest = release_assets.sha256_tree(tmp_path / "sources" / name)
        archives[name] = {
            "destination": destination,
            "tree": {"size": digest.size, "sha256": digest.sha256},
        }
        repo_id = POLICIES if name == CHECKPOINT else ASSETS
        served[repo_id].update({f"{folders[name]}/{path}": data for path, data in files.items()})
    catalog = {"schema_version": 1, "huggingface": {
        "policies": {"repo_id": POLICIES, "repo_type": "model", "revision": UNSET,
                     "folders": {"{inputs_root}/checkpoints/": ""}},
        "assets": {"repo_id": ASSETS, "repo_type": "dataset", "revision": UNSET,
                   "folders": {"{inputs_root}/scenes/": "scenes/",
                               "{repo_root}/robosuite_env/assets/": "assets/"}},
    }, "archives": archives, "runtime_bundles": {
        "yubi_right": [], "yubi_left": [], "fr3": ["fr3_collision"],
        "cup": ["trellis_cup_plate"], "plate": ["trellis_cup_plate"],
    }}
    (repo / "artifacts").mkdir(parents=True)
    (repo / release_assets.RELEASE_CATALOG).write_text(json.dumps(catalog, indent=2))
    profile = repo / "config/release/fr3/flat/fixture_room_cup.yaml"
    profile.parent.mkdir(parents=True)
    profile.write_text(yaml.safe_dump({
        "robot": "fr3", "task": {"objects": "cup_plate"},
        "scenes": [{"path": f"{{inputs_root}}/scenes/{SCENE}", "tag": "franka_flat"}],
        "policy": {"checkpoint": f"{{inputs_root}}/checkpoints/{CHECKPOINT}"},
    }))
    return repo, profile, catalog, served


def _publish(hub, served):
    hub.repos[(POLICIES, POLICY_REV)] = served[POLICIES]
    hub.repos[(ASSETS, ASSET_REV)] = served[ASSETS]


PINNED = ("--hf-revision", f"policies={POLICY_REV}", "--hf-revision", f"assets={ASSET_REV}")


def _run(repo, profile, tmp_path, *extra):
    return release_assets.cli([
        "--profile", str(profile), "--repo-root", str(repo),
        "--inputs-root", str(tmp_path / "inputs"), "--no-env", *extra,
    ])


def _installed(catalog, repo, tmp_path):
    for name, pin in catalog["archives"].items():
        target = Path(pin["destination"].format(inputs_root=tmp_path / "inputs", repo_root=repo))
        assert release_assets.matches_tree(target, pin["tree"]), name


def test_the_catalog_names_both_repositories_and_maps_every_input():
    root = Path(__file__).resolve().parents[2]
    catalog = json.loads((root / release_assets.RELEASE_CATALOG).read_text())
    repos = release_assets.hub_repos(catalog, environ={})
    assert {key: (repo.repo_id, repo.repo_type) for key, repo in repos.items()} == {
        "policies": (POLICIES, "model"), "assets": (ASSETS, "dataset")}
    for repo in repos.values():
        assert repo.revision.startswith(UNSET) or COMMIT_HASH.fullmatch(repo.revision), repo
    for name, pin in catalog["archives"].items():
        assert set(pin) == {"destination", "tree"}, name
    folders = {name: release_assets.locate(pin, repos) for name, pin in catalog["archives"].items()}
    for name, (repo, folder) in folders.items():
        expected = {"policies": name, "assets": f"scenes/{name}"
                    if catalog["archives"][name]["destination"].startswith("{inputs_root}/scenes/")
                    else f"assets/{name}"}[repo.key]
        assert folder == expected, name


def test_a_revision_comes_from_the_option_then_the_environment_then_the_catalog():
    catalog = {"huggingface": {"policies": {"repo_id": POLICIES, "repo_type": "model",
                                            "revision": UNSET, "folders": {}}}}
    assert release_assets.hub_repos(catalog, environ={})["policies"].revision == UNSET
    environ = {"REAL2SIM_HF_POLICIES_REVISION": "a" * 40}
    assert release_assets.hub_repos(catalog, environ=environ)["policies"].revision == "a" * 40
    chosen = release_assets.hub_repos(catalog, revisions={"policies": "b" * 40}, environ=environ)
    assert chosen["policies"].revision == "b" * 40 and not chosen["policies"].problems()
    unpinned = release_assets.hub_repos(catalog, allow_unpinned=True, environ={})["policies"]
    assert unpinned.revision == "main" and not unpinned.problems()
    with pytest.raises(release_assets.ReleaseAssetError, match="names no catalog repository"):
        release_assets.hub_repos(catalog, revisions={"other": "c"}, environ={})


def test_urls_name_the_repository_type_and_revision():
    model = release_assets.HubRepo("policies", POLICIES, "model", "refs/pr/1", ())
    dataset = release_assets.HubRepo("assets", ASSETS, "dataset", ASSET_REV, ())
    assert model.tree_url("pi05_cup") == (
        f"https://huggingface.co/{POLICIES}/tree/refs%2Fpr%2F1/pi05_cup")
    assert dataset.tree_url("scenes/a_room") == (
        f"https://huggingface.co/datasets/{ASSETS}/tree/{ASSET_REV}/scenes/a_room")


@pytest.mark.parametrize("options,message", [
    ((), "has no pinned revision"),
    (("--hf-revision", f"policies={POLICY_REV}"), "REAL2SIM_HF_ASSETS_REVISION"),
    (("--hf-revision", f"assets={ASSET_REV}"), "--allow-unpinned"),
])
def test_an_unpinned_repository_is_refused_before_any_request(
        hub, checkout, tmp_path, capsys, options, message):
    repo, profile, _, served = checkout
    _publish(hub, served)
    assert _run(repo, profile, tmp_path, *options) == 1
    assert message in capsys.readouterr().err
    unpinned = [call for call in hub.calls if call["revision"].startswith(UNSET)]
    assert unpinned == []


def test_dry_run_lists_the_folders_and_flags_an_unpinned_repository(hub, checkout, tmp_path, capsys):
    repo, profile, catalog, _ = checkout
    assert _run(repo, profile, tmp_path, "--dry-run") == 0
    assert capsys.readouterr().out.count("NOT DOWNLOADABLE") == 2
    assert _run(repo, profile, tmp_path, "--dry-run", *PINNED) == 0
    out = capsys.readouterr().out
    assert "NOT DOWNLOADABLE" not in out
    assert f"https://huggingface.co/{POLICIES}/tree/{POLICY_REV}/{CHECKPOINT}; tree sha256 " in out
    assert f"https://huggingface.co/datasets/{ASSETS}/tree/{ASSET_REV}/scenes/{SCENE}; " in out
    assert f"https://huggingface.co/datasets/{ASSETS}/tree/{ASSET_REV}/assets/fr3_collision; " in out
    assert hub.calls == []
    assert not (tmp_path / "inputs").exists()


def test_every_input_is_downloaded_from_its_folder_at_the_pinned_revision(hub, checkout, tmp_path):
    repo, profile, catalog, served = checkout
    _publish(hub, served)
    assert _run(repo, profile, tmp_path, *PINNED) == 0
    requested = sorted((call["repo_id"], call["repo_type"], call["revision"],
                        tuple(call["allow_patterns"])) for call in hub.calls)
    assert requested == sorted(
        [(POLICIES, "model", POLICY_REV, (f"{CHECKPOINT}/*",)),
         (ASSETS, "dataset", ASSET_REV, (f"scenes/{SCENE}/*",))]
        + [(ASSETS, "dataset", ASSET_REV, (f"assets/{name}/*",)) for name in COMMON])
    _installed(catalog, repo, tmp_path)
    downloads = tmp_path / "inputs/release-assets/downloads"
    assert not [path for path in downloads.rglob("*") if path.is_file()]
    receipt = json.loads((tmp_path / "inputs/release-assets/INSTALLED.json").read_text())
    assert receipt["source"] == {"huggingface": {
        "assets": {"repo_id": ASSETS, "revision": ASSET_REV},
        "policies": {"repo_id": POLICIES, "revision": POLICY_REV}}}


def test_the_environment_pins_the_revisions(hub, checkout, tmp_path, monkeypatch):
    repo, profile, catalog, served = checkout
    _publish(hub, served)
    monkeypatch.setenv("REAL2SIM_HF_POLICIES_REVISION", POLICY_REV)
    monkeypatch.setenv("REAL2SIM_HF_ASSETS_REVISION", ASSET_REV)
    assert _run(repo, profile, tmp_path) == 0
    _installed(catalog, repo, tmp_path)


@pytest.mark.parametrize("damage", ["changed", "missing", "extra"])
def test_a_download_that_differs_from_its_tree_pin_is_never_installed(
        hub, checkout, tmp_path, capsys, damage):
    repo, profile, _, served = checkout
    files = dict(served[POLICIES])
    if damage == "changed":
        files[f"{CHECKPOINT}/payload.bin"] = b"other weights\n"
    elif damage == "missing":
        del files[f"{CHECKPOINT}/nested/meta.json"]
    else:
        files[f"{CHECKPOINT}/unpinned.txt"] = b"x"
    _publish(hub, {POLICIES: files, ASSETS: served[ASSETS]})
    assert _run(repo, profile, tmp_path, *PINNED) == 1
    error = capsys.readouterr().err
    assert f"{CHECKPOINT}: the files downloaded from" in error and "nothing was installed" in error
    assert not (tmp_path / "inputs/checkpoints" / CHECKPOINT).exists()
    assert not (tmp_path / "inputs/release-assets/downloads/policies" / CHECKPOINT).exists()


@pytest.mark.parametrize("status", [401, 403])
def test_a_gated_repository_tells_the_user_to_log_in_and_accept_the_terms(
        hub, checkout, tmp_path, capsys, status):
    repo, profile, _, served = checkout
    _publish(hub, served)
    hub.failures.append(GatedRepoError("Access to model is restricted", status))
    assert _run(repo, profile, tmp_path, *PINNED) == 1
    error = capsys.readouterr().err
    assert ".venv/bin/hf auth login" in error and "accept its terms" in error
    assert "https://huggingface.co/datasets/" + ASSETS in error or "https://huggingface.co/" + POLICIES in error
    assert len(hub.calls) == 1


def test_a_transient_failure_is_retried_and_resumes(hub, checkout, tmp_path):
    repo, profile, catalog, served = checkout
    _publish(hub, served)
    hub.failures.extend([ConnectionError("reset"), TimeoutError("slow")])
    assert _run(repo, profile, tmp_path, *PINNED) == 0
    assert len(hub.calls) == 4 + 2
    _installed(catalog, repo, tmp_path)


@pytest.mark.parametrize("failure", [
    lambda: HfHubHTTPError("Request Timeout", 408),
    lambda: HfHubHTTPError("Too Many Requests", 429),
    lambda: HfHubHTTPError("Service Unavailable", 503),
    lambda: LocalEntryNotFoundError(HfHubHTTPError("Bad Gateway", 502)),
    lambda: LocalEntryNotFoundError(ConnectionError("reset")),
    lambda: RequestsConnectionError("Connection refused"),
    lambda: RequestsConnectTimeout("connect timeout"),
    lambda: _tls(ssl.SSLEOFError(8, "[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred")),
    lambda: _tls(ssl.SSLError(1, "[SSL: WRONG_VERSION_NUMBER] wrong version number")),
    lambda: _proxy(OSError("Tunnel connection failed: 503 Service Unavailable")),
    lambda: _proxy(ConnectionRefusedError(111, "Connection refused")),
    lambda: LocalEntryNotFoundError(_tls(ssl.SSLEOFError(8, "EOF occurred"))),
], ids=["408", "429", "503", "lookup-502", "lookup-connection", "requests-connection",
        "requests-connect-timeout", "tls-handshake-cut", "tls-not-spoken", "proxy-503",
        "proxy-refused", "lookup-tls-handshake-cut"])
def test_a_retryable_http_status_is_retried(hub, checkout, tmp_path, monkeypatch, failure):
    repo, profile, catalog, served = checkout
    _publish(hub, served)
    sleeps = []
    monkeypatch.setattr(release_assets.time, "sleep", sleeps.append)
    hub.failures.append(failure())
    assert _run(repo, profile, tmp_path, *PINNED) == 0
    assert len(hub.calls) == 4 + 1 and sleeps == [1]
    _installed(catalog, repo, tmp_path)


@pytest.mark.parametrize("failure,status", [
    (lambda: HfHubHTTPError("Bad Request", 400), 400),
    (lambda: EntryNotFoundError("Entry Not Found", 404), 404),
    (lambda: HfHubHTTPError("Conflict", 409), 409),
    (lambda: HfHubHTTPError("Not Implemented", 501), 501),
    (lambda: LocalEntryNotFoundError(HfHubHTTPError("Bad Request", 400)), 400),
], ids=["400", "entry-404", "409", "501", "lookup-400"])
def test_a_non_retryable_http_status_fails_at_once_and_names_the_status(
        hub, checkout, tmp_path, capsys, monkeypatch, failure, status):
    repo, profile, _, served = checkout
    _publish(hub, served)
    sleeps = []
    monkeypatch.setattr(release_assets.time, "sleep", sleeps.append)
    hub.failures.append(failure())
    assert _run(repo, profile, tmp_path, *PINNED) == 1
    error = capsys.readouterr().err
    assert f"returned HTTP {status}: " in error and "attempts" not in error
    assert len(hub.calls) == 1 and sleeps == []


@pytest.mark.parametrize("failure,reason", [
    (lambda: _tls(ssl.SSLCertVerificationError(1, CERTIFICATE)),
     "its TLS certificate could not be verified"),
    (lambda: _requests_error(RequestsSSLError, CertificateError("hostname mismatch")),
     "its TLS certificate could not be verified"),
    (lambda: _raised_while_handling(RequestsSSLError("handshake failed"),
                                    ssl.SSLCertVerificationError(1, CERTIFICATE)),
     "its TLS certificate could not be verified"),
    (lambda: LocalEntryNotFoundError(_tls(ssl.SSLCertVerificationError(1, CERTIFICATE))),
     "its TLS certificate could not be verified"),
    (lambda: _proxy(OSError("Tunnel connection failed: 407 Proxy Authentication Required")),
     "the proxy requires authentication (HTTP 407)"),
    (lambda: HfHubHTTPError("Proxy Authentication Required", 407),
     "the proxy requires authentication (HTTP 407)"),
], ids=["certificate", "host-name", "certificate-in-context", "lookup-certificate",
        "proxy-tunnel-407", "proxy-407"])
def test_a_tls_or_proxy_setup_failure_fails_at_once_and_names_the_setup(
        hub, checkout, tmp_path, capsys, monkeypatch, failure, reason):
    repo, profile, _, served = checkout
    _publish(hub, served)
    sleeps = []
    monkeypatch.setattr(release_assets.time, "sleep", sleeps.append)
    hub.failures.append(failure())
    assert _run(repo, profile, tmp_path, *PINNED) == 1
    error = capsys.readouterr().err
    assert f"could not be reached because {reason}" in error and "not retried" in error
    assert "HTTPS_PROXY" in error and "REQUESTS_CA_BUNDLE" in error and "system clock" in error
    assert "attempts" not in error
    assert len(hub.calls) == 1 and sleeps == []


def test_a_local_disk_error_is_not_retried(hub, checkout, tmp_path, capsys, monkeypatch):
    repo, profile, _, served = checkout
    _publish(hub, served)
    sleeps = []
    monkeypatch.setattr(release_assets.time, "sleep", sleeps.append)
    hub.failures.append(OSError(28, "No space left on device"))
    assert _run(repo, profile, tmp_path, *PINNED) == 1
    assert "No space left on device" in capsys.readouterr().err
    assert len(hub.calls) == 1 and sleeps == []


def test_a_persistent_transient_failure_stops_after_the_last_attempt(
        hub, checkout, tmp_path, capsys, monkeypatch):
    repo, profile, _, served = checkout
    _publish(hub, served)
    sleeps = []
    monkeypatch.setattr(release_assets.time, "sleep", sleeps.append)
    hub.failures.extend(HfHubHTTPError("Service Unavailable", 503)
                        for _ in range(release_assets.MAX_ATTEMPTS))
    assert _run(repo, profile, tmp_path, *PINNED) == 1
    captured = capsys.readouterr()
    error = captured.err
    assert f"failed after {release_assets.MAX_ATTEMPTS} attempts" in error
    assert len(hub.calls) == release_assets.MAX_ATTEMPTS
    # A wait follows every failed attempt except the last one.
    assert sleeps == [min(2**attempt, 30) for attempt in range(release_assets.MAX_ATTEMPTS - 1)]
    assert captured.out.count("retrying in") == release_assets.MAX_ATTEMPTS - 1
    assert "TLS or proxy" not in error


@pytest.mark.parametrize("failure", [
    lambda: _tls(ssl.SSLEOFError(8, "[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred")),
    lambda: _proxy(OSError("Tunnel connection failed: 503 Service Unavailable")),
    lambda: LocalEntryNotFoundError(_proxy(ConnectionRefusedError(111, "Connection refused"))),
], ids=["tls-handshake-cut", "proxy-503", "lookup-proxy-refused"])
def test_a_persistent_tls_or_proxy_failure_names_the_setup_after_the_last_attempt(
        hub, checkout, tmp_path, capsys, monkeypatch, failure):
    repo, profile, _, served = checkout
    _publish(hub, served)
    sleeps = []
    monkeypatch.setattr(release_assets.time, "sleep", sleeps.append)
    hub.failures.extend(failure() for _ in range(release_assets.MAX_ATTEMPTS))
    assert _run(repo, profile, tmp_path, *PINNED) == 1
    error = capsys.readouterr().err
    assert f"failed after {release_assets.MAX_ATTEMPTS} attempts" in error
    assert "The last attempt failed with a TLS or proxy error" in error
    assert "HTTPS_PROXY" in error and "REQUESTS_CA_BUNDLE" in error and "system clock" in error
    assert len(hub.calls) == release_assets.MAX_ATTEMPTS
    assert sleeps == [min(2**attempt, 30) for attempt in range(release_assets.MAX_ATTEMPTS - 1)]


def test_a_changed_installation_names_the_recovery_that_works(hub, checkout, tmp_path, capsys):
    repo, profile, catalog, served = checkout
    _publish(hub, served)
    assert _run(repo, profile, tmp_path, *PINNED) == 0
    target = tmp_path / "inputs/checkpoints" / CHECKPOINT
    (target / "extra.txt").write_text("left over from an earlier release\n")
    capsys.readouterr()
    assert _run(repo, profile, tmp_path, *PINNED) == 1
    error = capsys.readouterr().err
    assert f"remove {target} and rerun the same profile" in error and "--force" not in error
    shutil.rmtree(target)
    assert _run(repo, profile, tmp_path, *PINNED) == 0
    _installed(catalog, repo, tmp_path)
    assert not (target / "extra.txt").exists()


def test_an_install_without_no_env_writes_the_inputs_root_into_dot_env(hub, checkout, tmp_path):
    repo, profile, catalog, served = checkout
    (repo / ".env.example").write_text("# template\nREAL2SIM_INPUTS_ROOT=\n")
    _publish(hub, served)
    assert release_assets.cli([
        "--profile", str(profile), "--repo-root", str(repo),
        "--inputs-root", str(tmp_path / "inputs"), *PINNED,
    ]) == 0
    _installed(catalog, repo, tmp_path)
    assert (repo / ".env").read_text() == f"# template\nREAL2SIM_INPUTS_ROOT={tmp_path / 'inputs'}\n"
