"""OSF client against a fake API (no network)."""

import hashlib

import pytest
import requests

from cyclingml.io import osf

API = osf.API
ZIP_BYTES = b"PK fake zip content"
SHA = hashlib.sha256(ZIP_BYTES).hexdigest()


def _file(name, path, size=None, sha=SHA):
    size = len(ZIP_BYTES) if size is None else size
    return {
        "attributes": {"kind": "file", "name": name, "materialized_path": path, "size": size,
                       "extra": {"hashes": {"sha256": sha, "md5": ""}},
                       "date_modified": "2020-01-01"},
        "links": {"download": f"https://files.example/{name}"},
    }  # fmt: skip


def _folder(name, href):
    return {
        "attributes": {"kind": "folder", "name": name},
        "relationships": {"files": {"links": {"related": {"href": href}}}},
    }


class FakeResponse:
    def __init__(self, status=200, json_body=None, content=b"", headers=None):
        self.status_code = status
        self._json = json_body
        self._content = content
        self.headers = headers or {}

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def iter_content(self, n):
        yield self._content

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeSession:
    """Routes by URL prefix; the root listing is paginated and the first call is rate limited."""

    def __init__(self):
        self.headers = {}
        self.calls = []
        self.rate_limited_once = False

    def get(self, url, stream=False, timeout=None, **kw):
        self.calls.append(url)
        root = f"{API}/nodes/6hfpz/files/osfstorage/"
        if url.startswith(f"{API}/nodes/6hfpz/files/osfstorage/data/"):
            return FakeResponse(json_body={"data": [_file("a.zip", "/data/a.zip"),
                                                    _file("b.zip", "/data/b.zip", sha="bad")],
                                           "links": {}})  # fmt: skip
        if url.startswith(root) and not self.rate_limited_once:
            self.rate_limited_once = True
            return FakeResponse(429, headers={"Retry-After": "0"})
        if url.startswith(root) and "page=2" not in url:
            return FakeResponse(json_body={
                "data": [_folder("data", f"{API}/nodes/6hfpz/files/osfstorage/data/")],
                "links": {"next": f"{root}?page=2"}})  # fmt: skip
        if url.startswith(root) and "page=2" in url:
            return FakeResponse(json_body={"data": [_file("README.txt", "/README.txt", 5, "")],
                                           "links": {"next": None}})  # fmt: skip
        if url.startswith(f"{API}/nodes/6hfpz/children/"):
            return FakeResponse(json_body={"data": [], "links": {}})
        if url.startswith("https://files.example/"):
            return FakeResponse(content=ZIP_BYTES)
        return FakeResponse(404)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(osf.time, "sleep", lambda s: None)


def test_list_files_walks_folders_pages_and_retries():
    s = FakeSession()
    files = list(osf.list_files("6hfpz", session=s))
    assert sorted(f.path for f in files) == ["/README.txt", "/data/a.zip", "/data/b.zip"]
    assert s.rate_limited_once
    first_pages = [u for u in s.calls if "osfstorage" in u and "page=2" not in u]
    assert all("page[size]=100" in u for u in first_pages)


def test_manifest_roundtrip_and_summary(tmp_path):
    files = list(osf.list_files("6hfpz", session=FakeSession()))
    m = tmp_path / "manifest.csv"
    assert osf.write_manifest(iter(files), m) == 3
    back = osf.read_manifest(m)
    assert back == files
    text = osf.summarize(back)
    assert "3 files" in text and ".zip" in text and "README.txt" in text


def test_download_verifies_checksums_and_resumes(tmp_path):
    files = list(osf.list_files("6hfpz", session=FakeSession()))
    dest = tmp_path / "zips"
    counts = osf.download(files, dest, workers=2, session=FakeSession())
    assert counts == {"downloaded": 1, "skipped": 0, "failed": 1}  # b.zip has a bad checksum
    assert (dest / "a.zip").read_bytes() == ZIP_BYTES
    assert not (dest / "b.zip").exists() and not list(dest.glob("*.part"))
    assert "b.zip" in (dest / "failed.txt").read_text()
    again = osf.download(files, dest, workers=2, session=FakeSession(), limit=1)
    assert again == {"downloaded": 0, "skipped": 1, "failed": 0}


def test_token_header(monkeypatch):
    monkeypatch.setenv("OSF_TOKEN", "abc")
    assert osf.make_session().headers["Authorization"] == "Bearer abc"


def test_rate_limit_pauses_all_threads_and_eventually_succeeds():
    t = osf.Throttle(min_interval=0.0)
    calls = {"n": 0}

    class Flaky:
        headers = {}

        def get(self, url, **kw):
            calls["n"] += 1
            return FakeResponse(429) if calls["n"] <= 8 else FakeResponse(json_body={"ok": 1})

    r = osf._get(Flaky(), "https://x", throttle=t)  # 8 rate limits > old limit of 6
    assert r.json() == {"ok": 1}
    assert t._cooldown_until > 0  # the pause was shared


def test_rate_limit_gives_up_after_limit():
    class Always429:
        headers = {}

        def get(self, url, **kw):
            return FakeResponse(429)

    with pytest.raises(requests.HTTPError):
        osf._get(Always429(), "https://x", throttle=osf.Throttle(0.0), rate_limit_retries=3)


def test_throttle_spaces_requests(monkeypatch):
    slept = []
    monkeypatch.setattr(osf.time, "sleep", lambda s: slept.append(s))
    t = osf.Throttle(min_interval=1.0)
    for _ in range(3):
        t.wait()
    assert slept[-1] > 1.5  # third request waits about two intervals
