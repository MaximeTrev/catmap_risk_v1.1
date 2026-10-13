import json
from catmap.config import Settings


class FakeResponse:
    def __init__(self, status=200, payload=None, text="", headers=None):
        self.status_code = status
        self._payload = payload
        self.text = text or (json.dumps(payload) if payload is not None else "")
        self.headers = headers or {}

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


class FakeSession:
    """Rejoue une liste de réponses ou d'exceptions, mémorise les appels."""
    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def post(self, url, data=None, headers=None, timeout=None):
        self.calls.append({"url": url, "data": data, "headers": headers, "timeout": timeout})
        item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(item, Exception):
            raise item
        return item


def test_settings(tmp_path, **kw):
    base = dict(overpass_urls=("http://a/api", "http://b/api"), cache_dir=tmp_path, min_interval_s=0.0,
                backoff_base_s=0.0, backoff_max_s=0.0, max_attempts=2, contact="test@example.org")
    base.update(kw)
    return Settings(**base)


def way(id_, name, pts, **tags):
    return {"type": "way", "id": id_, "tags": {"name": name, **tags},
            "geometry": [{"lat": la, "lon": lo} for lo, la in pts]}


def square(lon, lat, d=0.001):
    return [(lon, lat), (lon + d, lat), (lon + d, lat + d), (lon, lat + d), (lon, lat)]


def node(id_, name, lon, lat, **tags):
    return {"type": "node", "id": id_, "lat": lat, "lon": lon, "tags": {"name": name, **tags}}


def payload(elements, remark=None, ts="2026-09-30T20:00:00Z"):
    p = {"version": 0.6, "osm3s": {"timestamp_osm_base": ts}, "elements": elements}
    if remark:
        p["remark"] = remark
    return p
