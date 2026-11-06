"""Copy files or folders from the Jupyter server's ~/work/cis634 into the local repo.

usage: jfetch.py LOCAL_REPO PATH [PATH ...]   (paths relative to the repo; folders recurse,
skipping `runs/` subfolders, which are git-ignored)
"""
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import truststore

truststore.inject_into_ssl()
BASE = "https://cis-neuron.mnsu.edu/user/15835731/api/contents/cis634/"
H = {"Authorization": f"token {os.environ['JUPYTERHUB_TOKEN']}"}


def get(path: str, content: bool = True) -> dict:
    url = BASE + urllib.parse.quote(path) + ("" if content else "?content=0")
    with urllib.request.urlopen(urllib.request.Request(url, headers=H), timeout=120) as r:
        return json.loads(r.read())


def fetch(repo: Path, path: str) -> int:
    meta = get(path)
    if meta["type"] == "directory":
        return sum(fetch(repo, c["path"].split("cis634/", 1)[1]) for c in meta["content"]
                   if c["name"] != "runs")
    dest = repo / path
    dest.parent.mkdir(parents=True, exist_ok=True)
    if meta["format"] == "base64":
        import base64
        dest.write_bytes(base64.b64decode(meta["content"]))
    else:
        dest.write_text(meta["content"], encoding="utf-8")
    return 1


repo = Path(sys.argv[1])
for p in sys.argv[2:]:
    print(p, fetch(repo, p.rstrip("/")), "files")
