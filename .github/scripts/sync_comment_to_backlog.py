#!/usr/bin/env python3
"""GitHub Issue コメント → Backlog 課題コメント 片方向同期."""
from __future__ import annotations

import json
import os
import re
import sys

import requests

BACKLOG_SPACE = os.environ["BACKLOG_SPACE"]
BACKLOG_API_KEY = os.environ["BACKLOG_API_KEY"]
GITHUB_EVENT_PATH = os.environ["GITHUB_EVENT_PATH"]

BACKLOG_BASE = f"https://{BACKLOG_SPACE}/api/v2"
MARKER_RE = re.compile(r"<!--\s*backlog-key:\s*([A-Z0-9_]+-\d+)\s*-->")
# Backlog→GitHub 同期で付くマーカー(無限ループ防止用)
ECHO_RE = re.compile(r"<!--\s*from-backlog\s*-->")


def main() -> int:
    with open(GITHUB_EVENT_PATH) as f:
        event = json.load(f)

    issue = event["issue"]
    comment = event["comment"]

    if ECHO_RE.search(comment.get("body") or ""):
        print("Comment originated from Backlog; skip to avoid loop.")
        return 0

    m = MARKER_RE.search(issue.get("body") or "")
    if not m:
        print("No backlog-key marker on this issue; skip.")
        return 0
    backlog_key = m.group(1)

    author = comment["user"]["login"]
    url = comment["html_url"]
    body = comment.get("body") or ""
    content = (
        f"**{author}** さんが GitHub にコメント\n"
        f"{url}\n\n"
        f"---\n"
        f"{body}\n\n"
        f"<!-- from-github -->"
    )

    r = requests.post(
        f"{BACKLOG_BASE}/issues/{backlog_key}/comments",
        params={"apiKey": BACKLOG_API_KEY},
        data={"content": content},
        timeout=30,
    )
    r.raise_for_status()
    print(f"Posted comment to Backlog {backlog_key}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
