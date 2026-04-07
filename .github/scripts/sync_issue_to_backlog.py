#!/usr/bin/env python3
"""GitHub Issue → Backlog 課題 片方向同期スクリプト.

設計方針:
- 正本は GitHub Issue
- Backlog 課題はミラー(閲覧用)
- ID マッピングは GitHub Issue 本文末尾のマーカーに保存(DB レス)
    <!-- backlog-key: PRJ_2-15 -->
- bot 起因の編集は workflow 側でスキップしているのでループしない
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import sys

import requests
import yaml

BACKLOG_SPACE = os.environ["BACKLOG_SPACE"]
BACKLOG_API_KEY = os.environ["BACKLOG_API_KEY"]
BACKLOG_PROJECT_KEY = os.environ["BACKLOG_PROJECT_KEY"]
GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
GITHUB_REPO = os.environ["GITHUB_REPOSITORY"]
GITHUB_EVENT_PATH = os.environ["GITHUB_EVENT_PATH"]

BACKLOG_BASE = f"https://{BACKLOG_SPACE}/api/v2"
GITHUB_BASE = "https://api.github.com"

MARKER_RE = re.compile(r"<!--\s*backlog-key:\s*([A-Z0-9_]+-\d+)\s*-->")

MAPPING_PATH = pathlib.Path(".github/sync-mapping.yml")


def load_mapping() -> dict:
    if not MAPPING_PATH.exists():
        return {}
    with MAPPING_PATH.open() as f:
        return yaml.safe_load(f) or {}


# ---------- Backlog API ----------
def bl_get(path, **params):
    params["apiKey"] = BACKLOG_API_KEY
    r = requests.get(f"{BACKLOG_BASE}{path}", params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def bl_post(path, data):
    r = requests.post(
        f"{BACKLOG_BASE}{path}",
        params={"apiKey": BACKLOG_API_KEY},
        data=data,
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def bl_patch(path, data):
    r = requests.patch(
        f"{BACKLOG_BASE}{path}",
        params={"apiKey": BACKLOG_API_KEY},
        data=data,
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


# ---------- GitHub API ----------
def gh_patch_issue_body(number: int, body: str) -> None:
    r = requests.patch(
        f"{GITHUB_BASE}/repos/{GITHUB_REPO}/issues/{number}",
        headers={
            "Authorization": f"Bearer {GITHUB_TOKEN}",
            "Accept": "application/vnd.github+json",
        },
        json={"body": body},
        timeout=30,
    )
    r.raise_for_status()


# ---------- Helpers ----------
def find_marker(body: str | None) -> str | None:
    if not body:
        return None
    m = MARKER_RE.search(body)
    return m.group(1) if m else None


def append_marker(body: str | None, key: str) -> str:
    base = (body or "").rstrip()
    url = f"https://{BACKLOG_SPACE}/view/{key}"
    return f"{base}\n\n---\n<!-- backlog-key: {key} -->\n<!-- backlog-url: {url} -->\n"


def build_summary(issue: dict) -> str:
    return f"[GH#{issue['number']}] {issue['title']}"


def build_description(issue: dict) -> str:
    body = MARKER_RE.sub("", issue.get("body") or "").rstrip()
    repo = GITHUB_REPO
    num = issue["number"]
    return (
        f"{body}\n\n---\n"
        f"GitHub Issue: {issue['html_url']}\n"
        f"<!-- github-issue: {repo}#{num} -->"
    )


def pick(items, key, names, fallback_index=0):
    for n in names:
        for it in items:
            if it[key] == n:
                return it["id"]
    return items[fallback_index]["id"]


# ---------- Main ----------
def main() -> int:
    with open(GITHUB_EVENT_PATH) as f:
        event = json.load(f)

    action = event.get("action")
    issue = event.get("issue")
    if not issue:
        print(f"No issue payload (action={action}); skip.")
        return 0

    mapping = load_mapping()

    project = bl_get(f"/projects/{BACKLOG_PROJECT_KEY}")
    issue_types = bl_get(f"/projects/{BACKLOG_PROJECT_KEY}/issueTypes")
    statuses = bl_get(f"/projects/{BACKLOG_PROJECT_KEY}/statuses")
    priorities = bl_get("/priorities")

    project_id = project["id"]
    priority_id = pick(priorities, "name", ["中", "Normal"])
    open_status_id = pick(statuses, "name", ["未対応", "Open"], 0)
    closed_status_id = pick(statuses, "name", ["完了", "Closed"], -1)

    type_id = resolve_issue_type_id(issue, issue_types, mapping)
    assignee_id = resolve_assignee_id(issue, mapping)

    backlog_key = find_marker(issue.get("body"))

    if backlog_key is None:
        data = {
            "projectId": project_id,
            "summary": build_summary(issue),
            "issueTypeId": type_id,
            "priorityId": priority_id,
            "description": build_description(issue),
        }
        if assignee_id is not None:
            data["assigneeId"] = assignee_id
        created = bl_post("/issues", data)
        backlog_key = created["issueKey"]
        gh_patch_issue_body(issue["number"], append_marker(issue.get("body"), backlog_key))
        print(f"Created Backlog {backlog_key} from GH#{issue['number']}")
        return 0

    data = {
        "summary": build_summary(issue),
        "description": build_description(issue),
        "statusId": closed_status_id if issue["state"] == "closed" else open_status_id,
        "issueTypeId": type_id,
    }
    # Backlog API: 担当者解除は assigneeId="" の空文字
    data["assigneeId"] = assignee_id if assignee_id is not None else ""
    bl_patch(f"/issues/{backlog_key}", data)
    print(f"Updated Backlog {backlog_key} from GH#{issue['number']} (action={action})")
    return 0


def resolve_issue_type_id(issue: dict, issue_types: list, mapping: dict) -> int:
    name_to_id = {t["name"]: t["id"] for t in issue_types}
    label_map = mapping.get("labels_to_issue_type") or {}
    for label in issue.get("labels", []):
        target = label_map.get(label["name"])
        if target and target in name_to_id:
            return name_to_id[target]
    default_name = mapping.get("default_issue_type")
    if default_name and default_name in name_to_id:
        return name_to_id[default_name]
    return issue_types[0]["id"]


def resolve_assignee_id(issue: dict, mapping: dict):
    user_map = mapping.get("users") or {}
    assignees = issue.get("assignees") or []
    for a in assignees:
        if a["login"] in user_map:
            return user_map[a["login"]]
    return mapping.get("default_assignee")


if __name__ == "__main__":
    sys.exit(main())
