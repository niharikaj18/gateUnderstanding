import httpx

from gitcontext.ingest.github import GitHubClient


def handler(req: httpx.Request) -> httpx.Response:
    u = req.url.path
    if u.endswith("/pulls"):
        return httpx.Response(200, json=[{"number": 7, "title": "Fix retry bug", "body": "Retries were infinite.",
                                          "state": "closed", "user": {"login": "bob"}, "labels": [],
                                          "updated_at": "2024-05-01T00:00:00Z", "html_url": "https://x/pull/7"}])
    if u.endswith("/issues/7/comments"):
        return httpx.Response(200, json=[{"user": {"login": "carol"}, "body": "LGTM, ship it", "created_at": "2024-05-01T01:00:00Z"}])
    if u.endswith("/pulls/7/comments") or u.endswith("/pulls/7/reviews"):
        return httpx.Response(200, json=[])
    if u.endswith("/issues"):
        return httpx.Response(200, json=[
            {"number": 8, "title": "Outage on 2024-04-30", "body": "DB failover", "state": "closed", "user": {"login": "d"},
             "labels": [{"name": "postmortem"}], "updated_at": "2024-05-02T00:00:00Z", "html_url": "https://x/8", "comments": 0},
            {"number": 7, "title": "pr", "pull_request": {}, "labels": []},
            {"number": 9, "title": "Typo", "body": "x", "state": "open", "user": {"login": "e"}, "labels": [],
             "updated_at": "2024-05-02T00:00:00Z", "html_url": "https://x/9", "comments": 0}])
    return httpx.Response(404)


def test_pr_issue_incident_ingestion():
    gh = GitHubClient("o/r", transport=httpx.MockTransport(handler))
    chunks = gh.fetch(None, 10, 10, ["postmortem"])
    by = {(c.source_type, c.pr_number): c for c in chunks}
    assert set(by) == {("pr", 7), ("incident", 8), ("issue", 9)}  # PR not double-counted via /issues
    assert "LGTM" in by[("pr", 7)].text and by[("pr", 7)].author == "bob"
    assert by[("pr", 7)].url == "https://x/pull/7" and by[("incident", 8)].date.startswith("2024-05-02")


def test_rate_limit_degrades_gracefully():
    gh = GitHubClient("o/r", transport=httpx.MockTransport(
        lambda r: httpx.Response(403, headers={"x-ratelimit-remaining": "0"})))
    assert gh.fetch(None, 10, 10, []) == []


def test_rate_limit_is_reported_not_silent():
    gh = GitHubClient("o/r", transport=httpx.MockTransport(
        lambda r: httpx.Response(403, headers={"x-ratelimit-remaining": "0"})))
    gh.fetch(None, 10, 10, [])
    assert any("rate limit" in w for w in gh.warnings)
