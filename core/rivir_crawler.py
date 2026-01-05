#!/usr/bin/env python3
"""
RIVIR web crawler (explicitly instructed, allowlist only).
"""

import argparse
import json
import time
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable, List, Set, Tuple
from urllib.parse import urljoin, urlparse
from urllib.request import urlopen
from urllib.robotparser import RobotFileParser

try:
    from .consent_gate import ConsentGate
except Exception:
    from consent_gate import ConsentGate


DEFAULT_ALLOWLIST = {
    "wikipedia.org",
    "stackoverflow.com",
    "arxiv.org",
    "docs.python.org",
    "developer.mozilla.org",
}


class LinkExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links: List[str] = []

    def handle_starttag(self, tag, attrs):
        if tag != "a":
            return
        for k, v in attrs:
            if k == "href" and v:
                self.links.append(v)


def _domain(url: str) -> str:
    return urlparse(url).netloc.lower()


def _allowed(url: str, allowlist: Set[str]) -> bool:
    host = _domain(url)
    return any(host == d or host.endswith("." + d) for d in allowlist)


def _safe_filename(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.strip("/").replace("/", "_")
    if not path:
        path = "index"
    return f"{parsed.netloc}_{path}.html"


@dataclass
class CrawlConfig:
    allowlist: Set[str] = field(default_factory=lambda: set(DEFAULT_ALLOWLIST))
    max_pages: int = 50
    max_depth: int = 2
    delay_seconds: float = 0.5
    respect_robots: bool = True


class Crawler:
    def __init__(self, config: CrawlConfig, out_dir: Path, log_path: Path):
        self.config = config
        self.out_dir = out_dir
        self.log_path = log_path
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._robots = {}

    def _log(self, event: dict):
        event = dict(event)
        event["ts"] = time.time()
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")

    def _robot_allows(self, url: str) -> bool:
        if not self.config.respect_robots:
            return True
        host = _domain(url)
        if host not in self._robots:
            rp = RobotFileParser()
            rp.set_url(urljoin(f"https://{host}", "/robots.txt"))
            try:
                rp.read()
            except Exception:
                self._robots[host] = None
                return True
            self._robots[host] = rp
        rp = self._robots.get(host)
        if rp is None:
            return True
        return rp.can_fetch("*", url)

    def crawl(self, seeds: Iterable[str]):
        queue: List[Tuple[str, int]] = [(s, 0) for s in seeds]
        seen: Set[str] = set()

        while queue and len(seen) < self.config.max_pages:
            url, depth = queue.pop(0)
            if url in seen or depth > self.config.max_depth:
                continue
            if not _allowed(url, self.config.allowlist):
                continue
            if not self._robot_allows(url):
                self._log({"event": "blocked_robots", "url": url})
                continue

            try:
                with urlopen(url, timeout=10) as resp:
                    content_type = resp.headers.get("Content-Type", "")
                    if "text/html" not in content_type:
                        self._log({"event": "skip_non_html", "url": url, "content_type": content_type})
                        continue
                    html = resp.read().decode("utf-8", errors="ignore")
            except Exception as e:
                self._log({"event": "error", "url": url, "error": str(e)})
                continue

            seen.add(url)
            out_file = self.out_dir / _safe_filename(url)
            out_file.write_text(html, encoding="utf-8")
            self._log({"event": "fetched", "url": url, "depth": depth, "file": str(out_file)})

            extractor = LinkExtractor()
            extractor.feed(html)
            for link in extractor.links:
                next_url = urljoin(url, link)
                if _allowed(next_url, self.config.allowlist):
                    queue.append((next_url, depth + 1))

            time.sleep(self.config.delay_seconds)


def main():
    parser = argparse.ArgumentParser(description="RIVIR allowlisted web crawler")
    parser.add_argument("--seed", action="append", required=True, help="Seed URL (repeatable)")
    parser.add_argument("--out", default="rivir_crawl", help="Output directory")
    parser.add_argument("--log", default="rivir_crawl_log.jsonl", help="Log file path")
    parser.add_argument("--max-pages", type=int, default=50, help="Max pages to fetch")
    parser.add_argument("--depth", type=int, default=2, help="Max crawl depth")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between requests (seconds)")
    parser.add_argument("--no-robots", action="store_true", help="Ignore robots.txt")
    parser.add_argument("--allow", action="append", help="Allowlist domain (repeatable)")
    parser.add_argument("--confirm", default="", help="Affirmative signal to run (e.g., yes)")
    args = parser.parse_args()

    gate = ConsentGate()
    if not gate.request("crawl", args.confirm):
        print("Consent not granted. Provide --confirm with an affirmative signal.")
        return

    allowlist = set(DEFAULT_ALLOWLIST)
    if args.allow:
        allowlist |= set(args.allow)

    config = CrawlConfig(
        allowlist=allowlist,
        max_pages=args.max_pages,
        max_depth=args.depth,
        delay_seconds=args.delay,
        respect_robots=not args.no_robots,
    )

    crawler = Crawler(config, Path(args.out), Path(args.log))
    crawler.crawl(args.seed)


if __name__ == "__main__":
    main()
