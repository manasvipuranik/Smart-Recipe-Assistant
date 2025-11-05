#!/usr/bin/env python3
"""
Indian Recipe Scraper — 15-site starter

Features
- Starts with 15 Indian recipe websites (easily extendable)
- Respects robots.txt (won't crawl disallowed paths)
- Pulls recipe data primarily from schema.org Recipe JSON-LD
- Follows sitemaps recursively (with depth and URL caps)
- De-duplicates by URL
- Saves consolidated results to both JSON Lines (.jsonl) and CSV

Usage
    python indian_recipe_scraper.py --out-prefix recipes --per-site-max 2000 --workers 8

Dependencies (install once)
    pip install requests beautifulsoup4 lxml pandas tqdm

Notes
- ALWAYS check each site's Terms of Service before crawling, and be respectful with rate limits.
- For heavy crawls, reduce workers, increase delay, or set smaller --per-site-max.
- You can add/remove sites in the SITES list at the bottom.
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import contextlib
import csv
import dataclasses
import io
import json
import random
import re
import sys
import time
from collections import deque
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from urllib import robotparser
from lxml import etree
from tqdm import tqdm

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; RecipeScraper/1.0; +https://example.com/contact)"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

SESSION = requests.Session()
SESSION.headers.update(DEFAULT_HEADERS)
SESSION_TIMEOUT = 20


def backoff_sleep(attempt: int, base: float = 0.5, cap: float = 10.0) -> None:
    delay = min(cap, base * (2 ** attempt) + random.uniform(0, 0.25))
    time.sleep(delay)


def http_get(url: str) -> Optional[requests.Response]:
    for attempt in range(4):
        try:
            resp = SESSION.get(url, timeout=SESSION_TIMEOUT, allow_redirects=True)
            if resp.status_code == 200:
                return resp
            elif 300 <= resp.status_code < 400:
                # follow handled by allow_redirects
                pass
            elif resp.status_code in (403, 404, 410):
                return None
        except requests.RequestException:
            pass
        backoff_sleep(attempt)
    return None



@dataclass
class SiteConfig:
    base: str  # e.g., "https://www.vegrecipesofindia.com"
    per_site_max: int = 2000


@dataclass
class CrawlPlan:
    site: SiteConfig
    allowed: robotparser.RobotFileParser
    sitemap_urls: List[str]


ROBOTS_CACHE: Dict[str, robotparser.RobotFileParser] = {}
SITEMAP_XML_CACHE: Dict[str, Optional[str]] = {}


def get_robots_parser(base: str) -> robotparser.RobotFileParser:
    origin = get_origin(base)
    if origin in ROBOTS_CACHE:
        return ROBOTS_CACHE[origin]
    rp = robotparser.RobotFileParser()
    robots_url = urljoin(origin, "/robots.txt")
    try:
        rp.set_url(robots_url)
        rp.read()
    except Exception:
        pass
    ROBOTS_CACHE[origin] = rp
    return rp


def get_origin(url: str) -> str:
    p = urlparse(url)
    scheme = p.scheme or "https"
    netloc = p.netloc
    return f"{scheme}://{netloc}"


def discover_sitemaps(base: str, rp: robotparser.RobotFileParser) -> List[str]:
    origin = get_origin(base)
    # Try from robots.txt first
    sitemaps = getattr(rp, "site_maps", lambda: None)()
    urls: List[str] = []
    if sitemaps:
        urls.extend(sitemaps)
    # Fallbacks
    for guess in ["/sitemap.xml", "/sitemap_index.xml", "/sitemap-index.xml"]:
        candidate = urljoin(origin, guess)
        if candidate not in urls:
            urls.append(candidate)
    # Dedup while preserving order
    seen: Set[str] = set()
    out = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def fetch_text(url: str) -> Optional[str]:
    if url in SITEMAP_XML_CACHE:
        return SITEMAP_XML_CACHE[url]
    resp = http_get(url)
    text = resp.text if resp and resp.status_code == 200 else None
    SITEMAP_XML_CACHE[url] = text
    return text


SITEMAP_NS = {
    "sm": "http://www.sitemaps.org/schemas/sitemap/0.9",
    "image": "http://www.google.com/schemas/sitemap-image/1.1",
    "news": "http://www.google.com/schemas/sitemap-news/0.9",
}


def parse_sitemap_urls(xml_text: str) -> Tuple[List[str], List[str]]:
    """Return (sitemaps, urls) from a sitemap or index XML."""
    try:
        root = etree.fromstring(xml_text.encode("utf-8"))
    except Exception:
        return [], []
    sitemaps = [loc.text for loc in root.findall(".//sm:sitemap/sm:loc", SITEMAP_NS) if loc is not None and loc.text]
    urls = [loc.text for loc in root.findall(".//sm:url/sm:loc", SITEMAP_NS) if loc is not None and loc.text]
    return sitemaps, urls


def gather_urls_from_sitemaps(start_sitemaps: List[str], host: str, rp: robotparser.RobotFileParser, max_urls: int) -> List[str]:
    """BFS over sitemap indexes; collect up to max_urls URL entries belonging to host."""
    q = deque(start_sitemaps)
    collected: List[str] = []
    visited: Set[str] = set()
    while q and len(collected) < max_urls:
        sm_url = q.popleft()
        if sm_url in visited:
            continue
        visited.add(sm_url)
        xml_text = fetch_text(sm_url)
        if not xml_text:
            continue
        subs, urls = parse_sitemap_urls(xml_text)
        for s in subs:
            if s not in visited:
                q.append(s)
        for u in urls:
            if urlparse(u).netloc.endswith(urlparse(host).netloc):
                collected.append(u)
                if len(collected) >= max_urls:
                    break
    return collected



RECIPE_TYPE_RE = re.compile(r"Recipe$|\bRecipe\b", re.I)


def extract_jsonld_objects(html: str) -> List[Any]:
    soup = BeautifulSoup(html, "lxml")
    blocks = []
    for tag in soup.find_all("script", attrs={"type": re.compile("application/ld\+json", re.I)}):
        try:
            # Some sites have multiple JSON objects in a single script tag
            data = json.loads(tag.string or tag.text or "{}")
            if isinstance(data, list):
                blocks.extend(data)
            else:
                blocks.append(data)
        except Exception:
            # Try to recover basic JSON (e.g., trailing commas)
            with contextlib.suppress(Exception):
                cleaned = (tag.string or tag.text or "").strip()
                cleaned = re.sub(r"/\*.*?\*/", "", cleaned, flags=re.S)  # strip comments
                data = json.loads(cleaned)
                if isinstance(data, list):
                    blocks.extend(data)
                else:
                    blocks.append(data)
    return blocks


def find_recipe_nodes(objs: List[Any]) -> List[Dict[str, Any]]:
    found = []
    def walk(o: Any):
        if isinstance(o, dict):
            t = o.get("@type")
            if isinstance(t, list):
                if any(RECIPE_TYPE_RE.search(str(x)) for x in t):
                    found.append(o)
            elif isinstance(t, str) and RECIPE_TYPE_RE.search(t):
                found.append(o)
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(objs)
    # dedupe by id/name combo
    uniq = []
    seen = set()
    for r in found:
        key = (r.get("@id"), r.get("name"))
        if key not in seen:
            uniq.append(r)
            seen.add(key)
    return uniq


def normalize_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return re.sub(r"\s+", " ", x).strip()
    if isinstance(x, (int, float)):
        return str(x)
    if isinstance(x, list):
        return ", ".join(normalize_text(i) for i in x if i)
    if isinstance(x, dict):
        # For HowToStep / HowToSection
        if "text" in x:
            return normalize_text(x.get("text"))
        if "name" in x:
            return normalize_text(x.get("name"))
        return normalize_text(list(x.values()))
    return str(x)


def flatten_instructions(ri: Any) -> str:
    # recipeInstructions can be string, list of strings, list of HowToStep/HowToSection
    if isinstance(ri, str):
        return normalize_text(ri)
    if isinstance(ri, list):
        parts = []
        for step in ri:
            if isinstance(step, dict):
                # HowToSection with itemListElement or steps
                if step.get("@type") in ("HowToSection", "ItemList"):
                    inner = step.get("itemListElement") or step.get("steps")
                    if inner:
                        for s in inner:
                            parts.append(normalize_text(s.get("text") if isinstance(s, dict) else s))
                    else:
                        parts.append(normalize_text(step.get("name") or step.get("text")))
                else:
                    parts.append(normalize_text(step.get("text") or step.get("name")))
            else:
                parts.append(normalize_text(step))
        return " \n".join(p for p in parts if p)
    if isinstance(ri, dict):
        return normalize_text(ri.get("text") or ri.get("name"))
    return normalize_text(ri)


def extract_recipe_from_html(html: str, url: str) -> Optional[Dict[str, Any]]:
    objs = extract_jsonld_objects(html)
    recipes = find_recipe_nodes(objs)
    if not recipes:
        return None
    # choose the first recipe node
    r = recipes[0]
    def g(key: str, default: Any = ""):
        return r.get(key, default)
    # Aggregate rating fields
    rating = g("aggregateRating") or {}
    if isinstance(rating, dict):
        rating_value = rating.get("ratingValue")
        rating_count = rating.get("ratingCount") or rating.get("reviewCount")
    else:
        rating_value = None
        rating_count = None

    # Ingredients can be list or string
    ingredients = g("recipeIngredient") or g("ingredients")
    if ingredients is None:
        ingredients = []

    data = {
        "url": url,
        "name": normalize_text(g("name")),
        "description": normalize_text(g("description")),
        "image": normalize_text(g("image")),
        "author": normalize_text((g("author") or {}).get("name") if isinstance(g("author"), dict) else g("author")),
        "datePublished": normalize_text(g("datePublished")),
        "prepTime": normalize_text(g("prepTime")),
        "cookTime": normalize_text(g("cookTime")),
        "totalTime": normalize_text(g("totalTime")),
        "recipeYield": normalize_text(g("recipeYield")),
        "recipeCategory": normalize_text(g("recipeCategory")),
        "recipeCuisine": normalize_text(g("recipeCuisine")),
        "keywords": normalize_text(g("keywords")),
        "aggregateRating.value": normalize_text(rating_value),
        "aggregateRating.count": normalize_text(rating_count),
        "nutrition": normalize_text((g("nutrition") or {}).get("calories") if isinstance(g("nutrition"), dict) else g("nutrition")),
        "recipeIngredient": " | ".join(normalize_text(i) for i in (ingredients if isinstance(ingredients, list) else [ingredients]) if i),
        "recipeInstructions": flatten_instructions(g("recipeInstructions")),
    }
    # Drop empty-only recipes
    if not data["name"] and not data["recipeIngredient"]:
        return None
    return data



def url_is_allowed(rp: robotparser.RobotFileParser, url: str, ua: str = DEFAULT_HEADERS["User-Agent"]) -> bool:
    with contextlib.suppress(Exception):
        return rp.can_fetch(ua, url)
    return True


def looks_like_recipe_url(u: str) -> bool:
    u_low = u.lower()
    # Heuristics: adjust as needed
    return any(k in u_low for k in ["/recipe", "/recipes", "-recipe", "_recipe"]) and not any(
        x in u_low for x in ["tag/", "category/", "author/", "/page/", "/search?"]
    )


def gather_candidate_urls(plan: CrawlPlan) -> List[str]:
    host = get_origin(plan.site.base)
    # First try sitemaps
    urls: List[str] = gather_urls_from_sitemaps(plan.sitemap_urls, host, plan.allowed, plan.site.per_site_max * 3)
    # If sitemap missing/weak, do a lightweight homepage crawl for anchors
    if not urls:
        resp = http_get(host)
        if resp and resp.text:
            soup = BeautifulSoup(resp.text, "lxml")
            anchors = [urljoin(host, a.get("href")) for a in soup.find_all("a", href=True)]
            urls.extend(anchors)
    # Filter + dedupe + trim
    urls = [u for u in urls if isinstance(u, str) and u.startswith(host)]
    urls = [u for u in urls if looks_like_recipe_url(u)]
    # robots filter
    urls = [u for u in urls if url_is_allowed(plan.allowed, u)]
    # Dedup and limit
    seen: Set[str] = set()
    kept: List[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            kept.append(u)
        if len(kept) >= plan.site.per_site_max:
            break
    return kept


def crawl_site(plan: CrawlPlan, delay: float = 0.6) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    urls = gather_candidate_urls(plan)
    for u in urls:
        if not url_is_allowed(plan.allowed, u):
            continue
        resp = http_get(u)
        if not resp or not resp.text:
            continue
        recipe = extract_recipe_from_html(resp.text, u)
        if recipe:
            results.append(recipe)
        time.sleep(delay + random.uniform(0, 0.4))
    return results




def build_plan(site: SiteConfig) -> CrawlPlan:
    rp = get_robots_parser(site.base)
    sm = discover_sitemaps(site.base, rp)
    return CrawlPlan(site=site, allowed=rp, sitemap_urls=sm)


def write_outputs(records: List[Dict[str, Any]], out_prefix: str) -> Tuple[str, str]:
    # JSONL
    jsonl_path = f"{out_prefix}.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    # CSV
    csv_path = f"{out_prefix}.csv"
    # Normalize columns
    all_keys: List[str] = []
    for r in records:
        for k in r.keys():
            if k not in all_keys:
                all_keys.append(k)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        for r in records:
            writer.writerow(r)
    return jsonl_path, csv_path


def main():
    parser = argparse.ArgumentParser(description="Scrape Indian recipes from multiple websites (JSON-LD based)")
    parser.add_argument("--out-prefix", default="indian_recipes", help="Output file prefix (creates .jsonl and .csv)")
    parser.add_argument("--per-site-max", type=int, default=2000, help="Max recipe URLs to attempt per site")
    parser.add_argument("--workers", type=int, default=6, help="Number of sites to crawl in parallel")
    parser.add_argument("--delay", type=float, default=0.6, help="Base delay between requests per site (seconds)")
    args = parser.parse_args()

    # 15 starter sites (edit as needed)
    SITES = [
        "https://www.vegrecipesofindia.com",     # Veg Recipes of India
        "https://www.indianhealthyrecipes.com",  # Swasthi's Recipes
        "https://www.tarladalal.com",            # Tarla Dalal
        "https://www.archanaskitchen.com",       # Archana's Kitchen
        "https://hebbarskitchen.com",            # Hebbar's Kitchen
        "https://www.vahrehvah.com",             # VahChef
        "https://www.sanjeevkapoor.com",         # Sanjeev Kapoor
        "https://nishamadhulika.com",            # Nisha Madhulika (Hindi)
        "https://www.spiceupthecurry.com",       # Spice Up The Curry
        "https://www.cookwithmanali.com",        # Cook With Manali
        "https://www.rakskitchen.net",           # Raks Kitchen
        "https://www.ministryofcurry.com",       # Ministry of Curry
        "https://www.maayeka.com",               # Maayeka
        "https://www.manjulaskitchen.com",       # Manjula's Kitchen
        "https://bharatzkitchen.com",            # Bharat's Kitchen (Hindi)
    ]

    site_confs = [SiteConfig(base=s, per_site_max=args.per_site_max) for s in SITES]

    plans = [build_plan(sc) for sc in site_confs]

    all_records: List[Dict[str, Any]] = []
    seen_urls: Set[str] = set()

    def run_plan(pl: CrawlPlan) -> List[Dict[str, Any]]:
        try:
            return crawl_site(pl, delay=args.delay)
        except Exception as e:
            print(f"[WARN] Site failed: {pl.site.base}: {e}")
            return []

    with futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for res in tqdm(ex.map(run_plan, plans), total=len(plans), desc="Sites"):
            for r in res:
                if r.get("url") not in seen_urls:
                    seen_urls.add(r.get("url"))
                    all_records.append(r)

    if not all_records:
        print("No recipes found. Try increasing per-site-max or adjusting delay.")
        return

    jsonl_path, csv_path = write_outputs(all_records, args.out_prefix)
    print(f"Saved {len(all_records)} recipes\nJSONL: {jsonl_path}\nCSV: {csv_path}")


if __name__ == "__main__":
    main()
