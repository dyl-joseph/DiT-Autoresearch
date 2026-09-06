from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Iterable


def iter_markdown(root: Path) -> Iterable[Path]:
    for p in sorted(root.rglob("*.md")):
        if p.is_file():
            yield p


def _title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def build_index(knowledge_dir: Path, db_path: Path) -> int:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("CREATE TABLE docs_meta(path TEXT PRIMARY KEY, title TEXT NOT NULL, text TEXT NOT NULL)")
        fts_ok = True
        try:
            conn.execute("CREATE VIRTUAL TABLE docs_fts USING fts5(path UNINDEXED, title, text)")
        except sqlite3.OperationalError:
            fts_ok = False
        count = 0
        for p in iter_markdown(knowledge_dir):
            text = p.read_text(encoding="utf-8", errors="replace")
            rel = str(p.relative_to(knowledge_dir)).replace("\\", "/")
            title = _title(text, p.stem)
            conn.execute("INSERT INTO docs_meta(path,title,text) VALUES(?,?,?)", (rel, title, text))
            if fts_ok:
                conn.execute("INSERT INTO docs_fts(path,title,text) VALUES(?,?,?)", (rel, title, text))
            count += 1
        conn.execute("CREATE TABLE settings(key TEXT PRIMARY KEY, value TEXT)")
        conn.execute("INSERT INTO settings(key,value) VALUES('fts5',?)", ("1" if fts_ok else "0",))
        conn.commit()
        return count
    finally:
        conn.close()


def _fallback_score(query: str, text: str, title: str) -> float:
    terms = [t for t in re.findall(r"[a-zA-Z0-9_+.-]+", query.lower()) if len(t) >= 2]
    low = text.lower()
    ttl = title.lower()
    score = 0.0
    for term in terms:
        score += 5.0 * ttl.count(term)
        score += low.count(term)
    return score


def search_index(db_path: Path, query: str, limit: int = 8) -> list[dict[str, object]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        fts = conn.execute("SELECT value FROM settings WHERE key='fts5'").fetchone()
        if fts and fts[0] == "1":
            # Quote punctuation-heavy terms individually to avoid FTS syntax surprises.
            terms = re.findall(r"[a-zA-Z0-9_]+", query)
            expr = " OR ".join(f'"{t}"' for t in terms if t)
            if expr:
                rows = conn.execute(
                    "SELECT path,title,snippet(docs_fts,2,'[[',']]', ' … ', 28) AS snippet, bm25(docs_fts) AS rank "
                    "FROM docs_fts WHERE docs_fts MATCH ? ORDER BY rank LIMIT ?",
                    (expr, limit),
                ).fetchall()
                if rows:
                    return [dict(r) for r in rows]
        rows = conn.execute("SELECT path,title,text FROM docs_meta").fetchall()
        scored = []
        for r in rows:
            score = _fallback_score(query, r["text"], r["title"])
            if score <= 0:
                continue
            text = r["text"].replace("\n", " ")
            scored.append({"path": r["path"], "title": r["title"], "snippet": text[:420], "rank": -score})
        scored.sort(key=lambda x: float(x["rank"]))
        return scored[:limit]
    finally:
        conn.close()
