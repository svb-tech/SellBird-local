"""SQLite store for the local sales agent (separate file: data/local_sales.db)."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

from sellbird.paths import PROJECT_ROOT

DB_PATH = PROJECT_ROOT / "data" / "local_sales.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS business (id INTEGER PRIMARY KEY CHECK (id = 1), name TEXT, raw_info TEXT,
  profile_json TEXT, model TEXT, updated_at REAL);
CREATE TABLE IF NOT EXISTS products (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE,
  description TEXT, pricing TEXT, details TEXT, icp_json TEXT, icp_model TEXT, created_at REAL);
CREATE TABLE IF NOT EXISTS prospects (id INTEGER PRIMARY KEY AUTOINCREMENT, company TEXT UNIQUE,
  contact_name TEXT, title TEXT, email TEXT, industry TEXT, size INTEGER, location TEXT,
  source TEXT, raw_notes TEXT, research_json TEXT, research_model TEXT, created_at REAL);
CREATE TABLE IF NOT EXISTS signals (id INTEGER PRIMARY KEY AUTOINCREMENT, prospect_id INTEGER,
  product_id INTEGER, signal_type TEXT, strength INTEGER, evidence TEXT, grounded INTEGER,
  method TEXT, model TEXT, created_at REAL);
CREATE TABLE IF NOT EXISTS problems (id INTEGER PRIMARY KEY AUTOINCREMENT, prospect_id INTEGER,
  product_id INTEGER, statement TEXT, evidence_json TEXT, model TEXT, created_at REAL);
CREATE TABLE IF NOT EXISTS scores (id INTEGER PRIMARY KEY AUTOINCREMENT, prospect_id INTEGER,
  product_id INTEGER, total REAL, tier TEXT, breakdown_json TEXT, created_at REAL);
CREATE TABLE IF NOT EXISTS emails (id INTEGER PRIMARY KEY AUTOINCREMENT, prospect_id INTEGER,
  product_id INTEGER, subject TEXT, body TEXT, checks_json TEXT, model TEXT, status TEXT DEFAULT 'draft',
  created_at REAL);
CREATE TABLE IF NOT EXISTS replies (id INTEGER PRIMARY KEY AUTOINCREMENT, prospect_id INTEGER,
  text TEXT, category TEXT, confidence REAL, rationale TEXT, next_action TEXT, method TEXT,
  model TEXT, expected TEXT, created_at REAL);
CREATE TABLE IF NOT EXISTS task_queue (id INTEGER PRIMARY KEY AUTOINCREMENT, workflow TEXT,
  payload_json TEXT, status TEXT DEFAULT 'queued', error TEXT, attempts INTEGER DEFAULT 0,
  created_at REAL, done_at REAL);
"""


class Store:
    def __init__(self, path: Path | str | None = None):
        self.path = str(path or DB_PATH)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # generic helpers --------------------------------------------------
    def q(self, sql: str, args: tuple = ()) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self.conn.execute(sql, args).fetchall()]

    def one(self, sql: str, args: tuple = ()) -> dict | None:
        rows = self.q(sql, args)
        return rows[0] if rows else None

    def x(self, sql: str, args: tuple = ()) -> int:
        with self._lock:
            cur = self.conn.execute(sql, args)
            self.conn.commit()
            return cur.lastrowid

    # business / products ---------------------------------------------
    def save_business(self, name: str, raw_info: str, profile: dict | None = None, model: str = ""):
        self.x(
            "INSERT INTO business(id,name,raw_info,profile_json,model,updated_at) VALUES(1,?,?,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET name=excluded.name, raw_info=excluded.raw_info, "
            "profile_json=excluded.profile_json, model=excluded.model, updated_at=excluded.updated_at",
            (name, raw_info, json.dumps(profile) if profile else None, model, time.time()),
        )

    def business(self) -> dict | None:
        b = self.one("SELECT * FROM business WHERE id=1")
        if b and b["profile_json"]:
            b["profile"] = json.loads(b["profile_json"])
        return b

    def upsert_product(self, name: str, description: str, pricing: str = "", details: str = "") -> int:
        self.x(
            "INSERT INTO products(name,description,pricing,details,created_at) VALUES(?,?,?,?,?) "
            "ON CONFLICT(name) DO UPDATE SET description=excluded.description, pricing=excluded.pricing, "
            "details=excluded.details",
            (name, description, pricing, details, time.time()),
        )
        return self.one("SELECT id FROM products WHERE name=?", (name,))["id"]

    def product(self, pid: int) -> dict | None:
        p = self.one("SELECT * FROM products WHERE id=?", (pid,))
        if p and p["icp_json"]:
            p["icp"] = json.loads(p["icp_json"])
        return p

    def products(self) -> list[dict]:
        return [self.product(r["id"]) for r in self.q("SELECT id FROM products ORDER BY id")]

    # prospects --------------------------------------------------------
    def upsert_prospect(self, d: dict) -> int:
        self.x(
            "INSERT INTO prospects(company,contact_name,title,email,industry,size,location,source,"
            "raw_notes,created_at) VALUES(?,?,?,?,?,?,?,?,?,?) ON CONFLICT(company) DO UPDATE SET "
            "contact_name=excluded.contact_name, title=excluded.title, email=excluded.email, "
            "industry=excluded.industry, size=excluded.size, location=excluded.location, "
            "source=excluded.source, raw_notes=excluded.raw_notes",
            (d["company"], d.get("contact_name", ""), d.get("title", ""), d.get("email", ""),
             d.get("industry", ""), int(d.get("size") or 0), d.get("location", ""),
             d.get("source", "local-file"), d.get("raw_notes", ""), time.time()),
        )
        return self.one("SELECT id FROM prospects WHERE company=?", (d["company"],))["id"]

    def prospect(self, pid: int) -> dict | None:
        p = self.one("SELECT * FROM prospects WHERE id=?", (pid,))
        if p and p["research_json"]:
            p["research"] = json.loads(p["research_json"])
        return p

    def prospects(self) -> list[dict]:
        return [self.prospect(r["id"]) for r in self.q("SELECT id FROM prospects ORDER BY id")]

    # queue ------------------------------------------------------------
    def enqueue(self, workflow: str, payload: dict, error: str) -> int:
        existing = self.one(
            "SELECT id FROM task_queue WHERE status='queued' AND workflow=? AND payload_json=?",
            (workflow, json.dumps(payload, sort_keys=True)),
        )
        if existing:
            return existing["id"]
        return self.x(
            "INSERT INTO task_queue(workflow,payload_json,error,created_at) VALUES(?,?,?,?)",
            (workflow, json.dumps(payload, sort_keys=True), error[:300], time.time()),
        )

    def queue(self, status: str | None = None) -> list[dict]:
        if status:
            return self.q("SELECT * FROM task_queue WHERE status=? ORDER BY id", (status,))
        return self.q("SELECT * FROM task_queue ORDER BY id")
