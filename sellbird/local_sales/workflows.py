"""The eight local sales-agent workflows.

Every AI step calls LocalLLM (Ollama). When local inference is unavailable a
step either (a) runs a clearly-labelled deterministic rules fallback and queues
an LLM re-run, or (b) is queued for later. There is no cloud fallback.

What is deterministic (NOT AI) is labelled `method: "rules"` in the output:
  * ingestion of company/product files and prospect files
  * the rule-based ICP-match and signal-weight parts of the score
  * unsubscribe / out-of-office detection in reply classification
  * validation checks on generated emails and the appended signature
"""
from __future__ import annotations

import csv
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

from sellbird.local_sales.embeddings import Embedder
from sellbird.local_sales.llm import LocalAIUnavailable, LocalLLM
from sellbird.local_sales.store import Store
from sellbird.paths import PROJECT_ROOT

DATA_DIR = PROJECT_ROOT / "data" / "local"

SIGNAL_TYPES = [
    "hiring", "funding", "security_incident", "compliance_deadline",
    "customer_security_demand", "infrastructure_change", "leadership_change",
    "tool_replacement", "no_signal",
]
REPLY_CATEGORIES = [
    "interested", "meeting_request", "question", "not_interested", "objection_price",
    "objection_timing", "referral", "out_of_office", "unsubscribe",
]
HYPE_WORDS = ["revolutionary", "game-changing", "game changer", "cutting-edge", "world-class",
              "best-in-class", "synergy", "unlock", "skyrocket", "guaranteed"]


class PreconditionError(Exception):
    """A required earlier step has not run yet (e.g. no ICP). Not an AI failure."""


@dataclass
class Ctx:
    store: Store
    llm: LocalLLM
    embedder: Embedder


# --------------------------------------------------------------- schema helpers
def _str_list():
    return {"type": "array", "items": {"type": "string"}}


def _obj(props: dict, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": props, "required": required or list(props)}


S_PROFILE = _obj({
    "company_summary": {"type": "string"}, "value_propositions": _str_list(),
    "target_customers": _str_list(), "differentiators": _str_list(), "brand_voice": {"type": "string"},
    "products": {"type": "array", "items": _obj({"name": {"type": "string"}, "one_liner": {"type": "string"}})},
})
S_ICP = _obj({
    "industries": _str_list(), "min_employees": {"type": "integer"}, "max_employees": {"type": "integer"},
    "buyer_titles": _str_list(), "geographies": _str_list(), "pain_points": _str_list(),
    "buying_triggers": _str_list(), "disqualifiers": _str_list(), "messaging_angle": {"type": "string"},
})
S_RESEARCH = _obj({
    "summary": {"type": "string"}, "key_facts": _str_list(), "tech_environment": {"type": "string"},
    "recent_events": _str_list(), "pain_hints": _str_list(),
})
S_SIGNALS = _obj({"signals": {"type": "array", "items": _obj({
    "type": {"type": "string", "enum": SIGNAL_TYPES}, "strength": {"type": "integer"},
    "evidence": {"type": "string"}, "why_it_matters": {"type": "string"}})}})
S_PROBLEM = _obj({"problem_statement": {"type": "string"}, "evidence_used": _str_list()})
S_FIT = _obj({"fit_score": {"type": "integer"}, "reason": {"type": "string"}})
S_EMAIL = _obj({"subject": {"type": "string"}, "body": {"type": "string"}})
# Order matters: the model writes its one-sentence rationale BEFORE choosing the category,
# which helps small models; the category itself is constrained to the enum.
S_REPLY = _obj({"rationale": {"type": "string"},
                "category": {"type": "string", "enum": REPLY_CATEGORIES},
                "confidence": {"type": "number"},
                "next_action": {"type": "string"}})


def _clip(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n] + " …"


def _strs(v) -> list[str]:
    return [str(x).strip() for x in (v or []) if str(x).strip()]


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2}


# ============================================================ 0. ingestion (rules)
def ingest_company_files(store: Store, directory: Path | None = None) -> dict:
    """Deterministic file ingestion (not AI): company overview + one file per product."""
    d = Path(directory or DATA_DIR / "company_info")
    files = sorted(d.glob("*.md"))
    if not files:
        raise FileNotFoundError(f"No company files in {d}")
    overview, product_ids = [], []
    for f in files:
        text = f.read_text(encoding="utf-8")
        m = re.match(r"#\s*Product:\s*(.+)", text)
        if m:
            name = m.group(1).strip()
            lines = [ln.strip("- ").strip() for ln in text.splitlines()[1:] if ln.strip()]
            desc = lines[0] if lines else name
            pricing = next((ln.split("Pricing:", 1)[1].strip() for ln in lines if "Pricing:" in ln), "")
            product_ids.append(store.upsert_product(name, desc, pricing, text))
        else:
            overview.append(text)
    first = overview[0] if overview else ""
    m = re.search(r"#\s*(.+?)\s+—", first)
    name = m.group(1).strip() if m else "Company"
    store.save_business(name, "\n\n".join(overview))
    return {"business": name, "files": [f.name for f in files], "product_ids": product_ids}


def load_prospects(store: Store, path: Path | str | None = None) -> dict:
    """Load permitted prospect data from a local JSON/CSV file (deterministic)."""
    p = Path(path or DATA_DIR / "prospects.json")
    if p.suffix == ".csv":
        rows = list(csv.DictReader(p.open(encoding="utf-8")))
    else:
        raw = json.loads(p.read_text(encoding="utf-8"))
        rows = raw["prospects"] if isinstance(raw, dict) else raw
    ids = [store.upsert_prospect(r) for r in rows if r.get("company")]
    return {"loaded": len(ids), "prospect_ids": ids, "source": str(p)}


# ========================================================== 1. business profile
def wf_business_profile(ctx: Ctx, payload: dict) -> dict:
    biz = ctx.store.business()
    if not biz:
        raise PreconditionError("No company information loaded. Click \"Load company + prospect files\" first.")
    system = ("You analyse a company's own documents and produce a structured business profile. "
              "Use ONLY facts stated in the documents. Do not invent customers, numbers or products.")
    user = f"COMPANY DOCUMENTS:\n{_clip(biz['raw_info'], 5000)}\n\nPRODUCTS KNOWN:\n" + \
           "\n".join(f"- {p['name']}: {p['description']}" for p in ctx.store.products())
    profile, rec = ctx.llm.generate_json("business_profile", system, user, S_PROFILE)
    ctx.store.save_business(biz["name"], biz["raw_info"], profile, rec.get("model_served") or ctx.llm.model)
    return {"profile": profile, "inference_id": rec["id"], "model": ctx.llm.model}


# =============================================================== 2. product ICP
def wf_icp(ctx: Ctx, payload: dict) -> dict:
    prod = ctx.store.product(payload["product_id"])
    biz = ctx.store.business()
    if not prod:
        raise ValueError("Unknown product_id")
    system = ("You are a B2B sales strategist. Build an Ideal Customer Profile for ONE product, "
              "grounded only in the product sheet and business profile. min_employees and max_employees "
              "must be integers. Titles must be real job titles. Keep every list to at most 6 short items.")
    profile = json.dumps((biz or {}).get("profile", {}))[:1500]
    user = f"PRODUCT SHEET:\n{_clip(prod['details'], 2500)}\n\nBUSINESS PROFILE:\n{profile}"
    icp, rec = ctx.llm.generate_json("icp", system, user, S_ICP)
    icp["min_employees"] = max(0, int(icp.get("min_employees") or 0))
    icp["max_employees"] = int(icp.get("max_employees") or 0) or 100000
    ctx.store.x("UPDATE products SET icp_json=?, icp_model=? WHERE id=?",
                (json.dumps(icp), rec.get("model_served") or ctx.llm.model, prod["id"]))
    return {"product": prod["name"], "icp": icp, "inference_id": rec["id"]}


# ========================================================== 3. prospect research
def wf_research(ctx: Ctx, payload: dict) -> dict:
    pr = ctx.store.prospect(payload["prospect_id"])
    if not pr:
        raise ValueError("Unknown prospect_id")
    notes = pr["raw_notes"] or ""
    # Also include any extra locally downloaded page text stored under data/local/research/<slug>.txt
    slug = re.sub(r"[^a-z0-9]+", "-", pr["company"].lower()).strip("-")
    extra = DATA_DIR / "research" / f"{slug}.txt"
    if extra.exists():
        notes += "\n" + extra.read_text(encoding="utf-8")
    system = ("You research a prospect company from locally stored web-page text. "
              "Extract only what the text says; write 'unknown' where it is silent. No speculation.")
    user = (f"COMPANY: {pr['company']} ({pr['industry']}, {pr['size']} employees, {pr['location']})\n"
            f"CONTACT: {pr['contact_name']}, {pr['title']}\n\nSTORED TEXT:\n{_clip(notes, 3500)}")
    research, rec = ctx.llm.generate_json("research", system, user, S_RESEARCH)
    ctx.store.x("UPDATE prospects SET research_json=?, research_model=? WHERE id=?",
                (json.dumps(research), rec.get("model_served") or ctx.llm.model, pr["id"]))
    return {"company": pr["company"], "research": research, "inference_id": rec["id"]}


# ======================================================= 4. buying-signal classify
_RULES = {
    "hiring": r"\bhiring\b|job (posting|ad)|careers page|open role",
    "funding": r"\bfunding\b|series [a-d]\b|seed round|raised \$",
    "security_incident": r"ransomware|cyber incident|phishing attack|breach|halted one plant|encrypted",
    "compliance_deadline": r"audit (scheduled|by|in)|audit-ready|soc 2|iso 27001|hipaa|pci-dss|regulator|circular",
    "customer_security_demand": r"security questionnaire|asked for (a )?soc 2|requires iso|security review",
    "infrastructure_change": r"migrat|moved its|consolidat|private cloud|replac\w+ the legacy",
    "leadership_change": r"new (cio|cto|ciso)\b|hired in",
    "tool_replacement": r"legacy .{0,30}(retire|replace)|replacing the legacy|evaluating (vendors|options)",
}


def _grounded(evidence: str, notes: str) -> bool:
    ew, nw = _words(evidence), _words(notes)
    return bool(ew) and len(ew & nw) / len(ew) >= 0.8


def _rule_signals(notes: str) -> list[dict]:
    out = []
    for stype, rx in _RULES.items():
        m = re.search(rx, notes, flags=re.I)
        if m:
            start = max(0, m.start() - 60)
            out.append({"type": stype, "strength": 3, "evidence": notes[start : m.end() + 80].strip(),
                        "why_it_matters": "keyword rule match (no LLM)"})
    return out


def _persist_signals(ctx: Ctx, pr: dict, prod_id: int, signals: list[dict], method: str, model: str):
    ctx.store.x("DELETE FROM signals WHERE prospect_id=? AND product_id=?", (pr["id"], prod_id))
    stored = []
    for s in signals:
        stype = s.get("type") if s.get("type") in SIGNAL_TYPES else "no_signal"
        if stype == "no_signal":
            continue
        strength = min(5, max(1, int(s.get("strength") or 1)))
        ev = str(s.get("evidence", "")).strip()
        g = 1 if _grounded(ev, pr["raw_notes"]) else 0
        ctx.store.x("INSERT INTO signals(prospect_id,product_id,signal_type,strength,evidence,grounded,"
                    "method,model,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                    (pr["id"], prod_id, stype, strength, ev, g, method, model, time.time()))
        stored.append({"type": stype, "strength": strength, "evidence": ev, "grounded": bool(g),
                       "why_it_matters": s.get("why_it_matters", ""), "method": method})
    return stored


def wf_signals_rules(ctx: Ctx, payload: dict) -> dict:
    pr = ctx.store.prospect(payload["prospect_id"])
    stored = _persist_signals(ctx, pr, payload["product_id"], _rule_signals(pr["raw_notes"]), "rules", "none")
    return {"company": pr["company"], "signals": stored, "method": "rules"}


def wf_signals(ctx: Ctx, payload: dict) -> dict:
    pr = ctx.store.prospect(payload["prospect_id"])
    prod = ctx.store.product(payload["product_id"])
    icp = prod.get("icp", {})
    system = ("You classify BUYING SIGNALS for a sales team. A buying signal is a fact in the text that "
              "suggests the company may need THIS product soon. For each signal give: type (from the enum), "
              "strength 1-5 for THIS product (5 = strong, imminent need), and 'evidence' copied "
              "word-for-word from the text. Return an empty list if there is no real signal. Never invent evidence.")
    research = json.dumps(pr.get("research", {}))[:1200]
    user = (f"PRODUCT: {prod['name']} — {prod['description']}\nBUYING TRIGGERS FOR THIS PRODUCT: "
            f"{', '.join(_strs(icp.get('buying_triggers')))}\n\nPROSPECT: {pr['company']}\n"
            f"TEXT:\n{_clip(pr['raw_notes'], 3000)}\n\nRESEARCH SUMMARY: {research}")
    data, rec = ctx.llm.generate_json("signals", system, user, S_SIGNALS)
    stored = _persist_signals(ctx, pr, prod["id"], data.get("signals", []),
                              "llm", rec.get("model_served") or ctx.llm.model)
    dropped = sum(1 for s in stored if not s["grounded"])
    return {"company": pr["company"], "signals": stored, "ungrounded_signals": dropped,
            "method": "llm", "inference_id": rec["id"]}


# ======================================================= 5. problem statement
def _signals_for(ctx: Ctx, pid: int, prod_id: int, grounded_only=True) -> list[dict]:
    rows = ctx.store.q("SELECT * FROM signals WHERE prospect_id=? AND product_id=? ORDER BY strength DESC",
                       (pid, prod_id))
    return [r for r in rows if r["grounded"]] if grounded_only else rows


def wf_problem(ctx: Ctx, payload: dict) -> dict:
    pr = ctx.store.prospect(payload["prospect_id"])
    prod = ctx.store.product(payload["product_id"])
    sigs = _signals_for(ctx, pr["id"], prod["id"])[:3]
    if not sigs:
        return {"company": pr["company"], "problem_statement": None,
                "note": "No grounded buying signal; no problem statement generated."}
    system = ("Write ONE problem statement (max 2 sentences) describing the business problem this prospect "
              "likely faces that the product can help with. Base it ONLY on the evidence given. "
              "No hype, no product pitch, no made-up numbers.")
    ev = "\n".join(f"- [{s['signal_type']}] {s['evidence']}" for s in sigs)
    user = (f"PROSPECT: {pr['company']} ({pr['industry']}), contact {pr['contact_name']}, {pr['title']}\n"
            f"PRODUCT: {prod['name']} — {prod['description']}\nEVIDENCE:\n{ev}")
    data, rec = ctx.llm.generate_json("problem_statement", system, user, S_PROBLEM)
    stmt = str(data.get("problem_statement", "")).strip()
    ctx.store.x("DELETE FROM problems WHERE prospect_id=? AND product_id=?", (pr["id"], prod["id"]))
    ctx.store.x("INSERT INTO problems(prospect_id,product_id,statement,evidence_json,model,created_at)"
                " VALUES(?,?,?,?,?,?)",
                (pr["id"], prod["id"], stmt, json.dumps([s["evidence"] for s in sigs]),
                 rec.get("model_served") or ctx.llm.model, time.time()))
    return {"company": pr["company"], "problem_statement": stmt, "evidence": [s["evidence"] for s in sigs],
            "inference_id": rec["id"]}


# ============================================================== 6. scoring
def _score_parts(ctx: Ctx, pr: dict, prod: dict) -> dict:
    icp = prod.get("icp") or {}
    bd: dict = {}
    # --- rule-based ICP match (40) ---
    ind_w = _words(pr["industry"])
    ind_hit = any(ind_w & _words(i) for i in _strs(icp.get("industries")))
    bd["industry"] = 15 if ind_hit else 0
    size = pr["size"] or 0
    lo, hi = icp.get("min_employees", 0), icp.get("max_employees", 10**6)
    bd["company_size"] = 10 if lo <= size <= hi else (4 if lo * 0.5 <= size <= hi * 1.5 else 0)
    t_w = _words(pr["title"])
    bd["buyer_title"] = 10 if any(t_w & _words(t) for t in _strs(icp.get("buyer_titles"))) else 0
    geos = _strs(icp.get("geographies"))
    bd["geography"] = 5 if (not geos or any(g.lower() in (pr["location"] or "").lower() or
                                             (pr["location"] or "").lower() in g.lower() for g in geos)) else 0
    # --- signals (30), grounded only ---
    sigs = _signals_for(ctx, pr["id"], prod["id"])[:3]
    bd["signals"] = round(sum(s["strength"] for s in sigs) / 15 * 30, 1)
    # --- semantic similarity (15) ---
    icp_text = " ".join(_strs(icp.get("pain_points")) + _strs(icp.get("buying_triggers")) +
                        _strs(icp.get("industries")) + [prod["description"]])
    sim = ctx.embedder.similarity(icp_text, pr["raw_notes"] + " " + json.dumps(pr.get("research", {})))
    denom = 0.5 if ctx.embedder.backend.startswith("sentence") else 0.25
    bd["semantic_similarity"] = round(min(1.0, sim / denom) * 15, 1)
    bd["_similarity_raw"] = round(sim, 3)
    bd["_embedding_backend"] = ctx.embedder.backend
    return bd


def _tier(total: float) -> str:
    return "A" if total >= 70 else "B" if total >= 50 else "C"


def wf_score(ctx: Ctx, payload: dict, with_llm: bool = True) -> dict:
    pr = ctx.store.prospect(payload["prospect_id"])
    prod = ctx.store.product(payload["product_id"])
    if not prod.get("icp"):
        raise PreconditionError("Generate the product ICP before scoring.")
    bd = _score_parts(ctx, pr, prod)
    rule_total = bd["industry"] + bd["company_size"] + bd["buyer_title"] + bd["geography"] + \
        bd["signals"] + bd["semantic_similarity"]
    rec = None
    if with_llm:
        system = ("Rate how well this prospect fits the product's Ideal Customer Profile on a 0-10 scale "
                  "(10 = ideal, 0 = clearly wrong). Judge only from the text. Give a one-sentence reason.")
        user = (f"ICP: {json.dumps(prod['icp'])[:1200]}\nPROSPECT: {pr['company']}, {pr['industry']}, "
                f"{pr['size']} employees, {pr['title']}\nNOTES: {_clip(pr['raw_notes'], 1200)}")
        data, rec = ctx.llm.generate_json("score_fit", system, user, S_FIT)
        fit = min(10, max(0, int(data.get("fit_score") or 0)))
        bd["llm_fit"] = round(fit * 1.5, 1)
        bd["_llm_reason"] = str(data.get("reason", ""))
        total = rule_total + bd["llm_fit"]
        bd["_llm_component"] = "included"
    else:
        total = rule_total / 85 * 100
        bd["_llm_component"] = "unavailable - total rescaled from the 85 rule-based points"
    total = round(total, 1)
    tier = _tier(total)
    ctx.store.x("DELETE FROM scores WHERE prospect_id=? AND product_id=?", (pr["id"], prod["id"]))
    ctx.store.x("INSERT INTO scores(prospect_id,product_id,total,tier,breakdown_json,created_at) VALUES(?,?,?,?,?,?)",
                (pr["id"], prod["id"], total, tier, json.dumps(bd), time.time()))
    out = {"company": pr["company"], "total": total, "tier": tier, "breakdown": bd}
    if rec:
        out["inference_id"] = rec["id"]
    return out


def wf_score_rules(ctx: Ctx, payload: dict) -> dict:
    out = wf_score(ctx, payload, with_llm=False)
    out["method"] = "rules"
    return out


# ============================================================ 7. email generation
def _email_checks(subject: str, body: str, pr: dict) -> dict:
    wc = len(body.split())
    first = (pr["contact_name"] or "").split(" ")[0].lower()
    lowered = (subject + " " + body).lower()
    checks = {
        "word_count_ok": 40 <= wc <= 130,
        "mentions_contact_or_company": bool(first and first in lowered) or pr["company"].lower() in lowered,
        "no_placeholders": not re.search(r"\[[^\]]+\]|\{\{|\}\}", subject + body),
        "no_hype_words": not any(w in lowered for w in HYPE_WORDS),
        "single_question_cta": body.count("?") == 1,
    }
    checks["passed"] = all(checks.values())
    checks["word_count"] = wc
    return checks


def wf_email(ctx: Ctx, payload: dict) -> dict:
    pr = ctx.store.prospect(payload["prospect_id"])
    prod = ctx.store.product(payload["product_id"])
    biz = ctx.store.business() or {}
    problem = ctx.store.one("SELECT * FROM problems WHERE prospect_id=? AND product_id=?", (pr["id"], prod["id"]))
    sigs = _signals_for(ctx, pr["id"], prod["id"])[:2]
    if not sigs:
        return {"company": pr["company"], "email": None,
                "note": "No grounded buying signal; email not generated (no basis for personalisation)."}
    voice = (biz.get("profile") or {}).get("brand_voice", "calm, practical, specific")
    angle = (prod.get("icp") or {}).get("messaging_angle", "")
    system = ("You write short, personalised B2B cold emails. Rules: 60-110 words; address the contact by first "
              "name; open with ONE specific observation taken from the evidence; connect it to the product in "
              "plain words; exactly ONE question as the call to action; no hype words, no bullet lists, no "
              "placeholders, no invented facts or numbers; do not include a signature or greeting line break "
              f"tricks. Tone: {voice}.")
    user = (f"CONTACT: {pr['contact_name']}, {pr['title']} at {pr['company']}\n"
            f"PRODUCT: {prod['name']} — {prod['description']}. Pricing: {prod['pricing']}\n"
            f"MESSAGING ANGLE: {angle}\nPROBLEM STATEMENT: {(problem or {}).get('statement', '')}\n"
            "EVIDENCE:\n" + "\n".join(f"- {s['evidence']}" for s in sigs))
    data, rec = ctx.llm.generate_json("email", system, user, S_EMAIL)
    subject, body = str(data.get("subject", "")).strip(), str(data.get("body", "")).strip()
    checks = _email_checks(subject, body, pr)
    if not checks["passed"]:  # one targeted repair attempt
        failed = [k for k, v in checks.items() if v is False]
        data, rec = ctx.llm.generate_json(
            "email_repair", system,
            user + f"\n\nYour previous draft failed these checks: {failed}.\nPrevious draft:\n{body}\n"
                   "Rewrite it so all rules pass.", S_EMAIL)
        subject, body = str(data.get("subject", subject)).strip(), str(data.get("body", body)).strip()
        checks = _email_checks(subject, body, pr)
        checks["repaired"] = True
    sender = f"Business Development, {biz.get('name', 'Our team')}"
    full_body = f"{body}\n\nBest,\n{sender}"  # signature is a deterministic template, not model output
    ctx.store.x("INSERT INTO emails(prospect_id,product_id,subject,body,checks_json,model,created_at) "
                "VALUES(?,?,?,?,?,?,?)",
                (pr["id"], prod["id"], subject, full_body, json.dumps(checks),
                 rec.get("model_served") or ctx.llm.model, time.time()))
    return {"company": pr["company"], "subject": subject, "body": full_body, "checks": checks,
            "delivery": "draft - never auto-sent", "inference_id": rec["id"]}


# ======================================================== 8. reply classification
_UNSUB = re.compile(r"\bunsubscribe\b|remove me\b|take me off|stop (emailing|contacting|sending)|"
                    r"do not (email|contact)|don'?t (email|contact) me", re.I)
_OOO = re.compile(r"out of (the )?office|auto-?reply|on (annual )?leave|limited access to (my )?email|"
                  r"back (in the office )?on", re.I)
_FALLBACK = [
    ("meeting_request", r"\b(call|demo|meeting|book|schedule)\b.*\b(thursday|friday|monday|tuesday|wednesday|week|tomorrow|today)\b|let'?s (book|schedule)"),
    ("referral", r"not the right person|talk to our|reach out to|you should (talk|speak)"),
    ("objection_price", r"\b(price|pricing|expensive|cost|cheaper|budget)\b.*\b(high|expensive|cheaper|too)\b|too expensive|anything cheaper"),
    ("objection_timing", r"not now|next (quarter|year|financial)|after the new year|no budget until|ping me in|reach out after"),
    ("not_interested", r"not interested|no thanks|already (use|have)|happy with|in-house"),
    ("question", r"\?"),
    ("interested", r"send (me )?more|sounds good|interested|exactly what we need|tell me more"),
]


REPLY_SYSTEM_PROMPT = """You classify a prospect's reply to a cold sales email into exactly ONE category.

Categories:
- meeting_request: proposes, asks for or agrees to a call, demo or meeting (even if they also ask for pricing or info).
- interested: positive and wants information (details, a case study, a one-pager) but does NOT ask for a call or demo.
- question: asks a specific factual question about the product and has not shown a decision either way.
- not_interested: declines, already has a solution or contract, or says it is not a priority.
- objection_price: the main point is that the price or cost is too high, or asks for a cheaper option.
- objection_timing: not now; wants to talk later (a date, quarter, "after X") without refusing.
- referral: says someone else is the right person and points to them.
- out_of_office: an automatic absence message.
- unsubscribe: asks to be removed or never contacted again.

Decide by the MAIN intent of the reply:
- A price complaint is objection_price even if it ends with a question mark.
- "No budget until <date>" or "come back in <time>" is objection_timing, not objection_price.
- Naming another person to talk to is referral, even if the writer also says they are not the right person.
- Interest plus a request for a call or demo is meeting_request.

Examples:
Reply: "Could you do Wednesday at 10 for a walkthrough?" -> meeting_request
Reply: "Send me the brochure and I'll have a look." -> interested
Reply: "Is the data encrypted at rest?" -> question
Reply: "We're all set with our current vendor, thanks." -> not_interested
Reply: "That is well beyond our budget for this year's tooling." -> objection_price
Reply: "Not a good moment, try me again in the spring." -> objection_timing
Reply: "Our COO Elena Park decides on this, email her." -> referral
Reply: "I am away from my desk until the 9th." -> out_of_office
Reply: "Remove my address from your database." -> unsubscribe

Give a one-sentence rationale first, then the category, a confidence from 0 to 1, and the next action for the sales rep."""


def _rules_reply(text: str) -> dict | None:
    if _UNSUB.search(text):
        return {"category": "unsubscribe", "confidence": 0.98, "method": "rules",
                "rationale": "explicit opt-out phrase", "next_action": "Suppress contact permanently; do not email again."}
    if _OOO.search(text):
        return {"category": "out_of_office", "confidence": 0.95, "method": "rules",
                "rationale": "auto-reply / absence phrase", "next_action": "Pause sequence; retry after return date."}
    return None


def _fallback_reply(text: str) -> dict:
    for cat, rx in _FALLBACK:
        if re.search(rx, text, re.I | re.S):
            return {"category": cat, "confidence": 0.5, "method": "rules-fallback",
                    "rationale": "keyword fallback (LLM unavailable)", "next_action": "Review manually."}
    return {"category": "question", "confidence": 0.2, "method": "rules-fallback",
            "rationale": "no rule matched (LLM unavailable)", "next_action": "Review manually."}


def classify_reply_rules(ctx: Ctx, payload: dict) -> dict:
    res = _rules_reply(payload["text"]) or _fallback_reply(payload["text"])
    return _save_reply(ctx, payload, res, "none")


def _save_reply(ctx: Ctx, payload: dict, res: dict, model: str) -> dict:
    ctx.store.x("INSERT INTO replies(prospect_id,text,category,confidence,rationale,next_action,method,model,"
                "expected,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (payload.get("prospect_id"), payload["text"], res["category"], res["confidence"],
                 res["rationale"], res["next_action"], res["method"], model, payload.get("expected"), time.time()))
    return res


def wf_reply(ctx: Ctx, payload: dict) -> dict:
    text = payload["text"]
    res = _rules_reply(text)
    if res:  # safety-critical categories are handled deterministically, no LLM needed
        return _save_reply(ctx, payload, res, "none")
    system = REPLY_SYSTEM_PROMPT
    data, rec = ctx.llm.generate_json("reply_classification", system, f"REPLY:\n{_clip(text, 1500)}", S_REPLY)
    res = {"category": data.get("category", "question"), "confidence": float(data.get("confidence") or 0),
           "rationale": str(data.get("rationale", "")), "next_action": str(data.get("next_action", "")),
           "method": "llm", "inference_id": rec["id"]}
    return _save_reply(ctx, payload, res, rec.get("model_served") or ctx.llm.model)


def evaluate_reply_classifier(ctx: Ctx, path: Path | None = None) -> dict:
    rows = json.loads(Path(path or DATA_DIR / "sample_replies.json").read_text(encoding="utf-8"))["replies"]
    results, correct = [], 0
    confusion: dict[str, dict[str, int]] = {}
    for r in rows:
        out = run_workflow(ctx, "reply", {"text": r["text"], "expected": r["expected"]})
        got = out.get("category")
        ok = got == r["expected"]
        correct += ok
        confusion.setdefault(r["expected"], {}).setdefault(str(got), 0)
        confusion[r["expected"]][str(got)] += 1
        results.append({"company": r["company"], "expected": r["expected"], "predicted": got,
                        "correct": ok, "method": out.get("method"), "status": out.get("status", "ok"),
                        "text": r["text"]})
    methods = {}
    for x in results:
        methods[x["method"]] = methods.get(x["method"], 0) + 1
    per_class = {c: f"{confusion[c].get(c, 0)}/{sum(confusion[c].values())}" for c in sorted(confusion)}
    return {"total": len(rows), "correct": correct, "accuracy": round(correct / len(rows), 3),
            "per_class_correct": per_class, "confusion": confusion, "methods_used": methods,
            "misclassified": [x for x in results if not x["correct"]], "results": results}


# ======================================================= dispatcher + queue
# workflow name -> (llm function, deterministic fallback or None)
WORKFLOWS = {
    "business_profile": (wf_business_profile, None),
    "icp": (wf_icp, None),
    "research": (wf_research, None),
    "signals": (wf_signals, wf_signals_rules),
    "problem": (wf_problem, None),
    "score": (wf_score, wf_score_rules),
    "email": (wf_email, None),
    "reply": (wf_reply, classify_reply_rules),
}


def run_workflow(ctx: Ctx, name: str, payload: dict) -> dict:
    """Run a workflow on local inference. On LocalAIUnavailable: use the deterministic
    fallback if one exists (and queue the LLM re-run), otherwise queue. Never calls a cloud API."""
    fn, fallback = WORKFLOWS[name]
    try:
        return {"status": "ok", **fn(ctx, payload)}
    except PreconditionError as e:
        return {"status": "blocked", "error": str(e)}
    except ValueError as e:  # unknown id, or the model never produced valid JSON
        return {"status": "failed", "error": str(e)}
    except LocalAIUnavailable as e:
        qid = ctx.store.enqueue(name, payload, str(e))
        if fallback:
            return {"status": "degraded", "queue_id": qid, "warning": str(e), **fallback(ctx, payload)}
        return {"status": "queued", "queue_id": qid, "error": str(e)}


def process_queue(ctx: Ctx) -> dict:
    """Re-run queued tasks once Ollama is back. Stops at the first connectivity failure."""
    health = ctx.llm.health()
    if not health["ok"]:
        return {"processed": 0, "remaining": len(ctx.store.queue("queued")), "error": health["error"]}
    done = failed = 0
    for t in ctx.store.queue("queued"):
        fn, _ = WORKFLOWS[t["workflow"]]
        try:
            fn(ctx, json.loads(t["payload_json"]))
            ctx.store.x("UPDATE task_queue SET status='done', done_at=?, attempts=attempts+1 WHERE id=?",
                        (time.time(), t["id"]))
            done += 1
        except LocalAIUnavailable as e:
            ctx.store.x("UPDATE task_queue SET error=?, attempts=attempts+1 WHERE id=?", (str(e)[:300], t["id"]))
            return {"processed": done, "failed": failed, "remaining": len(ctx.store.queue("queued")),
                    "error": str(e)}
        except Exception as e:  # bad model output etc.: keep it, record the reason
            ctx.store.x("UPDATE task_queue SET status='failed', error=?, attempts=attempts+1 WHERE id=?",
                        (str(e)[:300], t["id"]))
            failed += 1
    return {"processed": done, "failed": failed, "remaining": len(ctx.store.queue("queued"))}


# ================================================================ pipeline
def run_pipeline(ctx: Ctx, product_id: int, max_prospects: int = 6, email_tiers=("A", "B")) -> dict:
    """End-to-end: research -> signals -> problem -> score -> email for the top prospects."""
    prod = ctx.store.product(product_id)
    if not prod or not prod.get("icp"):
        return {"status": "blocked", "error": "Generate the product ICP before running the pipeline."}
    log = []
    prospects = ctx.store.prospects()[:max_prospects]
    if not prospects:
        return {"status": "blocked", "error": "No prospects loaded. Click \"Load company + prospect files\" first."}
    for pr in prospects:
        pl = {"prospect_id": pr["id"], "product_id": product_id}
        step = {"company": pr["company"]}
        step["research"] = run_workflow(ctx, "research", pl).get("status")
        step["signals"] = run_workflow(ctx, "signals", pl).get("status")
        step["problem"] = run_workflow(ctx, "problem", pl).get("status")
        sc = run_workflow(ctx, "score", pl)
        step["score"] = {"status": sc.get("status"), "total": sc.get("total"), "tier": sc.get("tier")}
        log.append(step)
    ranked = sorted(
        ctx.store.q("SELECT s.*, p.company FROM scores s JOIN prospects p ON p.id=s.prospect_id "
                    "WHERE s.product_id=? ORDER BY s.total DESC", (product_id,)), key=lambda r: -r["total"])
    emails = []
    for row in ranked:
        if row["tier"] in email_tiers:
            e = run_workflow(ctx, "email", {"prospect_id": row["prospect_id"], "product_id": product_id})
            emails.append({"company": row["company"], "status": e.get("status"),
                           "checks_passed": (e.get("checks") or {}).get("passed")})
    return {"steps": log, "ranking": [{"company": r["company"], "total": r["total"], "tier": r["tier"]}
                                      for r in ranked], "emails": emails}
