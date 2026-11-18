"""TEST-ONLY stand-in for an Ollama server (loopback HTTP). Returns schema-shaped canned
JSON so plumbing (queueing, grounding, scoring, logging) can be tested without a model.
It is NOT used by the product and its outputs are never presented as AI results."""
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer


def _between(text, start, end=None):
    i = text.find(start)
    if i < 0:
        return ""
    i += len(start)
    j = text.find(end, i) if end else len(text)
    return text[i : j if j > 0 else len(text)].strip()


def respond(wf, user):
    if wf == "business_profile":
        return {"company_summary": "Fictional data-resilience vendor.", "value_propositions": ["fast deploy"],
                "target_customers": ["regulated SMBs"], "differentiators": ["immutable backups"],
                "brand_voice": "calm, practical", "products": [{"name": "VaultGuard Backup", "one_liner": "backup"}]}
    if wf == "icp":
        return {"industries": ["Financial Services", "Healthcare", "Logistics"], "min_employees": 50,
                "max_employees": 1000, "buyer_titles": ["Head of IT", "CISO", "IT Manager"],
                "geographies": [], "pain_points": ["ransomware recovery", "backup audit findings"],
                "buying_triggers": ["security incident", "legacy backup replacement"],
                "disqualifiers": ["fully cloud native"], "messaging_angle": "prove you can recover"}
    if wf == "research":
        return {"summary": "Stored text summary.", "key_facts": ["fact"], "tech_environment": "unknown",
                "recent_events": [], "pain_hints": []}
    if wf == "signals":
        text = _between(user, "TEXT:\n", "\n\nRESEARCH SUMMARY")
        sents = re.split(r"(?<=[.!?])\s+", text)
        pick = next((s for s in sents if re.search(r"ransomware|backup|audit|hiring|funding", s, re.I)), "")
        sigs = [{"type": "security_incident", "strength": 4, "evidence": pick, "why_it_matters": "x"}] if pick else []
        sigs.append({"type": "funding", "strength": 5, "evidence": "raised a huge imaginary round", "why_it_matters": "hallucinated on purpose"}) if pick else None
        return {"signals": sigs}
    if wf == "problem_statement":
        return {"problem_statement": "They need provable recovery.", "evidence_used": []}
    if wf == "score_fit":
        return {"fit_score": 8, "reason": "Matches ICP."}
    if wf in ("email", "email_repair"):
        name = _between(user, "CONTACT: ", ",").split(" ")[0]
        body = (f"Hi {name}, I noticed your team recently dealt with an incident and is reviewing backups. "
                "We help IT teams prove that a restore actually works through one-click recovery drills, "
                "with immutable snapshots that stolen admin credentials cannot delete. Customers usually "
                "see first value within nine days and pricing is flat per server. Would a short call "
                "next week to compare approaches be useful?")
        return {"subject": "Proving restores actually work", "body": body}
    if wf == "reply_classification":
        t = user.lower()
        cat = "meeting_request" if "call" in t else "not_interested" if "not interested" in t else "interested"
        return {"category": cat, "confidence": 0.8, "rationale": "fake", "next_action": "fake"}
    return {}


class _H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj, code=200):
        raw = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path == "/api/tags":
            return self._send({"models": [{"name": self.server.model}]})
        self._send({}, 404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        sysmsg, user = body["messages"][0]["content"], body["messages"][1]["content"]
        wf = re.match(r"\[workflow:(\w+)\]", sysmsg).group(1)
        out = json.dumps(respond(wf, user))
        self._send({"model": body["model"], "message": {"role": "assistant", "content": out},
                    "prompt_eval_count": 120, "eval_count": 60, "eval_duration": 2_000_000_000})


def start(model="qwen3:1.7b", port=0):
    srv = HTTPServer(("127.0.0.1", port), _H)
    srv.model = model
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv
