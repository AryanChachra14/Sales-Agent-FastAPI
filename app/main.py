from fastapi import FastAPI, HTTPException, Request, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os, json, re, asyncpg, httpx
from dotenv import load_dotenv
import httpx
import difflib
import uvicorn

load_dotenv()
DB_URL_MAIN = os.getenv("DATABASE_URL_MAIN", "")
DB_URL_QUERIES = os.getenv("DATABASE_URL_QUERIES", "")
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TIMEOUT = int(os.getenv("OPENAI_TIMEOUT", "60"))
STATEMENT_TIMEOUT_MS = int(os.getenv("STATEMENT_TIMEOUT_MS", "2000"))

app = FastAPI(title="Sales Analyst Orchestrator", version="2.4.0", default_response_class=JSONResponse)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SQL_CACHE: dict[str, str] = {}
SCHEMA_MAP: dict[str, list[str]] = {}
SCHEMA_HINT: str = ""
FIELD_INVERT: dict[str, set[str]] = {}

class AnswerReq(BaseModel):
    chatInput: str
    needs_data: str | None = "YES"
    sessionId: str | None = None
    allow_llm: bool | None = True

class ReportRequest(BaseModel):
    question: str
    sqlData: Optional[List] = None

class DecisionRequest(BaseModel):
    chatInput: str

def normalize(q: str) -> str:
    return re.sub(r"\s+", " ", q.lower().strip())

def normkey(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())

def cte_names(sql: str) -> set[str]:
    names = set()
    for m in re.finditer(r"(?is)\bwith\s+([a-z_]\w*)\s+as\s*\(", sql or ""):
        names.add(m.group(1))
    for m in re.finditer(r"(?is),\s*([a-z_]\w*)\s+as\s*\(", sql or ""):
        names.add(m.group(1))
    return names

def is_select_only(sql: str) -> None:
    s = (sql or "").strip()
    if not re.match(r"(?is)^\s*(select|with)\b", s):
        raise HTTPException(400, "Only SELECT / WITH allowed")
    if ";" in s[:-1]:
        raise HTTPException(400, "Multiple statements not allowed")
    if re.search(r"(?is)\b(update|insert|delete|alter|drop|truncate|create|grant|revoke|call|do|copy)\b", s):
        raise HTTPException(400, "Forbidden keyword detected")

def tables_allowed(sql: str) -> None:
    ok = {"document_metadata", "document_rows", "documents", "memories"}
    ctes = cte_names(sql or "")
    found = set()
    for m in re.finditer(r"(?is)\bfrom\s+([a-zA-Z_][\w.]*)|\bjoin\s+([a-zA-Z_][\w.]*)", sql or ""):
        for g in m.groups():
            if g:
                found.add(g.split(".")[-1])
    bad = sorted(x for x in (found - ok) if x not in ctes)
    if bad:
        raise HTTPException(400, f"Unexpected tables: {', '.join(bad)}")

def sanitize_llm_sql(text: str) -> str:
    s = (text or "").strip()
    s = re.sub(r'(?is)^.*?\b(select|with)\b', r'\1', s, count=1).strip()
    s = s.split(';')[0].strip()
    if s and not s.endswith(';'):
        s += ';'
    return s

async def get_prompt(pool, prompt_type: str) -> str:
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT prompt FROM prompts WHERE title = $1", prompt_type)
    if not row:
        raise HTTPException(500, f"Prompt for type {prompt_type} not found in DB")
    return row["prompt"]

async def load_schema_cache(pool: asyncpg.Pool) -> None:
    global SCHEMA_MAP, FIELD_INDEX, SCHEMA_HINT
    async with pool.acquire() as c:
        rows = await c.fetch("SELECT title, schema FROM document_metadata;")
    mp: dict[str, list[str]] = {}
    for r in rows:
        title = (r["title"] or "").strip().lower()
        raw = r["schema"]
        cols: list[str] = []
        js = None
        try:
            js = json.loads(raw) if raw else None
        except:
            js = None
        if isinstance(js, dict):
            if isinstance(js.get("columns"), list):
                for x in js["columns"]:
                    if isinstance(x, dict) and "name" in x: cols.append(str(x["name"]))
                    elif isinstance(x, str): cols.append(x)
            if not cols and isinstance(js.get("fields"), list):
                for x in js["fields"]:
                    if isinstance(x, str): cols.append(x)
            if not cols and isinstance(js.get("properties"), dict):
                cols = list(js["properties"].keys())
        elif isinstance(js, list):
            cols = [str(x) for x in js if isinstance(x, str)]
        cols = [c for c in cols if isinstance(c, str)]
        if title:
            mp[title] = cols
    SCHEMA_MAP = mp

    finx: dict[str, dict[str, str]] = {}
    for ds, cols in SCHEMA_MAP.items():
        m = {}
        for col in cols:
            m[normkey(col)] = col
        finx[ds] = m
    FIELD_INDEX = finx

    lines = []
    for ds, cols in sorted(SCHEMA_MAP.items()):
        if cols:
            lines.append(f"- {ds}: {', '.join(cols[:40])}")
    join_rules = (
        "Join rules:\n"
        "- leads.stage_id -> stages.id\n"
        "- leads.partner_id -> contacts.id\n"
        "- leads.activity_id -> activities.id\n"
        "Never use activities.activity_id. Never use created_at unless present."
    )
    SCHEMA_HINT = "Dataset field hints:\n" + ("\n".join(lines[:60]) if lines else "(no schema fields discovered)") + "\n\n" + join_rules

def extract_alias_dataset_map(sql: str) -> dict[str, str]:
    m: dict[str, str] = {}
    for mo in re.finditer(r"(?is)\bjoin\s+document_metadata\s+(?:as\s+)?([a-zA-Z_]\w*)\s+on\s+([a-zA-Z_]\w*)\.dataset_id\s*=\s*\1\.id\s+and\s+\1\.title\s*=\s*'([^']+)'", sql or ""):
        m[mo.group(2)] = mo.group(3).lower()
    for mo in re.finditer(r"(?is)\b([a-zA-Z_]\w*)\.dataset_id\s*=\s*\(\s*select\s+id\s+from\s+document_metadata\s+where\s+title\s*=\s*'([^']+)'\s*\)", sql or ""):
        m.setdefault(mo.group(1), mo.group(2).lower())
    for mo in re.finditer(r"(?is)\b([a-zA-Z_]\w*)\.dataset_id\s+in\s*\(\s*select\s+id\s+from\s+document_metadata\s+where\s+title\s+in\s*\(([^)]+)\)\s*\)", sql or ""):
        titles = [t.strip().strip("'").lower() for t in mo.group(2).split(",") if t.strip()]
        if titles:
            m.setdefault(mo.group(1), titles[0])
    return m

def extract_alias_field_uses(sql: str) -> list[tuple[str, str]]:
    uses = []
    for mo in re.finditer(r"(?is)\b([a-zA-Z_]\w*)\s*\.\s*row_data\s*->>\s*'([^']+)'", sql or ""):
        uses.append((mo.group(1), mo.group(2)))
    return uses

def aliases_missing_scoping(sql: str) -> set[str]:
    alias2ds = extract_alias_dataset_map(sql)
    need = set()
    for alias, _field in extract_alias_field_uses(sql):
        if alias not in alias2ds:
            need.add(alias)
    return need

def needs_date_wrapping(sql: str) -> bool:
    for mo in re.finditer(r"row_data->>'([^']*date[^']*)'", sql, re.I):
        window = (sql[max(0, mo.start()-50):mo.end()+50] or "").lower()
        if any(op in window for op in [" between ", ">", "<", ">=", "<=", "="]):
            if "to_date(" not in window and "coalesce(" not in window:
                return True
    return False

def auto_fix_fields(sql: str) -> str:
    alias2ds = extract_alias_dataset_map(sql)
    changed = sql
    for alias, field in extract_alias_field_uses(sql):
        ds = alias2ds.get(alias)
        if not ds:
            continue
        cols = SCHEMA_MAP.get(ds, [])
        if field not in cols and cols:
            target = FIELD_INDEX.get(ds, {}).get(normkey(field))
            if not target:
                cand = difflib.get_close_matches(field, cols, n=1, cutoff=0.8)
                target = cand[0] if cand else None
            if target:
                changed = re.sub(
                    rf"({re.escape(alias)}\s*\.\s*row_data\s*->>\s*')({re.escape(field)})(')",
                    rf"\1{target}\3",
                    changed
                )
    return changed

def scrub_illegal_numeric_casts(sql: str) -> str:
    sql = re.sub(r"\(\s*([a-zA-Z_]\w*)\s*\.\s*row_data\s*->>\s*'stage_id'\s*\)\s*::\s*numeric", r"(\1.row_data->>'stage_id')", sql, flags=re.I)
    sql = re.sub(r"\(\s*([a-zA-Z_]\w*)\s*\.\s*row_data\s*->>\s*'id'\s*\)\s*::\s*numeric", r"(\1.row_data->>'id')", sql, flags=re.I)
    return sql

def simplify_stage_literal(sql: str) -> str:
    return re.sub(
        r"(\bleads\s*\.\s*row_data\s*->>\s*'stage_id'\s*)=\s*\(\s*select\s+[^)]*?where\s+[^)]*?stages\s*\.\s*row_data\s*->>\s*'(?:name|id)'\s*=\s*'([^']+)'\s*\)",
        r"\1= '\2'",
        sql,
        flags=re.I | re.S
    )

def validate_fields_against_schema(sql: str) -> None:
    alias2ds = extract_alias_dataset_map(sql)
    uses = extract_alias_field_uses(sql)
    unknown: list[tuple[str, str, str]] = []
    for alias, field in uses:
        ds = alias2ds.get(alias)
        if not ds:
            continue
        cols = SCHEMA_MAP.get(ds, [])
        if field not in cols:
            unknown.append((alias, ds, field))
    if unknown:
        msgs = []
        for alias, ds, field in unknown[:16]:
            have = ", ".join(SCHEMA_MAP.get(ds, [])[:30]) or "(no fields discovered)"
            msgs.append(f"{alias} -> dataset '{ds}', field '{field}' not in schema. Available: {have}")
        raise HTTPException(400, "Schema field mismatch: " + " | ".join(msgs))

async def openai_chat(messages):
    if not OPENAI_KEY:
        raise HTTPException(400, "OPENAI_API_KEY missing")
    payload = {"model": OPENAI_MODEL, "temperature": 0, "messages": messages}
    async with httpx.AsyncClient(timeout=OPENAI_TIMEOUT) as cx:
        r = await cx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"},
            json=payload,
        )
        r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def dynamic_hint_for_question(q: str) -> str:
    ql = (q or "").lower()
    hints = []
    if "latest" in ql and "last contact" in ql:
        hints.append("Use activities.'Last Contact Date' and join via leads.activity_id = activities.id; return activities for each specified lead with MAX(Last Contact Date).")
    return ("\n".join(hints)).strip()

async def gpt_emit_sql(question: str, extra: str = "") -> str:
    system_prompt = await get_prompt(app.state.pool, "SYSTEM_PROMPT_SQL_AGENT")
    sys = {"role": "system", "content": system_prompt }
    dyn = dynamic_hint_for_question(question)
    user = {"role": "user", "content": f"{SCHEMA_HINT}\n\n{extra}\n{('- ' + dyn) if dyn else ''}\n\nUser question: {question}\nReturn ONLY the SQL."}
    out = await openai_chat([sys, user])
    return sanitize_llm_sql(out)

async def repair_sql(question: str, sql: str, reason: str) -> str:
    system_prompt = await get_prompt(app.state.pool, "SYSTEM_PROMPT_SQL_AGENT")
    sys = {"role": "system", "content": system_prompt }
    user = {"role": "user", "content": f"{SCHEMA_HINT}\n\nIssue to fix:\n{reason}\n\nFix the SQL (same intent) and return ONLY the corrected SQL:\n{sql}"}
    out = await openai_chat([sys, user])
    return sanitize_llm_sql(out)

def wrap_limit_one(sql: str) -> str:
    core = (sql or "").strip()
    if core.endswith(";"):
        core = core[:-1]
    return f"SELECT * FROM ({core}) __q LIMIT 1;"

async def dry_validate(conn, sql: str) -> None:
    await conn.execute(f"SET LOCAL statement_timeout = {STATEMENT_TIMEOUT_MS}")
    await conn.fetch(wrap_limit_one(sql))

def guidance_from_schema_mismatch(detail: str) -> str:
    g = []
    if re.search(r"activities[^,]*field 'activity_id'", detail, re.I):
        g.append("Do NOT use activities.row_data->>'activity_id'. Join using leads.row_data->>'activity_id' = activities.row_data->>'id'.")
    if re.search(r"leads[^,]*field 'created_at'", detail, re.I):
        g.append("Do NOT use leads.row_data->>'created_at' (it is not in schema). Use available date fields only.")
    return " ".join(g)

def guidance_from_db_error(err: str) -> str:
    g = []
    if "invalid input syntax for type numeric" in (err or "").lower():
        if "Opportunity" in err or "opportunity" in err:
            g.append("Do NOT cast 'Opportunity' to numeric; treat it as text. Use a numeric like Power::numeric or COUNT/COUNT DISTINCT as appropriate.")
    return " ".join(g)

async def gpt_sql_with_repair(question: str) -> str:
    sql = await gpt_emit_sql(question)
    if not sql:
        raise HTTPException(400, "No SQL produced")
    is_select_only(sql)
    tables_allowed(sql)
    if aliases_missing_scoping(sql):
        sql = await repair_sql(question, sql, "Add dataset scoping (via document_metadata.title) for all aliases touching row_data.")
        is_select_only(sql); tables_allowed(sql)
    if needs_date_wrapping(sql):
        sql = await repair_sql(question, sql, "Wrap ALL date comparisons with the COALESCE(to_date(...)) parser specified in the rules.")
        is_select_only(sql); tables_allowed(sql)
    sql = scrub_illegal_numeric_casts(sql)
    sql = simplify_stage_literal(sql)
    sql = auto_fix_fields(sql)
    validate_fields_against_schema(sql)
    return sql

async def repair_from_error(question: str, sql: str, error_msg: str) -> str:
    guidance = guidance_from_db_error(error_msg)
    if not guidance:
        guidance = "Keep the same intent and joins; fix syntax/aliases/date parsing only. Never cast stage_id/id to numeric."
    fixed = await repair_sql(question, sql, f"DB error: {error_msg}. {guidance}")
    is_select_only(fixed); tables_allowed(fixed)
    if aliases_missing_scoping(fixed):
        fixed = await repair_sql(question, fixed, "Add dataset scoping for all aliases touching row_data.")
        is_select_only(fixed); tables_allowed(fixed)
    if needs_date_wrapping(fixed):
        fixed = await repair_sql(question, fixed, "Ensure ALL date comparisons use COALESCE(to_date(...)).")
        is_select_only(fixed); tables_allowed(fixed)
    fixed = scrub_illegal_numeric_casts(fixed)
    fixed = simplify_stage_literal(fixed)
    fixed = auto_fix_fields(fixed)
    validate_fields_against_schema(fixed)
    return fixed

async def repair_from_schema_mismatch(question: str, sql: str, mismatch_detail: str) -> str:
    extra = guidance_from_schema_mismatch(mismatch_detail)
    reason = f"Schema field mismatch: {mismatch_detail}. {extra} Use only fields present in each dataset's schema (see hints)."
    fixed = await repair_sql(question, sql, reason)
    is_select_only(fixed); tables_allowed(fixed)
    if aliases_missing_scoping(fixed):
        fixed = await repair_sql(question, fixed, "Add dataset scoping for all aliases touching row_data.")
        is_select_only(fixed); tables_allowed(fixed)
    if needs_date_wrapping(fixed):
        fixed = await repair_sql(question, fixed, "Ensure ALL date comparisons use COALESCE(to_date(...)).")
        is_select_only(fixed); tables_allowed(fixed)
    fixed = scrub_illegal_numeric_casts(fixed)
    fixed = simplify_stage_literal(fixed)
    fixed = auto_fix_fields(fixed)
    validate_fields_against_schema(fixed)
    return fixed

async def get_exact_query(question: str) -> str | None:
    pool = app.state.pool
    async with app.state.pool_queries.acquire() as conn:
        row = await conn.fetchrow("SELECT sql FROM common_questions WHERE question = $1", question)
        if not row:
            return None, None
        sql = row["sql"]
        is_select_only(sql)
        tables_allowed(sql)
        await pool.execute(f"SET LOCAL statement_timeout = {STATEMENT_TIMEOUT_MS}")
        rows = await pool.fetch(sql)
    data = [dict(r) for r in rows]
    return sql, data

@app.on_event("startup")
async def startup():
    app.state.pool = await asyncpg.create_pool(dsn=DB_URL_MAIN, min_size=1, max_size=8, ssl='require')
    app.state.pool_queries = await asyncpg.create_pool(dsn=DB_URL_QUERIES, min_size=1, max_size=4, ssl='require')
    await load_schema_cache(app.state.pool)

@app.on_event("shutdown")
async def shutdown():
    if app.state.pool:
        await app.state.pool.close()
    if app.state.pool_queries:
        await app.state.pool_queries.close()

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

@app.post("/decide")
async def decide(data: DecisionRequest):
    url = "https://api.openai.com/v1/chat/completions"
    system_prompt = await get_prompt(app.state.pool, "SYSTEM_PROMPT_QUERY_DECISION_AGENT")
    payload = {
        "model": "gpt-4.1",
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": str(data)}
        ]
    }
    headers = {
        "Authorization": f"Bearer {OPENAI_KEY}",
        "Content-Type": "application/json"
    }
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        decision_data = json.loads(content)
    return decision_data 

@app.post("/answer")
async def answer(body: AnswerReq):
    pool = app.state.pool
    qn = normalize(body.chatInput)
    if qn in SQL_CACHE:
        sql = SQL_CACHE[qn]
    else:
        if re.match(r"(?is)^\s*(select|with)\b", body.chatInput.strip()):
            sql = sanitize_llm_sql(body.chatInput.strip())
        else:
            if not body.allow_llm:
                raise HTTPException(400, "No deterministic match and LLM disabled")
            sql = await gpt_sql_with_repair(body.chatInput)
    is_select_only(sql)
    tables_allowed(sql)
    if body.needs_data and (body.needs_data or "").upper() == "NO":
        SQL_CACHE[qn] = sql
        return {"session_id": body.sessionId or "none", "needs_data": "NO", "question": body.chatInput.strip(), "sql": sql, "data": []}
    try:
        async with pool.acquire() as conn:
            try:
                await dry_validate(conn, sql)
            except Exception as ve:
                repaired = await repair_from_error(body.chatInput, sql, str(ve))
                sql = repaired
            await conn.execute(f"SET LOCAL statement_timeout = {STATEMENT_TIMEOUT_MS}")
            rows = await conn.fetch(sql)
        data = [dict(r) for r in rows]
        SQL_CACHE[qn] = sql
        return {"session_id": body.sessionId or "none", "needs_data": (body.needs_data or "YES").upper(), "question": body.chatInput.strip(), "sql": sql, "data": data}
    except Exception as e:
        return {"session_id": body.sessionId or "none", "error": f"DB error: {e}", "sql": sql}

@app.post("/report")
async def generate_report(data: ReportRequest):
    url = "https://api.openai.com/v1/chat/completions"
    system_prompt = await get_prompt(app.state.pool, "SYSTEM_PROMPT_REPORT_AGENT")
    payload = {
        "model": "gpt-4.1",
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": str(data)}
        ]
    }
    headers = {
        "Authorization": f"Bearer {OPENAI_KEY}",
        "Content-Type": "application/json"
    }
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
    return {"report": result["choices"][0]["message"]["content"]}

@app.post("/answer-report")
async def answer_report(decide_req: DecisionRequest):
    decide_response = await decide(decide_req)
    if decide_response.get("is_data_query") == False:
        report_req = ReportRequest(question=decide_req.chatInput, sqlData=None)
        report_response = await generate_report(report_req)
        return {
            "report": report_response.get("report")
        }
    elif decide_response.get("is_data_query") == True:
        sql, data = await get_exact_query(decide_req.chatInput)
        if sql is not None:
            report_req = ReportRequest(question=decide_req.chatInput, sqlData=data)
            report_response = await generate_report(report_req)
            return {
                "report": report_response.get("report")
            }
        else:
            answer_req = AnswerReq(chatInput=decide_req.chatInput, needs_data=str(decide_response.get("is_data_query")), allow_llm=True)
            answer_response = await answer(answer_req)
            if "error" in answer_response:
                return answer_response
            report_req = ReportRequest(
                question=answer_req.chatInput,
                sqlData=answer_response.get("data", [])
            )
            report_response = await generate_report(report_req)
            return{
                "answer": answer_response,
                "report": report_response.get("report")
            }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info", reload=False)
