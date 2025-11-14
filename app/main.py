from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Tuple
import os, json, re, asyncpg, httpx, uvicorn, ssl, asyncio
from dotenv import load_dotenv
import certifi

load_dotenv()

DB_URL = os.getenv("DATABASE_URL", "")
DB_SSLMODE = (os.getenv("DATABASE_SSLMODE", "") or "verify-full").strip().lower()
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_DECISION_MODEL = os.getenv("OPENAI_DECISION_MODEL", OPENAI_MODEL)
OPENAI_REPORT_MODEL = os.getenv("OPENAI_REPORT_MODEL", OPENAI_MODEL)
OPENAI_TIMEOUT = int(os.getenv("OPENAI_TIMEOUT", "60"))
OPENAI_MAX_TOTAL = int(os.getenv("OPENAI_MAX_TOTAL", "70"))
STATEMENT_TIMEOUT_MS = int(os.getenv("STATEMENT_TIMEOUT_MS", "2000"))
DB_CLIENT_MAX_MS = int(os.getenv("DB_CLIENT_MAX_MS", "5000"))
ALLOW_LLM = (os.getenv("ALLOW_LLM", "true").strip().lower() != "false")

app = FastAPI(title="Sales Analyst Orchestrator", version="4.1.0", default_response_class=JSONResponse)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

SQL_CACHE: Dict[str, str] = {}
SCHEMA_HINT: str = ""
ALLOWED_TABLES = {"leads", "contacts", "activities", "stages"}

class AnswerReq(BaseModel):
    chatInput: str
    needs_data: Optional[str] = "YES"
    sessionId: Optional[str] = None
    allow_llm: Optional[bool] = True

class ReportRequest(BaseModel):
    question: str
    sqlData: Optional[List] = None

class DecisionRequest(BaseModel):
    chatInput: str

def normalize(q: str) -> str:
    return re.sub(r"\s+", " ", q.lower().strip())

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
    ok = set(ALLOWED_TABLES)
    ctes = cte_names(sql or "")
    found = set()
    for m in re.finditer(r"(?is)\bfrom\s+([a-zA-Z_][\w.\"']*)|\bjoin\s+([a-zA-Z_][\w.\"']*)", sql or ""):
        for g in m.groups():
            if g:
                base = g.split(".")[-1].strip('"').lower()
                found.add(base)
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
    if not row or not (row["prompt"] or "").strip():
        raise HTTPException(500, f"Prompt missing: {prompt_type}")
    return row["prompt"]

async def load_schema_hint(pool: asyncpg.Pool) -> None:
    global SCHEMA_HINT
    async with pool.acquire() as c:
        rows = await c.fetch("""
            select table_name, column_name
            from information_schema.columns
            where table_schema = 'public'
              and table_name in ('leads','contacts','activities','stages')
            order by table_name, ordinal_position
        """)
    mp: Dict[str, List[str]] = {}
    for r in rows:
        t = r["table_name"]
        mp.setdefault(t, []).append(r["column_name"])
    lines = []
    for t in ("leads","contacts","activities","stages"):
        cols = mp.get(t, [])
        lines.append(f"- {t}: {', '.join(cols)}")
    join_rules = (
        "Join rules:\n"
        "- leads.stage_id = stages.id\n"
        "- leads.partner_id = contacts.id\n"
        "- leads.activity_id = activities.id\n"
        "Columns with spaces must be double-quoted."
    )
    SCHEMA_HINT = "Table/field hints:\n" + "\n".join(lines) + "\n\n" + join_rules

def build_ssl_context():
    mode = DB_SSLMODE
    if mode == "disable":
        return None
    if mode == "require":
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx
    if mode == "verify-ca":
        ctx = ssl.create_default_context(cafile=certifi.where())
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_REQUIRED
        return ctx
    ctx = ssl.create_default_context(cafile=certifi.where())
    ctx.check_hostname = True
    ctx.verify_mode = ssl.CERT_REQUIRED
    return ctx

openai_client = httpx.AsyncClient(
    timeout=httpx.Timeout(connect=10.0, read=OPENAI_TIMEOUT, write=30.0, pool=OPENAI_TIMEOUT),
    headers={"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"},
)

async def openai_chat(messages, model=None) -> str:
    if not ALLOW_LLM or not OPENAI_KEY:
        raise HTTPException(503, "LLM disabled or OPENAI_API_KEY missing")
    payload = {"model": model or OPENAI_MODEL, "temperature": 0, "messages": messages}
    try:
        async def _call():
            r = await openai_client.post("https://api.openai.com/v1/chat/completions", json=payload)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        return await asyncio.wait_for(_call(), timeout=OPENAI_MAX_TOTAL)
    except asyncio.TimeoutError:
        raise HTTPException(504, "OpenAI request timed out")
    except httpx.TimeoutException:
        raise HTTPException(504, "OpenAI client timeout")
    except httpx.HTTPStatusError as e:
        raise HTTPException(e.response.status_code, f"OpenAI error: {e.response.text[:300]}")
    except Exception as e:
        raise HTTPException(502, f"OpenAI call failed: {e}")

def dynamic_hint_for_question(q: str) -> str:
    ql = (q or "").lower()
    hints = []
    if "latest" in ql and ("last contact" in ql or "last touch" in ql):
        hints.append('Use activities."Last Contact Date" with a correlated MAX via leads.activity_id = activities.id.')
    if any(k in ql for k in ["missed", "inactive", "not touched", "no follow-up", "stale"]):
        hints.append('LEFT JOIN activities and treat NULL contact dates as missed.')
    if any(k in ql for k in ["target", "outreach", "follow up", "follow-up", "re-engage", "reengage", "re engage", "who next"]):
        hints.append('Detect overdue/NULL next-contact or last-contact older than 30 days, group by company, include lead details, order by days overdue.')
    return ("\n".join(hints)).strip()

def _inline_safe_date(expr: str) -> str:
    x = expr.strip()
    return f"""(CASE
  WHEN trim({x}) ~ '^[0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}$' THEN to_date(trim({x}),'YYYY-MM-DD')
  WHEN trim({x}) ~ '^[0-9]{{4}}/[0-9]{{2}}/[0-9]{{2}}$' THEN to_date(trim({x}),'YYYY/MM/DD')
  WHEN trim({x}) ~ '^[0-9]{{2}}-[0-9]{{2}}-[0-9]{{4}}$' THEN to_date(trim({x}),'DD-MM-YYYY')
  WHEN trim({x}) ~ '^[0-9]{{2}}/[0-9]{{2}}/[0-9]{{4}}$' THEN to_date(trim({x}),'DD/MM/YYYY')
  WHEN trim({x}) ~ '^[0-9]{{1,2}}-[0-9]{{1,2}}-[0-9]{{4}}$' THEN to_date(trim({x}),'FMDD-FMMM-YYYY')
  WHEN trim({x}) ~ '^[0-9]{{1,2}}/[0-9]{{1,2}}/[0-9]{{4}}$' THEN to_date(trim({x}),'FMDD/FMMM/YYYY')
  WHEN trim({x}) ~ '^[0-9]{{2}}-[0-9]{{2}}-[0-9]{{2}}$' THEN to_date(trim({x}),'YY-MM-DD')
  WHEN trim({x}) ~ '^[0-9]{{2}}/[0-9]{{2}}/[0-9]{{2}}$' THEN to_date(trim({x}),'YY/MM/DD')
  WHEN trim({x}) ~ '^[0-9]{{1,2}}-[0-9]{{1,2}}-[0-9]{{2}}$' THEN to_date(trim({x}),'FMDD-FMMM-YY')
  WHEN trim({x}) ~ '^[0-9]{{1,2}}/[0-9]{{1,2}}/[0-9]{{2}}$' THEN to_date(trim({x}),'FMDD/FMMM/YY')
  WHEN trim({x}) ~ '^[0-9]{{1,2}}-[0-9]{{1,2}}-[0-9]{{4}}$' THEN to_date(trim({x}),'FMMM-FMDD-YYYY')
  WHEN trim({x}) ~ '^[0-9]{{1,2}}/[0-9]{{1,2}}/[0-9]{{4}}$' THEN to_date(trim({x}),'FMMM/FMDD/YYYY')
  ELSE NULL
END)"""

_SAFE_DATE_CALL_RE = re.compile(r'\bSAFE_DATE\s*\(\s*([^)]+?)\s*\)', re.IGNORECASE | re.DOTALL)

def replace_safe_date_calls(sql: str) -> str:
    def _repl(m):
        inner = m.group(1)
        return _inline_safe_date(inner)
    return _SAFE_DATE_CALL_RE.sub(_repl, sql)

async def gpt_emit_sql(question: str, extra: str = "") -> str:
    system_prompt = await get_prompt(app.state.pool, "SYSTEM_PROMPT_SQL_AGENT")
    sys = {"role": "system", "content": system_prompt}
    dyn = dynamic_hint_for_question(question)
    user = {"role": "user", "content": f"{SCHEMA_HINT}\n\n{extra}\n{('- ' + dyn) if dyn else ''}\n\nUser question: {question}\nReturn ONLY the SQL."}
    out = await openai_chat([sys, user], model=OPENAI_MODEL)
    sql = sanitize_llm_sql(out)
    sql = replace_safe_date_calls(sql)
    return sql

async def repair_sql(question: str, sql: str, reason: str) -> str:
    system_prompt = await get_prompt(app.state.pool, "SYSTEM_PROMPT_SQL_AGENT")
    sys = {"role": "system", "content": system_prompt}
    user = {"role": "user", "content": f"{SCHEMA_HINT}\n\nIssue to fix:\n{reason}\n\nFix the SQL (same intent) and return ONLY the corrected SQL:\n{sql}"}
    out = await openai_chat([sys, user], model=OPENAI_MODEL)
    fixed = sanitize_llm_sql(out)
    fixed = replace_safe_date_calls(fixed)
    return fixed

def wrap_limit_one(sql: str) -> str:
    core = (sql or "").strip()
    if core.endswith(";"):
        core = core[:-1]
    return f"SELECT * FROM ({core}) __q LIMIT 1;"

async def dry_validate(conn, sql: str) -> None:
    await conn.execute(f"SET LOCAL statement_timeout = {STATEMENT_TIMEOUT_MS}")
    async def _call():
        await conn.fetch(wrap_limit_one(sql))
    await asyncio.wait_for(_call(), timeout=max(0.1, DB_CLIENT_MAX_MS/1000.0))

def guidance_from_db_error(err: str) -> str:
    g = []
    el = (err or "").lower()
    if "invalid input syntax for type numeric" in el and ("power" in el):
        g.append('Cast only leads."Power"::numeric; never cast stage_id or id.')
    if "column" in el and "does not exist" in el:
        g.append('Double-quote columns with spaces and use only allowed tables.')
    if "missing from-clause entry" in el:
        g.append('Do not reference a table alias outside its scope; include it in the FROM/JOIN.')
    if "date/time field value out of range" in el or "invalid input syntax for type date" in el:
        g.append('Guard any date comparison with a regex-guarded CASE to_date block.')
    return " ".join(g)

def needs_date_wrapping(sql: str) -> bool:
    for mo in re.finditer(r'"[^"]*date[^"]*"', sql or "", re.I):
        window = (sql[max(0, mo.start()-160):mo.end()+160] or "").lower()
        if any(op in window for op in [" between ", ">", "<", ">=", "<=", "="]):
            if "case" not in window or "to_date" not in window:
                return True
    return False

def scrub_illegal_numeric_casts(sql: str) -> str:
    sql = re.sub(r"\b(stage_id|id)\s*::\s*numeric", r"\1", sql, flags=re.I)
    return sql

async def get_exact_query(question: str) -> Tuple[Optional[str], Optional[List[Dict[str, Any]]]]:
    pool = app.state.pool
    async with pool.acquire() as conn:
        await conn.execute(f"SET LOCAL statement_timeout = {STATEMENT_TIMEOUT_MS}")
        async def _call():
            row = await conn.fetchrow("SELECT sql FROM common_questions WHERE question = $1", question)
            if not row:
                return None, None
            sql = replace_safe_date_calls(sanitize_llm_sql(row["sql"]))
            is_select_only(sql); tables_allowed(sql)
            rows = await conn.fetch(sql)
            return sql, [dict(r) for r in rows]
        return await asyncio.wait_for(_call(), timeout=max(0.1, DB_CLIENT_MAX_MS/1000.0))

def pgbouncer_pool_kwargs() -> Dict[str, Any]:
    return {
        "statement_cache_size": 0,
        "max_cached_statement_lifetime": 0,
        "command_timeout": max(1, int(DB_CLIENT_MAX_MS/1000)),
    }

@app.on_event("startup")
async def startup():
    if not DB_URL:
        raise RuntimeError("DATABASE_URL is not set.")
    if not OPENAI_KEY and ALLOW_LLM:
        raise RuntimeError("OPENAI_API_KEY is not set (set ALLOW_LLM=false to run without LLM).")
    ssl_ctx = build_ssl_context()
    app.state.pool = await asyncpg.create_pool(
        dsn=DB_URL, min_size=1, max_size=8, ssl=ssl_ctx, **pgbouncer_pool_kwargs(),
    )
    await load_schema_hint(app.state.pool)

@app.on_event("shutdown")
async def shutdown():
    if getattr(app.state, "pool", None):
        await app.state.pool.close()
    try:
        await openai_client.aclose()
    except Exception:
        pass

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

@app.get("/diag")
async def diag(check: Optional[str] = None):
    info: Dict[str, Any] = {"env": {
        "ALLOW_LLM": ALLOW_LLM,
        "OPENAI_MODEL": OPENAI_MODEL,
        "OPENAI_TIMEOUT": OPENAI_TIMEOUT,
        "OPENAI_MAX_TOTAL": OPENAI_MAX_TOTAL,
        "STATEMENT_TIMEOUT_MS": STATEMENT_TIMEOUT_MS,
        "DB_CLIENT_MAX_MS": DB_CLIENT_MAX_MS,
        "DB_SSLMODE": DB_SSLMODE
    }}
    if check in (None, "db", "all"):
        try:
            async with app.state.pool.acquire() as conn:
                await conn.execute(f"SET LOCAL statement_timeout = {STATEMENT_TIMEOUT_MS}")
                await asyncio.wait_for(conn.fetchval("SELECT 1;"), timeout=max(0.1, DB_CLIENT_MAX_MS/1000.0))
            info["db"] = "ok"
        except Exception as e:
            info["db"] = f"error: {e}"
    if check in ("openai", "all"):
        try:
            if not ALLOW_LLM:
                info["openai"] = "disabled"
            else:
                payload = {"model": OPENAI_MODEL, "temperature": 0,
                           "messages": [{"role":"system","content":"ping"},{"role":"user","content":"pong"}]}
                async def _call():
                    r = await openai_client.post("https://api.openai.com/v1/chat/completions", json=payload)
                    r.raise_for_status()
                await asyncio.wait_for(_call(), timeout=OPENAI_MAX_TOTAL)
                info["openai"] = "ok"
        except Exception as e:
            info["openai"] = f"error: {e}"
    return info

@app.post("/decide")
async def decide(data: DecisionRequest):
    system_prompt = await get_prompt(app.state.pool, "SYSTEM_PROMPT_QUERY_DECISION_AGENT")
    if not ALLOW_LLM:
        q = (data.chatInput or "").lower()
        is_data = any(k in q for k in
            ["show","list","which","top","count","how many","where","filter","find","leads","contacts","activities","stages"])
        return {"is_data_query": bool(is_data)}
    content = await openai_chat(
        [{"role":"system","content":system_prompt},{"role":"user","content":str(data)}],
        model=OPENAI_DECISION_MODEL
    )
    return json.loads(content)

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
            if not (body.allow_llm if body.allow_llm is not None else ALLOW_LLM):
                raise HTTPException(400, "No deterministic match and LLM disabled")
            sql = await gpt_emit_sql(body.chatInput)

    sql = replace_safe_date_calls(sql)
    is_select_only(sql); tables_allowed(sql)

    critic_prompt = await get_prompt(app.state.pool, "SYSTEM_PROMPT_SQL_CRITIC")
    critic_input = [
        {"role": "system", "content": critic_prompt},
        {"role": "user", "content": json.dumps({"question": body.chatInput, "sql": sql})}
    ]
    critic_raw = await openai_chat(critic_input, model=OPENAI_MODEL)
    try:
        critic = json.loads(critic_raw)
    except:
        critic = {"ok": True}
    if not critic.get("ok", True):
        fix_reason = "\n".join(critic.get("must_fix", []))
        sql = await repair_sql(body.chatInput, sql, fix_reason)
        sql = replace_safe_date_calls(sql)
        is_select_only(sql); tables_allowed(sql)

    if needs_date_wrapping(sql):
        sql = await repair_sql(body.chatInput, sql, "Wrap any date comparison with the inline CASE regex-guarded to_date block.")
        sql = replace_safe_date_calls(sql)
        is_select_only(sql); tables_allowed(sql)

    if (body.needs_data or "YES").upper() == "NO":
        SQL_CACHE[qn] = sql
        return {"session_id": body.sessionId or "none", "needs_data": "NO",
                "question": body.chatInput.strip(), "sql": sql, "data": []}

    try:
        async with pool.acquire() as conn:
            try:
                await dry_validate(conn, sql)
            except Exception as ve:
                guidance = guidance_from_db_error(str(ve)) or ""
                fixed = await repair_sql(body.chatInput, sql, f"DB error: {ve}. {guidance}")
                fixed = replace_safe_date_calls(fixed)
                is_select_only(fixed); tables_allowed(fixed)
                if needs_date_wrapping(fixed):
                    fixed = await repair_sql(body.chatInput, fixed, "All date comparisons must use the inline CASE regex-guarded to_date block.")
                    fixed = replace_safe_date_calls(fixed)
                    is_select_only(fixed); tables_allowed(fixed)
                sql = scrub_illegal_numeric_casts(fixed)

            await conn.execute(f"SET LOCAL statement_timeout = {STATEMENT_TIMEOUT_MS}")
            async def _run():
                rows = await conn.fetch(sql)
                return [dict(r) for r in rows]
            data = await asyncio.wait_for(_run(), timeout=max(0.1, DB_CLIENT_MAX_MS/1000.0))

        SQL_CACHE[qn] = sql
        return {"session_id": body.sessionId or "none",
                "needs_data": (body.needs_data or "YES").upper(),
                "question": body.chatInput.strip(),
                "sql": sql,
                "data": data}

    except asyncio.TimeoutError:
        return {"session_id": body.sessionId or "none",
                "error": f"DB client timeout after {DB_CLIENT_MAX_MS} ms",
                "sql": sql}
    except Exception as e:
        return {"session_id": body.sessionId or "none", "error": f"DB error: {e}", "sql": sql}

@app.post("/report")
async def generate_report(data: ReportRequest):
    system_prompt = await get_prompt(app.state.pool, "SYSTEM_PROMPT_REPORT_AGENT")
    if not ALLOW_LLM:
        tbl = data.sqlData or []
        return {"report": f"Data rows: {len(tbl)}"}
    content = await openai_chat(
        [{"role":"system","content":system_prompt},{"role":"user","content":str(data)}],
        model=OPENAI_REPORT_MODEL
    )
    return {"report": content}

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
    port = int(os.getenv("PORT", "10000"))
    uvicorn.run("newmain:app", host="0.0.0.0", port=port, log_level="info", reload=False)
