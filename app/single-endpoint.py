from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os, json, re, asyncpg, httpx
from dotenv import load_dotenv


load_dotenv()
DB_URL = os.getenv("DATABASE_URL", "").replace("postgresql+psycopg2", "postgresql")
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
STATEMENT_TIMEOUT_MS = int(os.getenv("STATEMENT_TIMEOUT_MS", "2000"))

SYSTEM_PROMPT_SQL_AGENT = """
You convert natural language into one precise PostgreSQL query for this schema:

document_metadata(id TEXT PRIMARY KEY, title TEXT, url TEXT, created_at TIMESTAMP, schema TEXT)
document_rows(id INT PRIMARY KEY, dataset_id TEXT REFERENCES document_metadata(id), row_data JSONB)
documents(id BIGINT PRIMARY KEY, content TEXT, metadata JSONB, embedding USER-DEFINED)
memories(id BIGINT PRIMARY KEY, content TEXT, metadata JSONB, embedding USER-DEFINED)
messages(id INT PRIMARY KEY, session_id VARCHAR NOT NULL, message JSONB NOT NULL)

Rules:
- Output a single executable query only. Start with SELECT or WITH. No prose.
- Use only these physical tables: document_rows, document_metadata, documents, memories.
- Datasets (leads, activities, contacts, stages, users) are identified via document_metadata.title; never use them as table names.
- Resolve dataset_id via (SELECT id FROM document_metadata WHERE title = '<dataset>') or equivalent IN form. Do not hardcode dataset ids.
- Access JSON with row_data->>'Exact Field' using exact case from each dataset schema in document_metadata.schema.
- Do not add filters/group/limit beyond the user request.
- Use to_date(text,'MM-DD-YYYY') or to_date(text,'MM/DD/YYYY') as appropriate when comparing dates.
- For “highest/maximum/top”, use MAX(...) with a correlated comparison or subquery to return only rows matching that maximum.
- For “latest/most recent/last per entity”, use a correlated subquery comparing each row’s date to MAX(date) per entity; do not rely only on ORDER BY.
- For inequalities (less/greater/below/above), cast text numbers to numeric: (row_data->>'Field')::numeric.
- When joining metadata or helper tables, always SELECT only the relevant data fields (e.g., from 'leads.row_data', 'contacts.row_data', etc.), never use SELECT *.Do not return metadata fields like schema, title, or created_at.
- For stage relations, join leads to stages on leads.row_data->>'stage_id' = stages.row_data->>'id' using document_rows/document_metadata only.
- For partner relations, join contacts to leads via contacts.row_data->>'id' = leads.row_data->>'partner_id'.
- Deterministic: same input → same SQL.
Return only the SQL.
"""

SYSTEM_PROMPT_REPORT_AGENT = """
You are an assistant that produces a concise, structured inference (maximum 250 characters) along with a complete data table based on the corpus of documents and/or database query results provided.

Key Instructions:
- When database query results are given in JSON, treat them as the authoritative data source.
- Present all data in a Markdown table before the inference.
- Use these results to generate a brief, focused inference.
- Do not create or execute SQL queries or fetch data yourself.
- Avoid hallucinating or inferring facts not present in the data.
- Highlight missing details or limitations only if relevant to the inference.
- Format the inference in Markdown.
- Use headers (###) only if needed for clarity.
- Use inline formatting (italic or bold) sparingly for emphasis.
- Use code blocks (```) only if a JSON snippet or formula is essential.
- If no data is provided but the question can be answered from documents, generate a concise, well-reasoned inference without mentioning data absence.
- Maintain a pragmatic, professional, and analytical tone.
- Prioritize clarity and precision over length.
"""


app = FastAPI(title="RAG + Report Orchestrator", version="1.0.0", default_response_class=JSONResponse)

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

def normalize(q: str) -> str:
    return re.sub(r"\s+", " ", q.lower().strip())

def is_select_only(sql: str) -> None:
    s = sql.strip()
    if not re.match(r"(?is)^\s*(select|with)\b", s):
        raise HTTPException(400, "Only SELECT statements allowed")
    if ";" in s[:-1]:
        raise HTTPException(400, "Multiple statements not allowed")
    if re.search(r"\b(update|insert|delete|alter|drop|truncate|create|grant|revoke|call|do|copy)\b", s, re.I):
        raise HTTPException(400, "Forbidden keyword")

def tables_allowed(sql: str) -> None:
    ok = {"document_metadata", "document_rows", "documents", "memories"}
    found = {t.split(".")[-1] for p in re.findall(r"(?is)\bfrom\s+([a-zA-Z0-9_.]+)|\bjoin\s+([a-zA-Z0-9_.]+)", sql) for t in p if t}
    bad = sorted(found - ok)
    if bad:
        raise HTTPException(400, f"Unexpected tables: {', '.join(bad)}")

def sanitize_llm_sql(text: str) -> str:
    s = text.strip()
    s = re.sub(r'(?is)^.*?\b(select|with)\b', r'\1', s, count=1).strip()
    s = s.split(';')[0].strip()
    if s and not s.endswith(';'):
        s += ';'
    return s

async def load_schema_cache(pool: asyncpg.Pool) -> None:
    global SCHEMA_MAP, SCHEMA_HINT, FIELD_INVERT
    async with pool.acquire() as c:
        rows = await c.fetch("SELECT title, schema FROM document_metadata;")
    schema_map: dict[str, list[str]] = {}
    for r in rows:
        title = (r["title"] or "").strip()
        raw = r["schema"]
        cols = []
        try:
            js = json.loads(raw) if raw else {}
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
        except:
            cols = []
        if title:
            schema_map[title.lower()] = cols
    SCHEMA_MAP = schema_map
    lines = []
    for t, cols in SCHEMA_MAP.items():
        if cols:
            lines.append(f"- {t}: {', '.join(cols[:40])}")
    SCHEMA_HINT = "Dataset field hints:\n" + "\n".join(lines[:60])
    inv: dict[str, set[str]] = {}
    for ds, cols in SCHEMA_MAP.items():
        for col in cols:
            for tok in re.findall(r"[a-zA-Z0-9]+", col.lower()):
                inv.setdefault(tok, set()).add(ds)
        for tok in re.findall(r"[a-zA-Z0-9]+", ds):
            inv.setdefault(tok, set()).add(ds)
    FIELD_INVERT = inv

def choose_candidate_datasets(question: str, k: int = 3) -> list[str]:
    q = question.lower()
    tokens = set(re.findall(r"[a-zA-Z0-9]+", q))
    scores: dict[str, int] = {}
    for tok in tokens:
        if tok in FIELD_INVERT:
            for ds in FIELD_INVERT[tok]:
                scores[ds] = scores.get(ds, 0) + 2
    for ds in SCHEMA_MAP.keys():
        if ds in q:
            scores[ds] = scores.get(ds, 0) + 3
    for hint, ds in [("lead", "leads"), ("contact", "contacts"), ("stage", "stages"), ("activit", "activities"), ("user", "users"), ("partner", "contacts")]:
        if hint in q and ds in SCHEMA_MAP:
            scores[ds] = scores.get(ds, 0) + 2
    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return [ds for ds, _ in ranked[:k]]

def build_dynamic_hint(question: str) -> str:
    cands = choose_candidate_datasets(question)
    parts = []
    if cands:
        for ds in cands:
            cols = SCHEMA_MAP.get(ds, [])[:30]
            if cols:
                parts.append(f"Dataset '{ds}' fields: {', '.join(cols)}")
    q = question.lower()
    rules = []
    if any(w in q for w in ["highest", "max", "maximum", "top", "greatest"]):
        rules.append("Use MAX(...) with a correlated comparison so only rows matching the maximum value are returned.")
    if any(w in q for w in ["latest", "most recent", "last", "newest", "recent"]):
        rules.append("Use a correlated subquery comparing each row's date to MAX(date) per entity; do not rely only on ORDER BY.")
    if any(w in q for w in ["less than", "<", "greater than", ">", "below", "above", "at least", "at most", "no more than", "no less than"]):
        rules.append("Cast numeric text with ::numeric before comparison.")
    if "stage" in q:
        rules.append("Join leads to stages on leads.row_data->>'stage_id' = stages.row_data->>'id' using document_rows/document_metadata only.")
    if "partner" in q or "partners" in q:
        rules.append("Join contacts to leads via contacts.row_data->>'id' = leads.row_data->>'partner_id'.")
    if "decision maker" in q or "buying role" in q:
        rules.append("Filter contacts by row_data->>'Buying Role' = 'Decision Maker'.")
    if parts or rules:
        return ("Context:\n" + ("\n".join(parts) if parts else "")) + ("\nGuidance:\n- " + "\n- ".join(rules) if rules else "")
    return ""

async def openai_chat(messages):
    if not OPENAI_KEY:
        raise HTTPException(400, "OPENAI_API_KEY missing")
    payload = {"model": OPENAI_MODEL, "temperature": 0, "messages": messages}
    async with httpx.AsyncClient(timeout=60) as cx:
        try:
            r = await cx.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"},
                json=payload,
            )
            r.raise_for_status()
        except httpx.TimeoutException:
            raise HTTPException(504, "OpenAI request timed out")
    return r.json()["choices"][0]["message"]["content"]

async def gpt_sql_with_repair(question: str) -> str:
    dyn = build_dynamic_hint(question)
    base_user = f"{SCHEMA_HINT}\n\n{dyn}\n\nUser question: {question}\nReturn ONLY the SQL."
    sys = {"role": "system", "content": SYSTEM_PROMPT_SQL_AGENT}
    attempts = []
    prompts = [
        base_user,
        f"{SCHEMA_HINT}\n\nRegenerate using ONLY document_rows and document_metadata. Never use dataset names as tables. {dyn}\n\nUser question: {question}\nReturn ONLY the SQL.",
        f"{SCHEMA_HINT}\n\nRegenerate using CTEs if helpful. Ensure dataset ids resolve via document_metadata.title. Use correlated subqueries for MAX/Latest semantics. {dyn}\n\nUser question: {question}\nReturn ONLY the SQL.",
        f"{SCHEMA_HINT}\n\nFinal attempt. Start with SELECT or WITH. Use only document_rows/document_metadata. Deterministic. {dyn}\n\nUser question: {question}\nReturn ONLY the SQL.",
    ]
    for i in range(4):
        out = await openai_chat([sys, {"role": "user", "content": prompts[i]}])
        sql = sanitize_llm_sql(out)
        try:
            is_select_only(sql)
            tables_allowed(sql)
            return sql
        except Exception as e:
            attempts.append((sql, str(e)))
            continue
    raise HTTPException(400, "Could not generate valid SQL")

def wrap_limit_one(sql: str) -> str:
    core = sql.strip()
    if core.endswith(";"):
        core = core[:-1]
    return f"SELECT * FROM ({core}) __q LIMIT 1;"

async def dry_validate(conn, sql: str) -> None:
    await conn.execute(f"SET LOCAL statement_timeout = {STATEMENT_TIMEOUT_MS}")
    await conn.fetch(wrap_limit_one(sql))

async def repair_from_error(question: str, sql: str, error_msg: str) -> str:
    dyn = build_dynamic_hint(question)
    sys = {"role": "system", "content": SYSTEM_PROMPT_SQL_AGENT}
    user = f"{SCHEMA_HINT}\n\n{dyn}\n\nPrevious SQL caused error:\n{error_msg}\n\nFix and return ONLY corrected SQL for the same question:\n{sql}"
    out = await openai_chat([sys, {"role": "user", "content": user}])
    fixed = sanitize_llm_sql(out)
    is_select_only(fixed)
    tables_allowed(fixed)
    return fixed

@app.on_event("startup")
async def startup():
    app.state.pool = await asyncpg.create_pool(dsn=DB_URL, min_size=1, max_size=8)
    await load_schema_cache(app.state.pool)

@app.on_event("shutdown")
async def shutdown():
    pool = app.state.pool
    if pool:
        await pool.close()

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

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
        # return {"session_id": body.sessionId or "none", "needs_data": (body.needs_data or "YES").upper(), "question": body.chatInput.strip(), "sql": sql, "data": data}
    # except Exception as e:
    #     return {"session_id": body.sessionId or "none", "error": f"DB error: {e}", "sql": sql}
    
        report_payload = {
            "question": body.chatInput.strip(),
            "sqlData": data if data else None,
        }
        # You can reuse the internal logic of the generate_report function here directly
        url = "https://api.openai.com/v1/chat/completions"
        payload = {
            "model": "gpt-4.1",
            "temperature": 0,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_REPORT_AGENT},
                {"role": "user", "content": str(report_payload)}
            ]
        }
        headers = {
            "Authorization": f"Bearer {OPENAI_KEY}",
            "Content-Type": "application/json"
        }
        import httpx
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            report_result = response.json()
        report_content = report_result["choices"][0]["message"]["content"]

        # Step 3: Return combined response
        return {
            "session_id": body.sessionId or "none",
            "needs_data": (body.needs_data or "YES").upper(),
            "question": body.chatInput.strip(),
            "sql": sql,
            "data": data,
            "report": report_content
        }
    except Exception as e:
        return {
            "session_id": body.sessionId or "none",
            "error": f"DB error: {e}",
            "sql": sql,
            "report": None
        }
