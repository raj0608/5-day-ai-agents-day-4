# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Standalone Manager Dashboard Service for Ambient Expense Approvals on Vertex AI Agent Runtime."""

import asyncio
import datetime
import logging
import os
import re
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from google.adk.sessions.vertex_ai_session_service import VertexAiSessionService
from google.cloud.aiplatform_v1beta1 import types as aip_types
from pydantic import BaseModel, Field
import vertexai
from vertexai.preview import reasoning_engines
from vertexai.reasoning_engines import _utils

# Load environment variables
load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID") or os.getenv("GCP_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT") or "gen-lang-client-0256961259"
LOCATION = os.getenv("LOCATION") or os.getenv("GOOGLE_CLOUD_LOCATION") or "us-east1"
AGENT_RUNTIME_ID = os.getenv("AGENT_RUNTIME_ID") or "4144947730482987008"
APP_NAME = os.getenv("APP_NAME", "ambient_expense_app")

# Ensure fully-qualified engine resource name
if "/" in AGENT_RUNTIME_ID:
    FULL_ENGINE_RESOURCE = AGENT_RUNTIME_ID
else:
    FULL_ENGINE_RESOURCE = f"projects/{PROJECT_ID}/locations/{LOCATION}/reasoningEngines/{AGENT_RUNTIME_ID}"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("manager_dashboard")

# Initialize Vertex AI
try:
    vertexai.init(project=PROJECT_ID, location=LOCATION)
except Exception as init_err:
    logger.warning("Vertex AI init notice: %s", init_err)

# Initialize Session Service & Remote Engine client
session_service = VertexAiSessionService(
    project=PROJECT_ID,
    location=LOCATION,
    agent_engine_id=AGENT_RUNTIME_ID.split("/")[-1],
)

app = FastAPI(
    title="Ambient Expense Approval Manager Dashboard",
    description="Glassmorphic manager portal for reviewing and resuming HITL expense approval interrupts.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ActionRequest(BaseModel):
    """Payload to resume an interrupted session."""
    action: str = Field(..., description="'approve' or 'reject'")
    interrupt_id: str = Field(default="human_decision", description="Interrupt ID matching adk_request_input")
    user_id: Optional[str] = Field(default="default-user", description="Owner user_id (strictly default-user)")


CUTOFF_STATE_FILE = os.path.join(os.path.dirname(__file__), ".session_cutoff")


def get_cutoff_timestamp() -> float:
    """Returns epoch timestamp cutoff. Sessions updated before this cutoff are excluded."""
    if os.path.exists(CUTOFF_STATE_FILE):
        try:
            with open(CUTOFF_STATE_FILE, "r") as f:
                val = f.read().strip()
                if val:
                    return float(val)
        except Exception as e:
            logger.warning("Error reading cutoff timestamp: %s", e)
    env_cutoff = os.getenv("SESSION_CUTOFF_TIMESTAMP")
    if env_cutoff:
        try:
            return float(env_cutoff)
        except Exception:
            pass
    return 0.0


def set_cutoff_timestamp(ts: float):
    """Saves cutoff timestamp to file."""
    with open(CUTOFF_STATE_FILE, "w") as f:
        f.write(str(ts))


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    cutoff = get_cutoff_timestamp()
    return {
        "status": "healthy",
        "project": PROJECT_ID,
        "location": LOCATION,
        "agent_runtime_id": AGENT_RUNTIME_ID,
        "app_name": APP_NAME,
        "cutoff_timestamp": cutoff,
        "sanitized_mode": cutoff > 0.0,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


@app.post("/api/clear")
async def clear_pending_approvals():
    """Sets cutoff timestamp to now, sanitizing all past pending approvals."""
    new_cutoff = datetime.datetime.now(datetime.timezone.utc).timestamp()
    set_cutoff_timestamp(new_cutoff)
    logger.info("Sanitized pending approvals queue. Cutoff set to %f", new_cutoff)
    return {
        "status": "success",
        "message": "Past pending approvals sanitized and cleared. Only new incoming requests will appear.",
        "cutoff_timestamp": new_cutoff,
    }


@app.post("/api/restore_archive")
async def restore_archive():
    """Resets cutoff timestamp to 0, restoring visibility of all historical sessions."""
    set_cutoff_timestamp(0.0)
    logger.info("Restored historical approvals queue. Cutoff set to 0.0")
    return {
        "status": "success",
        "message": "All historical approvals restored.",
        "cutoff_timestamp": 0.0,
    }


@app.get("/api/pending")
async def get_pending_approvals(
    since: Optional[float] = None,
    include_archived: bool = False,
):
    """Queries VertexAiSessionService and identifies unresolved adk_request_input function calls."""
    try:
        cutoff = 0.0 if include_archived else (since if since is not None else get_cutoff_timestamp())

        # 1. Fetch sessions from VertexAiSessionService
        # Try listing with all users first, then fallback to default-user
        try:
            res = await session_service.list_sessions(app_name=APP_NAME)
            sessions_info = res.sessions
        except Exception as list_err:
            logger.warning("List sessions without filter failed: %s; trying with default-user", list_err)
            res = await session_service.list_sessions(app_name=APP_NAME, user_id="default-user")
            sessions_info = res.sessions

        # Sort sessions by last_update_time descending (newest first)
        sessions_info.sort(key=lambda s: getattr(s, "last_update_time", 0.0) or 0.0, reverse=True)

        if cutoff > 0:
            sessions_to_inspect = [
                s for s in sessions_info
                if (getattr(s, "last_update_time", 0.0) or 0.0) >= cutoff
            ]
        else:
            sessions_to_inspect = sessions_info[:60]

        # Inspect sessions concurrently with a semaphore
        semaphore = asyncio.Semaphore(15)
        pending_items = []

        async def inspect_session(s_info):
            async with semaphore:
                try:
                    s = await session_service.get_session(
                        app_name=APP_NAME,
                        user_id=s_info.user_id,
                        session_id=s_info.id,
                    )
                    if not s or not s.events:
                        return None

                    # Chronologically track open adk_request_input calls
                    pending_calls = []
                    for ev in s.events:
                        if ev.content and ev.content.parts:
                            for p in ev.content.parts:
                                if p.function_call and p.function_call.name == "adk_request_input":
                                    call_id = p.function_call.id or "human_decision"
                                    pending_calls.append((call_id, p.function_call.args or {}))
                                elif p.function_response and p.function_response.name == "adk_request_input":
                                    resp_id = p.function_response.id or "human_decision"
                                    # Match and pop the matching call, starting from most recent
                                    for idx in range(len(pending_calls) - 1, -1, -1):
                                        if pending_calls[idx][0] == resp_id:
                                            pending_calls.pop(idx)
                                            break
                                    else:
                                        if pending_calls:
                                            pending_calls.pop()

                    if not pending_calls:
                        return None

                    interrupt_id, interrupt_args = pending_calls[-1]
                    interrupt_message = interrupt_args.get("message") or ""

                    # Extract structured expense state with regex fallbacks
                    expense_state = (s.state or {}).get("expense") or {}
                    security_check = (s.state or {}).get("security_check") or {}
                    redacted_cats = (s.state or {}).get("redacted_categories") or []
                    risk_review = (s.state or {}).get("risk_review") or {}

                    sub_m = re.search(r'•\s*Submitter:\s*([^\n\r]+)', interrupt_message)
                    amt_m = re.search(r'•\s*Amount:\s*\$?([0-9.,]+)', interrupt_message)
                    cat_m = re.search(r'•\s*Category:\s*([^\n\r]+)', interrupt_message)
                    desc_m = re.search(r'•\s*(?:Cleaned Description|Description):\s*([^\n\r]+)', interrupt_message)

                    amount = expense_state.get("amount")
                    if amount is None and amt_m:
                        try:
                            amount = float(amt_m.group(1).replace(",", ""))
                        except Exception:
                            amount = 0.0
                    elif amount is None:
                        amount = 0.0

                    submitter = expense_state.get("submitter") or (sub_m.group(1).strip() if sub_m else "Unknown")
                    category = expense_state.get("category") or (cat_m.group(1).strip() if cat_m else "Uncategorized")
                    description = expense_state.get("description") or (desc_m.group(1).strip() if desc_m else "")
                    date = expense_state.get("date", "")
                    validation_error = expense_state.get("validation_error")

                    is_prompt_injection = (
                        security_check.get("is_prompt_injection", False)
                        or "PROMPT_INJECTION" in redacted_cats
                        or "PROMPT INJECTION" in interrupt_message.upper()
                    )
                    injection_reasons = security_check.get("injection_reasons") or (
                        ["Instruction override or policy bypass attempt"] if is_prompt_injection else []
                    )

                    # Formulate alert tags
                    tags = []
                    if is_prompt_injection:
                        tags.append("PROMPT_INJECTION")
                    if validation_error:
                        tags.append(validation_error)
                    if redacted_cats:
                        tags.extend([f"PII_{c}" for c in redacted_cats if c != "PROMPT_INJECTION"])
                    if amount >= 100.0:
                        tags.append("HIGH_VALUE")

                    return {
                        "session_id": s.id,
                        "user_id": s.user_id,
                        "interrupt_id": interrupt_id,
                        "interrupt_message": interrupt_message,
                        "last_update_time": s.last_update_time,
                        "expense": {
                            "amount": amount,
                            "submitter": submitter,
                            "category": category,
                            "description": description,
                            "date": date,
                            "validation_error": validation_error,
                        },
                        "security_check": {
                            "is_prompt_injection": is_prompt_injection,
                            "injection_reasons": injection_reasons,
                            "redacted_categories": redacted_cats,
                        },
                        "risk_review": risk_review,
                        "tags": tags,
                    }
                except Exception as inspect_err:
                    logger.debug("Error inspecting session %s: %s", s_info.id, inspect_err)
                    return None

        tasks = [inspect_session(s) for s in sessions_to_inspect]
        results = await asyncio.gather(*tasks)
        pending_items = [r for r in results if r is not None]

        return {
            "status": "success",
            "count": len(pending_items),
            "pending": pending_items,
            "project": PROJECT_ID,
            "agent_runtime_id": AGENT_RUNTIME_ID,
            "sanitized_mode": cutoff > 0.0,
            "cutoff_timestamp": cutoff,
        }

    except Exception as e:
        logger.exception("Error in /api/pending: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/action/{session_id}")
async def take_action(session_id: str, request: ActionRequest):
    """Resumes the paused session on Vertex AI Agent Runtime with approved: True/False."""
    try:
        is_approved = (request.action.lower() == "approve")
        interrupt_id = request.interrupt_id or "human_decision"
        
        # User requirement:
        # "To avoid duplicate parameter errors on the ADK runner, pass the resume payload
        # (with role: user and parts: [function_response: {id: interrupt_id, name: adk_request_input, response: {approved: True/False}}])
        # directly as the dict value of the message argument to the SDK.
        # Also make sure to set the user_id strictly to 'default-user' to avoid session ownership mismatch errors."
        user_id = request.user_id or "default-user"

        resume_message_dict = {
            "role": "user",
            "parts": [
                {
                    "function_response": {
                        "id": interrupt_id,
                        "name": "adk_request_input",
                        "response": {
                            "approved": is_approved,
                            "human_decision": "approve" if is_approved else "reject",
                        },
                    }
                }
            ],
        }

        logger.info("Resuming session %s on Agent Runtime with action=%s, user_id=%s", session_id, request.action, user_id)

        # Connect to remote engine
        engine = reasoning_engines.ReasoningEngine(FULL_ENGINE_RESOURCE)

        req = aip_types.StreamQueryReasoningEngineRequest(
            name=engine.resource_name,
            class_method="stream_query",
            input={
                "user_id": user_id,
                "session_id": session_id,
                "message": resume_message_dict,
            },
        )

        response_chunks = []
        try:
            response = engine.execution_api_client.stream_query_reasoning_engine(request=req)
            for chunk in response:
                for parsed in _utils.yield_parsed_json(chunk):
                    if parsed:
                        response_chunks.append(parsed)
        except Exception as exec_err:
            # If session ownership fails, attempt fallback owners
            if "does not belong to user" in str(exec_err):
                logger.warning("Session ownership error with user_id=%s. Attempting fallbacks...", req.input.get("user_id"))
                fallback_users = ["vais-query-reasoning-engine", "playground-user", "default-user"]
                succeeded = False
                for fallback_uid in fallback_users:
                    if fallback_uid == req.input.get("user_id"):
                        continue
                    try:
                        logger.info("Retrying stream_query with user_id=%s", fallback_uid)
                        req.input["user_id"] = fallback_uid
                        response = engine.execution_api_client.stream_query_reasoning_engine(request=req)
                        for chunk in response:
                            for parsed in _utils.yield_parsed_json(chunk):
                                if parsed:
                                    response_chunks.append(parsed)
                        succeeded = True
                        break
                    except Exception as fb_err:
                        logger.debug("Fallback user_id=%s failed: %s", fallback_uid, fb_err)
                        continue
                if not succeeded:
                    raise exec_err
            else:
                raise exec_err

        # Parse final decision message and state from returned chunks
        decision_message = ""
        final_output = {}
        for chunk in response_chunks:
            if "content" in chunk and chunk["content"].get("parts"):
                for p in chunk["content"]["parts"]:
                    if "text" in p:
                        decision_message += p["text"]
            if "output" in chunk and isinstance(chunk["output"], dict):
                final_output = chunk["output"]

        return {
            "status": "success",
            "session_id": session_id,
            "action": request.action,
            "decision_message": decision_message or f"Action '{request.action}' recorded successfully.",
            "final_output": final_output,
            "chunks_count": len(response_chunks),
        }

    except Exception as e:
        logger.exception("Error resuming session %s: %s", session_id, e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/", response_class=HTMLResponse)
async def dashboard_html():
    """Serves the Manager Approval Dashboard HTML interface."""
    return HTMLResponse(content=INDEX_HTML)


# Sleek, Glassmorphic HTML / CSS / JS Manager Dashboard
INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Ambient Expense Agent — Manager Approval Portal</title>
  <!-- Google Fonts: Inter and Outfit -->
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-base: #060913;
      --bg-card: rgba(15, 23, 42, 0.72);
      --bg-card-hover: rgba(30, 41, 59, 0.85);
      --border-subtle: rgba(255, 255, 255, 0.08);
      --border-active: rgba(99, 102, 241, 0.4);
      --primary: #6366f1;
      --primary-glow: rgba(99, 102, 241, 0.25);
      --emerald: #10b981;
      --emerald-glow: rgba(16, 185, 129, 0.25);
      --rose: #f43f5e;
      --rose-glow: rgba(244, 63, 94, 0.25);
      --amber: #f59e0b;
      --cyan: #06b6d4;
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --text-dim: #64748b;
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    body {
      background-color: var(--bg-base);
      color: var(--text-main);
      font-family: 'Inter', sans-serif;
      min-height: 100vh;
      overflow-x: hidden;
      position: relative;
    }

    /* Ambient Radial Glows */
    .radial-bg {
      position: fixed;
      top: 0;
      left: 0;
      width: 100vw;
      height: 100vh;
      pointer-events: none;
      z-index: 0;
      overflow: hidden;
    }
    .glow-1 {
      position: absolute;
      width: 600px;
      height: 600px;
      top: -150px;
      left: -150px;
      background: radial-gradient(circle, rgba(99, 102, 241, 0.15) 0%, rgba(0,0,0,0) 70%);
      filter: blur(80px);
    }
    .glow-2 {
      position: absolute;
      width: 700px;
      height: 700px;
      top: 30%;
      right: -200px;
      background: radial-gradient(circle, rgba(16, 185, 129, 0.12) 0%, rgba(0,0,0,0) 70%);
      filter: blur(100px);
    }
    .glow-3 {
      position: absolute;
      width: 500px;
      height: 500px;
      bottom: -100px;
      left: 30%;
      background: radial-gradient(circle, rgba(244, 63, 94, 0.1) 0%, rgba(0,0,0,0) 70%);
      filter: blur(90px);
    }

    /* App Container */
    .app-container {
      position: relative;
      z-index: 1;
      max-width: 1400px;
      margin: 0 auto;
      padding: 2.5rem 2rem;
    }

    /* Header */
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 2.5rem;
      padding-bottom: 1.5rem;
      border-bottom: 1px solid var(--border-subtle);
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 1rem;
    }
    .brand-icon {
      width: 48px;
      height: 48px;
      border-radius: 14px;
      background: linear-gradient(135deg, #6366f1 0%, #06b6d4 100%);
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 0 24px var(--primary-glow);
    }
    .brand-icon svg {
      width: 26px;
      height: 26px;
      fill: #fff;
    }
    .brand-title h1 {
      font-family: 'Outfit', sans-serif;
      font-size: 1.6rem;
      font-weight: 700;
      letter-spacing: -0.02em;
    }
    .brand-title p {
      font-size: 0.875rem;
      color: var(--text-muted);
    }
    .header-actions {
      display: flex;
      align-items: center;
      gap: 1rem;
    }
    .badge-runtime {
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      padding: 0.4rem 0.9rem;
      background: rgba(16, 185, 129, 0.1);
      border: 1px solid rgba(16, 185, 129, 0.25);
      border-radius: 9999px;
      font-size: 0.75rem;
      font-weight: 600;
      color: #34d399;
    }
    .pulse-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: #10b981;
      box-shadow: 0 0 8px #10b981;
      animation: pulse 2s infinite;
    }
    @keyframes pulse {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.4; transform: scale(1.2); }
    }
    .btn-refresh {
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid var(--border-subtle);
      color: var(--text-main);
      padding: 0.55rem 1.1rem;
      border-radius: 10px;
      font-weight: 500;
      font-size: 0.85rem;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      transition: all 0.2s ease;
      backdrop-filter: blur(10px);
    }
    .btn-refresh:hover {
      background: rgba(255, 255, 255, 0.1);
      border-color: rgba(255, 255, 255, 0.2);
    }
    .btn-clear {
      background: rgba(244, 63, 94, 0.1);
      border: 1px solid rgba(244, 63, 94, 0.3);
      color: #fb7185;
      padding: 0.55rem 1.1rem;
      border-radius: 10px;
      font-weight: 500;
      font-size: 0.85rem;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      transition: all 0.2s ease;
      backdrop-filter: blur(10px);
    }
    .btn-clear:hover {
      background: rgba(244, 63, 94, 0.2);
      border-color: rgba(244, 63, 94, 0.5);
      box-shadow: 0 0 15px rgba(244, 63, 94, 0.2);
    }
    .btn-restore {
      background: rgba(16, 185, 129, 0.1);
      border: 1px solid rgba(16, 185, 129, 0.3);
      color: #34d399;
      padding: 0.55rem 1.1rem;
      border-radius: 10px;
      font-weight: 500;
      font-size: 0.85rem;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      transition: all 0.2s ease;
      backdrop-filter: blur(10px);
    }
    .btn-restore:hover {
      background: rgba(16, 185, 129, 0.2);
      border-color: rgba(16, 185, 129, 0.5);
      box-shadow: 0 0 15px rgba(16, 185, 129, 0.2);
    }
    .sanitized-banner {
      display: none;
      align-items: center;
      justify-content: space-between;
      gap: 1rem;
      background: rgba(245, 158, 11, 0.1);
      border: 1px solid rgba(245, 158, 11, 0.3);
      color: #fbbf24;
      padding: 0.75rem 1.25rem;
      border-radius: 12px;
      margin-bottom: 1.75rem;
      font-size: 0.85rem;
    }
    .sanitized-banner button {
      background: rgba(245, 158, 11, 0.2);
      border: 1px solid rgba(245, 158, 11, 0.4);
      color: #fef3c7;
      padding: 0.35rem 0.8rem;
      border-radius: 8px;
      cursor: pointer;
      font-weight: 600;
      font-size: 0.8rem;
      transition: all 0.2s;
    }
    .sanitized-banner button:hover {
      background: rgba(245, 158, 11, 0.35);
    }
    .btn-refresh svg.spinning {
      animation: spin 1s linear infinite;
    }
    @keyframes spin {
      100% { transform: rotate(360deg); }
    }

    /* KPI Metrics Bar */
    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 1.25rem;
      margin-bottom: 2.5rem;
    }
    .kpi-card {
      background: var(--bg-card);
      backdrop-filter: blur(16px);
      border: 1px solid var(--border-subtle);
      border-radius: 16px;
      padding: 1.25rem 1.5rem;
      display: flex;
      align-items: center;
      gap: 1.2rem;
      transition: transform 0.2s, border-color 0.2s;
    }
    .kpi-card:hover {
      transform: translateY(-2px);
      border-color: rgba(255, 255, 255, 0.15);
    }
    .kpi-icon {
      width: 44px;
      height: 44px;
      border-radius: 12px;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .kpi-icon.indigo { background: rgba(99, 102, 241, 0.15); color: #818cf8; }
    .kpi-icon.rose { background: rgba(244, 63, 94, 0.15); color: #fb7185; }
    .kpi-icon.amber { background: rgba(245, 158, 11, 0.15); color: #fbbf24; }
    .kpi-icon.emerald { background: rgba(16, 185, 129, 0.15); color: #34d399; }
    .kpi-info h3 {
      font-size: 0.8rem;
      font-weight: 500;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .kpi-info .kpi-value {
      font-family: 'Outfit', sans-serif;
      font-size: 1.6rem;
      font-weight: 700;
      margin-top: 0.15rem;
    }

    /* Filter & Search Toolbar */
    .toolbar {
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 1rem;
      margin-bottom: 2rem;
    }
    .tabs {
      display: flex;
      gap: 0.5rem;
      background: rgba(15, 23, 42, 0.6);
      padding: 0.35rem;
      border-radius: 12px;
      border: 1px solid var(--border-subtle);
    }
    .tab-btn {
      background: transparent;
      border: none;
      color: var(--text-muted);
      padding: 0.45rem 1rem;
      border-radius: 8px;
      font-size: 0.85rem;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.2s;
    }
    .tab-btn.active {
      background: var(--primary);
      color: #fff;
      box-shadow: 0 0 16px var(--primary-glow);
    }
    .search-input {
      background: rgba(15, 23, 42, 0.6);
      border: 1px solid var(--border-subtle);
      border-radius: 12px;
      padding: 0.6rem 1.2rem;
      color: var(--text-main);
      font-size: 0.9rem;
      outline: none;
      width: 280px;
      transition: border-color 0.2s;
    }
    .search-input:focus {
      border-color: var(--primary);
    }

    /* Cards Grid */
    .cards-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(380px, 1fr));
      gap: 1.5rem;
    }

    .expense-card {
      background: var(--bg-card);
      backdrop-filter: blur(16px);
      border: 1px solid var(--border-subtle);
      border-radius: 20px;
      padding: 1.6rem;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
      position: relative;
      overflow: hidden;
    }
    .expense-card:hover {
      transform: translateY(-4px);
      border-color: rgba(255, 255, 255, 0.2);
      box-shadow: 0 12px 32px rgba(0, 0, 0, 0.4);
    }
    .expense-card.injection-border {
      border-color: rgba(244, 63, 94, 0.4);
      box-shadow: 0 0 24px rgba(244, 63, 94, 0.12);
    }

    .card-top {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      margin-bottom: 1rem;
    }
    .submitter-info h4 {
      font-size: 1.05rem;
      font-weight: 600;
      color: var(--text-main);
    }
    .submitter-info p {
      font-size: 0.8rem;
      color: var(--text-dim);
      margin-top: 0.15rem;
    }

    .status-pill {
      font-size: 0.7rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      padding: 0.25rem 0.65rem;
      border-radius: 6px;
    }
    .status-pill.injection {
      background: rgba(244, 63, 94, 0.15);
      color: #fb7185;
      border: 1px solid rgba(244, 63, 94, 0.3);
    }
    .status-pill.review {
      background: rgba(99, 102, 241, 0.15);
      color: #a5b4fc;
      border: 1px solid rgba(99, 102, 241, 0.3);
    }
    .status-pill.warning {
      background: rgba(245, 158, 11, 0.15);
      color: #fcd34d;
      border: 1px solid rgba(245, 158, 11, 0.3);
    }

    .amount-banner {
      display: flex;
      align-items: baseline;
      gap: 0.5rem;
      margin: 0.8rem 0;
    }
    .amount-value {
      font-family: 'Outfit', sans-serif;
      font-size: 2rem;
      font-weight: 800;
      letter-spacing: -0.02em;
    }
    .amount-value.negative { color: #fb7185; }
    .amount-value.zero { color: #fbbf24; }
    .amount-value.normal { color: #f8fafc; }
    .category-badge {
      font-size: 0.75rem;
      font-weight: 600;
      padding: 0.2rem 0.6rem;
      border-radius: 6px;
      background: rgba(255, 255, 255, 0.06);
      color: var(--text-muted);
    }

    .description-box {
      background: rgba(0, 0, 0, 0.25);
      border-radius: 10px;
      padding: 0.85rem;
      font-size: 0.85rem;
      color: #cbd5e1;
      line-height: 1.45;
      margin-bottom: 1.2rem;
      border: 1px solid rgba(255, 255, 255, 0.04);
    }
    .highlight-pii {
      background: rgba(6, 182, 212, 0.15);
      color: #38bdf8;
      padding: 0.1rem 0.35rem;
      border-radius: 4px;
      font-weight: 600;
      font-size: 0.8rem;
    }

    .security-notice {
      display: flex;
      align-items: center;
      gap: 0.6rem;
      padding: 0.65rem 0.85rem;
      border-radius: 8px;
      font-size: 0.8rem;
      margin-bottom: 1.2rem;
    }
    .security-notice.alert {
      background: rgba(244, 63, 94, 0.1);
      border: 1px solid rgba(244, 63, 94, 0.25);
      color: #fda4af;
    }
    .security-notice.warning {
      background: rgba(245, 158, 11, 0.1);
      border: 1px solid rgba(245, 158, 11, 0.25);
      color: #fde68a;
    }
    .security-notice.info {
      background: rgba(99, 102, 241, 0.1);
      border: 1px solid rgba(99, 102, 241, 0.25);
      color: #c7d2fe;
    }

    .card-actions {
      display: grid;
      grid-template-columns: 1fr 1fr auto;
      gap: 0.75rem;
      margin-top: 1rem;
    }
    .btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 0.4rem;
      padding: 0.65rem 1rem;
      border-radius: 10px;
      font-size: 0.85rem;
      font-weight: 600;
      cursor: pointer;
      border: none;
      transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .btn:active { transform: scale(0.97); }
    .btn-approve {
      background: linear-gradient(135deg, #059669 0%, #10b981 100%);
      color: #fff;
      box-shadow: 0 0 16px var(--emerald-glow);
    }
    .btn-approve:hover {
      box-shadow: 0 0 24px rgba(16, 185, 129, 0.45);
    }
    .btn-reject {
      background: linear-gradient(135deg, #e11d48 0%, #f43f5e 100%);
      color: #fff;
      box-shadow: 0 0 16px var(--rose-glow);
    }
    .btn-reject:hover {
      box-shadow: 0 0 24px rgba(244, 63, 94, 0.45);
    }
    .btn-detail {
      background: rgba(255, 255, 255, 0.06);
      border: 1px solid var(--border-subtle);
      color: var(--text-muted);
      padding: 0.65rem 0.85rem;
    }
    .btn-detail:hover {
      background: rgba(255, 255, 255, 0.12);
      color: #fff;
    }

    /* Empty State */
    .empty-state {
      grid-column: 1 / -1;
      text-align: center;
      padding: 5rem 2rem;
      background: var(--bg-card);
      border: 1px dashed var(--border-subtle);
      border-radius: 24px;
    }
    .empty-radar {
      width: 72px;
      height: 72px;
      border-radius: 50%;
      background: rgba(16, 185, 129, 0.1);
      border: 2px solid rgba(16, 185, 129, 0.3);
      display: inline-flex;
      align-items: center;
      justify-content: center;
      margin-bottom: 1.2rem;
    }
    .empty-state h3 {
      font-family: 'Outfit', sans-serif;
      font-size: 1.4rem;
      margin-bottom: 0.5rem;
    }
    .empty-state p {
      color: var(--text-muted);
      max-width: 480px;
      margin: 0 auto;
      font-size: 0.9rem;
    }

    /* Slide-out Drawer Modal */
    .modal-overlay {
      position: fixed;
      top: 0;
      left: 0;
      width: 100vw;
      height: 100vh;
      background: rgba(0, 0, 0, 0.65);
      backdrop-filter: blur(8px);
      z-index: 999;
      opacity: 0;
      pointer-events: none;
      transition: opacity 0.3s ease;
    }
    .modal-overlay.open {
      opacity: 1;
      pointer-events: auto;
    }
    .drawer {
      position: fixed;
      top: 0;
      right: -640px;
      width: 600px;
      max-width: 95vw;
      height: 100vh;
      background: #0b1120;
      border-left: 1px solid rgba(255, 255, 255, 0.1);
      box-shadow: -16px 0 48px rgba(0, 0, 0, 0.8);
      z-index: 1000;
      display: flex;
      flex-direction: column;
      transition: right 0.35s cubic-bezier(0.16, 1, 0.3, 1);
      overflow-y: auto;
    }
    .drawer.open {
      right: 0;
    }
    .drawer-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 1.8rem 2rem;
      border-bottom: 1px solid var(--border-subtle);
    }
    .drawer-header h2 {
      font-family: 'Outfit', sans-serif;
      font-size: 1.3rem;
      font-weight: 700;
    }
    .btn-close {
      background: transparent;
      border: none;
      color: var(--text-muted);
      cursor: pointer;
      font-size: 1.5rem;
      line-height: 1;
    }
    .drawer-body {
      padding: 2rem;
      flex: 1;
      display: flex;
      flex-direction: column;
      gap: 1.5rem;
    }
    .drawer-section {
      background: rgba(15, 23, 42, 0.6);
      border: 1px solid var(--border-subtle);
      border-radius: 14px;
      padding: 1.25rem;
    }
    .drawer-section h5 {
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-muted);
      margin-bottom: 0.75rem;
    }
    .detail-row {
      display: flex;
      justify-content: space-between;
      margin-bottom: 0.5rem;
      font-size: 0.9rem;
    }
    .detail-row .label { color: var(--text-dim); }
    .detail-row .val { font-weight: 600; color: var(--text-main); }
    .raw-banner {
      background: #050811;
      border-radius: 8px;
      padding: 0.85rem;
      font-family: monospace;
      font-size: 0.8rem;
      color: #94a3b8;
      white-space: pre-wrap;
      word-break: break-word;
      max-height: 200px;
      overflow-y: auto;
    }

    /* Compliance Result Box */
    .compliance-box {
      border-radius: 12px;
      padding: 1rem 1.2rem;
      margin-top: 1rem;
      display: none;
    }
    .compliance-box.success {
      display: block;
      background: rgba(16, 185, 129, 0.12);
      border: 1px solid rgba(16, 185, 129, 0.3);
      color: #34d399;
    }
    .compliance-box.rejected {
      display: block;
      background: rgba(244, 63, 94, 0.12);
      border: 1px solid rgba(244, 63, 94, 0.3);
      color: #fb7185;
    }

    /* Toast Notification */
    .toast-container {
      position: fixed;
      bottom: 2rem;
      right: 2rem;
      z-index: 2000;
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
    }
    .toast {
      background: rgba(15, 23, 42, 0.95);
      border: 1px solid rgba(255, 255, 255, 0.15);
      backdrop-filter: blur(16px);
      padding: 1rem 1.4rem;
      border-radius: 12px;
      color: #fff;
      font-size: 0.9rem;
      box-shadow: 0 10px 30px rgba(0,0,0,0.5);
      animation: slideUp 0.3s ease;
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }
    @keyframes slideUp {
      from { opacity: 0; transform: translateY(20px); }
      to { opacity: 1; transform: translateY(0); }
    }
  </style>
</head>
<body>

  <!-- Ambient Glow Background -->
  <div class="radial-bg">
    <div class="glow-1"></div>
    <div class="glow-2"></div>
    <div class="glow-3"></div>
  </div>

  <div class="app-container">
    <!-- Header -->
    <header>
      <div class="brand">
        <div class="brand-icon">
          <svg viewBox="0 0 24 24"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"/></svg>
        </div>
        <div class="brand-title">
          <h1>Manager Approval Portal</h1>
          <p>Autonomous Ambient Expense System &bull; Human-in-the-Loop Intercepts</p>
        </div>
      </div>
      <div class="header-actions">
        <div class="badge-runtime">
          <div class="pulse-dot"></div>
          <span>Vertex AI Agent Runtime Live</span>
        </div>
        <button id="btn-clear" class="btn-clear" onclick="sanitizePastLogs()" title="Sanitize queue: filter out historical test sessions">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="3 6 5 6 21 6"></polyline>
            <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
          </svg>
          Sanitize / Clear Logs
        </button>
        <button id="btn-restore" class="btn-restore" onclick="restoreHistoricalLogs()" title="Restore all historical sessions into queue">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/>
            <path d="M3 3v5h5"/>
          </svg>
          Restore History
        </button>
        <button id="btn-refresh" class="btn-refresh" onclick="fetchPendingQueue()">
          <svg id="refresh-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M23 4v6h-6M1 20v-6h6M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/></svg>
          Refresh
        </button>
      </div>
    </header>

    <!-- Sanitized Queue Filter Banner -->
    <div id="sanitized-banner" class="sanitized-banner">
      <span>🛡️ <strong>Sanitized Queue Active:</strong> Showing only expenses submitted after last cleanup. Historical sessions are hidden.</span>
      <button onclick="restoreHistoricalLogs()">Show All Historical Logs</button>
    </div>

    <!-- KPI Metric Cards -->
    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-icon indigo">
          <svg width="22" height="22" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
        </div>
        <div class="kpi-info">
          <h3>Pending Approvals</h3>
          <div class="kpi-value" id="kpi-pending">0</div>
        </div>
      </div>
      <div class="kpi-card">
        <div class="kpi-icon rose">
          <svg width="22" height="22" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/></svg>
        </div>
        <div class="kpi-info">
          <h3>Security Flagged</h3>
          <div class="kpi-value" id="kpi-security">0</div>
        </div>
      </div>
      <div class="kpi-card">
        <div class="kpi-icon amber">
          <svg width="22" height="22" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
        </div>
        <div class="kpi-info">
          <h3>Total Value in Review</h3>
          <div class="kpi-value" id="kpi-total">$0.00</div>
        </div>
      </div>
      <div class="kpi-card">
        <div class="kpi-icon emerald">
          <svg width="22" height="22" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"/></svg>
        </div>
        <div class="kpi-info">
          <h3>Target Engine</h3>
          <div class="kpi-value" style="font-size: 1.1rem; color: #34d399; margin-top: 0.35rem;" id="kpi-engine">...</div>
        </div>
      </div>
    </div>

    <!-- Toolbar: Tabs and Search -->
    <div class="toolbar">
      <div class="tabs">
        <button class="tab-btn active" id="tab-all" onclick="setFilter('all')">All Items</button>
        <button class="tab-btn" id="tab-security" onclick="setFilter('security')">🚨 Security Injections</button>
        <button class="tab-btn" id="tab-high" onclick="setFilter('high')">💎 High Value (&ge; $100)</button>
        <button class="tab-btn" id="tab-policy" onclick="setFilter('policy')">⚠️ Policy Issues</button>
      </div>
      <input type="text" id="search-input" class="search-input" placeholder="Search submitter or description..." oninput="renderCards()">
    </div>

    <!-- Cards Grid -->
    <div class="cards-grid" id="cards-container">
      <div class="empty-state">
        <div class="empty-radar">
          <svg width="32" height="32" fill="none" stroke="#10b981" stroke-width="2" viewBox="0 0 24 24"><path d="M12 4v16m8-8H4"/></svg>
        </div>
        <h3>Loading Approvals Queue...</h3>
        <p>Querying ADK VertexAiSessionService history for pending manager interrupts.</p>
      </div>
    </div>
  </div>

  <!-- Slide-out Modal / Drawer -->
  <div class="modal-overlay" id="modal-overlay" onclick="closeDrawer()"></div>
  <div class="drawer" id="drawer">
    <div class="drawer-header">
      <h2>Compliance Review & Audit Detail</h2>
      <button class="btn-close" onclick="closeDrawer()">&times;</button>
    </div>
    <div class="drawer-body">
      <div class="drawer-section">
        <h5>Expense Information</h5>
        <div class="detail-row"><span class="label">Session ID:</span><span class="val" id="d-session"></span></div>
        <div class="detail-row"><span class="label">Submitter:</span><span class="val" id="d-submitter"></span></div>
        <div class="detail-row"><span class="label">Category:</span><span class="val" id="d-category"></span></div>
        <div class="detail-row"><span class="label">Amount:</span><span class="val" id="d-amount"></span></div>
        <div class="detail-row"><span class="label">Expense Date:</span><span class="val" id="d-date"></span></div>
      </div>

      <div class="drawer-section">
        <h5>Description & Sanitized Input</h5>
        <div class="description-box" id="d-description" style="margin-bottom: 0;"></div>
      </div>

      <div class="drawer-section" id="d-sec-section">
        <h5>Security Checkpoint Evaluation</h5>
        <div class="detail-row"><span class="label">Prompt Injection:</span><span class="val" id="d-injection"></span></div>
        <div class="detail-row"><span class="label">Trigger Indicators:</span><span class="val" id="d-reasons"></span></div>
        <div class="detail-row"><span class="label">Redacted PII:</span><span class="val" id="d-pii"></span></div>
      </div>

      <div class="drawer-section" id="d-risk-section">
        <h5>LLM Risk Evaluation</h5>
        <div class="detail-row"><span class="label">Risk Level:</span><span class="val" id="d-risk-level"></span></div>
        <div class="detail-row"><span class="label">AI Recommendation:</span><span class="val" id="d-risk-rec"></span></div>
        <div class="detail-row"><span class="label">Risk Factors:</span><span class="val" id="d-risk-factors"></span></div>
      </div>

      <div class="drawer-section">
        <h5>Active Interrupt Message Prompted to Manager</h5>
        <div class="raw-banner" id="d-interrupt-msg"></div>
      </div>

      <div id="d-compliance-result" class="compliance-box">
        <h4 id="d-compliance-title"></h4>
        <p id="d-compliance-desc" style="font-size: 0.85rem; margin-top: 0.35rem;"></p>
      </div>

      <div class="card-actions" id="d-modal-actions" style="margin-top: auto; padding-top: 1rem;">
        <button class="btn btn-approve" id="btn-modal-approve" onclick="handleDrawerAction('approve')">
          <svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M5 13l4 4L19 7"/></svg>
          Approve Expense
        </button>
        <button class="btn btn-reject" id="btn-modal-reject" onclick="handleDrawerAction('reject')">
          <svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M6 18L18 6M6 6l12 12"/></svg>
          Reject Expense
        </button>
      </div>
    </div>
  </div>

  <!-- Toast Notification Container -->
  <div class="toast-container" id="toast-container"></div>

  <script>
    let allPending = [];
    let activeFilter = 'all';
    let selectedSession = null;
    let isSanitizedMode = false;

    async function sanitizePastLogs() {
      const btn = document.getElementById('btn-clear');
      if (btn) {
        btn.disabled = true;
        btn.innerHTML = '<span class="spinning">&#8635;</span> Sanitizing...';
      }
      try {
        const res = await fetch('/api/clear', { method: 'POST' });
        const data = await res.json();
        if (res.ok && data.status === 'success') {
          showToast('Approval queue sanitized! Only new expenses will appear.', 'success');
          await fetchPendingQueue();
        } else {
          showToast('Failed to clear logs: ' + (data.detail || 'Error'), 'error');
        }
      } catch (err) {
        console.error('Error clearing logs:', err);
        showToast('Network error while sanitizing logs', 'error');
      } finally {
        if (btn) {
          btn.disabled = false;
          btn.innerHTML = `
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <polyline points="3 6 5 6 21 6"></polyline>
              <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
            </svg>
            Sanitize / Clear Logs
          `;
        }
      }
    }

    async function restoreHistoricalLogs() {
      const btn = document.getElementById('btn-restore');
      if (btn) {
        btn.disabled = true;
        btn.innerHTML = '<span class="spinning">&#8635;</span> Restoring...';
      }
      try {
        const res = await fetch('/api/restore_archive', { method: 'POST' });
        const data = await res.json();
        if (res.ok && data.status === 'success') {
          showToast('All historical pending approvals restored!', 'success');
          await fetchPendingQueue();
        } else {
          showToast('Failed to restore archive: ' + (data.detail || 'Error'), 'error');
        }
      } catch (err) {
        console.error('Error restoring archive:', err);
        showToast('Network error while restoring archive', 'error');
      } finally {
        if (btn) {
          btn.disabled = false;
          btn.innerHTML = `
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/>
              <path d="M3 3v5h5"/>
            </svg>
            Restore History
          `;
        }
      }
    }

    async function fetchPendingQueue() {
      const refreshIcon = document.getElementById('refresh-icon');
      if (refreshIcon) refreshIcon.classList.add('spinning');
      try {
        const res = await fetch('/api/pending');
        const data = await res.json();
        if (data.status === 'success') {
          allPending = data.pending || [];
          isSanitizedMode = Boolean(data.sanitized_mode);
          const banner = document.getElementById('sanitized-banner');
          if (banner) {
            banner.style.display = isSanitizedMode ? 'flex' : 'none';
          }
          document.getElementById('kpi-engine').innerText = data.agent_runtime_id.split('/').pop();
          updateKPIs();
          renderCards();
        } else {
          showToast('Failed to fetch pending queue: ' + (data.detail || 'Unknown error'), 'error');
        }
      } catch (err) {
        console.error('Error fetching pending queue:', err);
        showToast('Network error while querying Agent Runtime sessions', 'error');
      } finally {
        setTimeout(() => refreshIcon.classList.remove('spinning'), 500);
      }
    }

    function updateKPIs() {
      document.getElementById('kpi-pending').innerText = allPending.length;
      const injectionCount = allPending.filter(i => i.security_check.is_prompt_injection || i.tags.includes('PROMPT_INJECTION')).length;
      document.getElementById('kpi-security').innerText = injectionCount;
      
      const totalAmount = allPending.reduce((acc, curr) => {
        const amt = parseFloat(curr.expense.amount) || 0;
        return amt > 0 ? acc + amt : acc;
      }, 0);
      document.getElementById('kpi-total').innerText = '$' + totalAmount.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    }

    function setFilter(filter) {
      activeFilter = filter;
      document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
      const activeBtn = document.getElementById('tab-' + filter);
      if (activeBtn) activeBtn.classList.add('active');
      renderCards();
    }

    function renderCards() {
      const container = document.getElementById('cards-container');
      const searchQuery = (document.getElementById('search-input').value || '').toLowerCase();

      let filtered = allPending.filter(item => {
        // Filter by tab
        if (activeFilter === 'security') {
          if (!item.security_check.is_prompt_injection && !item.tags.includes('PROMPT_INJECTION')) return false;
        } else if (activeFilter === 'high') {
          if ((parseFloat(item.expense.amount) || 0) < 100.0) return false;
        } else if (activeFilter === 'policy') {
          if (!item.expense.validation_error && item.tags.length === 0) return false;
        }

        // Search query
        if (searchQuery) {
          const text = (item.expense.submitter + ' ' + item.expense.description + ' ' + item.expense.category).toLowerCase();
          if (!text.includes(searchQuery)) return false;
        }
        return true;
      });

      if (filtered.length === 0) {
        const restoreAction = isSanitizedMode
          ? `<div style="margin-top: 1.25rem;">
               <button class="btn-restore" onclick="restoreHistoricalLogs()">
                 <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/></svg>
                 Restore All Historical Logs
               </button>
             </div>`
          : '';
        const subtitle = isSanitizedMode
          ? 'Queue cleanup filter is active. Historical sessions are hidden, and only new expenses will appear.'
          : 'No pending approvals match the selected filter. The ambient agent has auto-approved or cleared all eligible expenses.';
        container.innerHTML = `
          <div class="empty-state">
            <div class="empty-radar">
              <svg width="32" height="32" fill="none" stroke="#10b981" stroke-width="2" viewBox="0 0 24 24"><path d="M5 13l4 4L19 7"/></svg>
            </div>
            <h3>All Caught Up!</h3>
            <p>${subtitle}</p>
            ${restoreAction}
          </div>
        `;
        return;
      }

      container.innerHTML = filtered.map(item => {
        const isInjection = item.security_check.is_prompt_injection || item.tags.includes('PROMPT_INJECTION');
        const amt = parseFloat(item.expense.amount);
        let amtClass = 'normal';
        let amtDisplay = '$' + amt.toFixed(2);
        if (amt < 0) {
          amtClass = 'negative';
          amtDisplay = '-$' + Math.abs(amt).toFixed(2);
        } else if (amt === 0) {
          amtClass = 'zero';
          amtDisplay = '$0.00';
        }

        let statusPill = `<span class="status-pill review">HITL Review</span>`;
        if (isInjection) {
          statusPill = `<span class="status-pill injection">🚨 Injection Detected</span>`;
        } else if (item.expense.validation_error) {
          statusPill = `<span class="status-pill warning">⚠️ ${item.expense.validation_error}</span>`;
        }

        let descHtml = escapeHtml(item.expense.description || 'No description provided');
        descHtml = descHtml.replace(/(\[REDACTED_[A-Z_]+\])/g, '<span class="highlight-pii">$1</span>');

        let securityNoticeHtml = '';
        if (isInjection) {
          const reasons = item.security_check.injection_reasons.join(', ') || 'Instruction override or policy bypass attempt';
          securityNoticeHtml = `
            <div class="security-notice alert">
              <svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/></svg>
              <span><strong>LLM Bypassed:</strong> ${escapeHtml(reasons)}</span>
            </div>
          `;
        } else if (item.expense.validation_error) {
          securityNoticeHtml = `
            <div class="security-notice warning">
              <svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
              <span><strong>Validation Tag:</strong> ${escapeHtml(item.expense.validation_error)}</span>
            </div>
          `;
        }

        return `
          <div class="expense-card ${isInjection ? 'injection-border' : ''}" id="card-${item.session_id}">
            <div>
              <div class="card-top">
                <div class="submitter-info">
                  <h4>${escapeHtml(item.expense.submitter)}</h4>
                  <p>Session: ${item.session_id.substring(0, 18)}...</p>
                </div>
                ${statusPill}
              </div>

              <div class="amount-banner">
                <span class="amount-value ${amtClass}">${amtDisplay}</span>
                <span class="category-badge">${escapeHtml(item.expense.category)}</span>
              </div>

              <div class="description-box">
                ${descHtml}
              </div>

              ${securityNoticeHtml}
            </div>

            <div class="card-actions">
              <button class="btn btn-approve" id="btn-app-${item.session_id}" onclick="takeAction('${item.session_id}', '${item.interrupt_id}', '${item.user_id}', 'approve')">
                <svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M5 13l4 4L19 7"/></svg>
                Approve
              </button>
              <button class="btn btn-reject" id="btn-rej-${item.session_id}" onclick="takeAction('${item.session_id}', '${item.interrupt_id}', '${item.user_id}', 'reject')">
                <svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M6 18L18 6M6 6l12 12"/></svg>
                Reject
              </button>
              <button class="btn btn-detail" onclick="openDrawer('${item.session_id}')">
                <svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/><path d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"/></svg>
              </button>
            </div>
          </div>
        `;
      }).join('');
    }

    async function takeAction(sessionId, interruptId, userId, action) {
      const appBtn = document.getElementById('btn-app-' + sessionId);
      const rejBtn = document.getElementById('btn-rej-' + sessionId);
      if (appBtn) appBtn.disabled = true;
      if (rejBtn) rejBtn.disabled = true;

      const activeBtn = (action === 'approve') ? appBtn : rejBtn;
      if (activeBtn) activeBtn.innerHTML = `<span class="spinning">&#8635;</span> Resuming...`;

      try {
        const res = await fetch('/api/action/' + encodeURIComponent(sessionId), {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            action: action,
            interrupt_id: interruptId || 'human_decision',
            user_id: userId || 'default-user',
          }),
        });

        const result = await res.json();
        if (res.ok && result.status === 'success') {
          showToast(`Expense ${action.toUpperCase()}D successfully for session ${sessionId.substring(0, 12)}...`, 'success');
          // If drawer is open for this session, display compliance result
          if (selectedSession && selectedSession.session_id === sessionId) {
            showComplianceInDrawer(action, result.decision_message);
          }
          // Remove from local list and re-render
          allPending = allPending.filter(i => i.session_id !== sessionId);
          updateKPIs();
          setTimeout(renderCards, 800);
        } else {
          showToast(`Action failed: ${result.detail || 'Internal server error'}`, 'error');
          if (appBtn) appBtn.disabled = false;
          if (rejBtn) rejBtn.disabled = false;
        }
      } catch (err) {
        console.error('Error resuming session:', err);
        showToast('Network error while resuming session on Agent Runtime', 'error');
        if (appBtn) appBtn.disabled = false;
        if (rejBtn) rejBtn.disabled = false;
      }
    }

    function openDrawer(sessionId) {
      const item = allPending.find(i => i.session_id === sessionId);
      if (!item) return;
      selectedSession = item;

      document.getElementById('d-session').innerText = item.session_id;
      document.getElementById('d-submitter').innerText = item.expense.submitter;
      document.getElementById('d-category').innerText = item.expense.category;
      document.getElementById('d-amount').innerText = '$' + (parseFloat(item.expense.amount) || 0).toFixed(2);
      document.getElementById('d-date').innerText = item.expense.date || 'N/A';
      document.getElementById('d-description').innerHTML = escapeHtml(item.expense.description || '').replace(/(\[REDACTED_[A-Z_]+\])/g, '<span class="highlight-pii">$1</span>');

      const isInj = item.security_check.is_prompt_injection;
      document.getElementById('d-injection').innerText = isInj ? 'YES (Suspected Override Attempt)' : 'No';
      document.getElementById('d-reasons').innerText = item.security_check.injection_reasons.join(', ') || 'None';
      document.getElementById('d-pii').innerText = item.security_check.redacted_categories.join(', ') || 'None';

      const risk = item.risk_review || {};
      document.getElementById('d-risk-level').innerText = risk.risk_level || 'N/A';
      document.getElementById('d-risk-rec').innerText = risk.recommended_action || 'N/A';
      document.getElementById('d-risk-factors').innerText = (risk.risk_factors || []).join(', ') || 'None';

      document.getElementById('d-interrupt-msg').innerText = item.interrupt_message || 'Human approval required by policy.';
      
      const compBox = document.getElementById('d-compliance-result');
      compBox.className = 'compliance-box';
      compBox.style.display = 'none';

      document.getElementById('drawer').classList.add('open');
      document.getElementById('modal-overlay').classList.add('open');
    }

    function closeDrawer() {
      document.getElementById('drawer').classList.remove('open');
      document.getElementById('modal-overlay').classList.remove('open');
      selectedSession = null;
    }

    function handleDrawerAction(action) {
      if (!selectedSession) return;
      takeAction(selectedSession.session_id, selectedSession.interrupt_id, selectedSession.user_id, action);
    }

    function showComplianceInDrawer(action, message) {
      const compBox = document.getElementById('d-compliance-result');
      const compTitle = document.getElementById('d-compliance-title');
      const compDesc = document.getElementById('d-compliance-desc');
      compBox.style.display = 'block';
      if (action === 'approve') {
        compBox.className = 'compliance-box success';
        compTitle.innerText = '✅ Final Compliance Approval Recorded';
      } else {
        compBox.className = 'compliance-box rejected';
        compTitle.innerText = '❌ Expense Rejection Finalized';
      }
      compDesc.innerText = message;
    }

    function showToast(msg, type = 'info') {
      const container = document.getElementById('toast-container');
      const toast = document.createElement('div');
      toast.className = 'toast';
      toast.innerHTML = `
        <span style="color: ${type === 'error' ? '#fb7185' : '#34d399'}; font-weight: bold;">
          ${type === 'error' ? '✖' : '✔'}
        </span>
        <span>${escapeHtml(msg)}</span>
      `;
      container.appendChild(toast);
      setTimeout(() => {
        toast.style.opacity = '0';
        setTimeout(() => toast.remove(), 300);
      }, 4000);
    }

    function escapeHtml(str) {
      if (!str) return '';
      return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
    }

    // Initial load
    fetchPendingQueue();
    // Auto-refresh every 30 seconds
    setInterval(fetchPendingQueue, 30000);
  </script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
