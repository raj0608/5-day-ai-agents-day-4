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

"""Ambient Expense Approval Web Service accepting Pub/Sub event triggers."""

import contextlib
import json
import logging
import os
import re
import time
from collections.abc import AsyncIterator
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from google.adk.cli.fast_api import get_fast_api_app
from google.adk.runners import Runner
from google.genai import types

from app.agent import app as adk_app
from app.app_utils import services
from expense_agent.agent import scrub_pii

load_dotenv()
if os.getenv("GEMINI_API_KEY"):
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "false"

# Standard Python console logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ambient_expense_agent")

AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Lifespan context manager initializing the ADK runner."""
    runner = Runner(
        app=adk_app,
        session_service=services.get_session_service(),
        artifact_service=services.get_artifact_service(),
        auto_create_session=True,
    )
    app.state.runner = runner
    app.state.agent_app_name = adk_app.name
    logger.info("ADK Ambient Runner initialized successfully.")
    yield


# Initialize FastAPI app using ADK helper with otel_to_cloud=False per requirements
app: FastAPI = get_fast_api_app(
    agents_dir=AGENT_DIR,
    web=True,
    artifact_service_uri=services.ARTIFACT_SERVICE_URI,
    session_service_uri=services.SESSION_SERVICE_URI,
    otel_to_cloud=False,  # Telemetry requirement: otel_to_cloud=False
    lifespan=lifespan,
)
app.title = "Ambient Expense Approval Service"
app.description = (
    "Ambient web service processing GCP Pub/Sub triggers for expense workflow approvals"
)


def normalize_subscription_name(raw_subscription: str | None) -> str:
    """Normalize fully-qualified GCP Pub/Sub subscription path to a short name.

    Example:
        'projects/my-gcp-project/subscriptions/expense-approval-sub'
        => 'expense-approval-sub'
    """
    if not raw_subscription:
        return "expense-sub"
    # Extract short subscription identifier from path
    short_name = raw_subscription.split("/")[-1]
    return short_name if short_name else "expense-sub"


@app.get("/health")
def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "service": "ambient-expense-agent", "port": "8080"}


@app.get("/")
def root_welcome() -> dict[str, Any]:
    """Welcome endpoint for browser GET requests."""
    return {
        "service": "Ambient Expense Approval Service",
        "status": "running",
        "port": 8080,
        "endpoints": {
            "health_check": "GET /health",
            "interactive_docs": "GET /docs",
            "pubsub_trigger": "POST /pubsub or POST /",
        },
        "usage": "Send Pub/Sub push notification payloads via POST to /pubsub or /",
    }


@app.post("/")
@app.post("/pubsub")
async def process_pubsub_event(request: Request) -> dict[str, Any]:
    """Pub/Sub push event trigger handler endpoint.

    Accepts GCP Pub/Sub push messages, extracts data, normalizes subscription path,
    and feeds the expense report into the ADK 2.0 graph workflow.
    """
    try:
        body = await request.json()
    except Exception as err:
        logger.error(f"Failed to parse request JSON body: {err}")
        raise HTTPException(status_code=400, detail="Invalid JSON body") from err

    # Redact PII from raw request payload before writing to server logs
    raw_body_str = json.dumps(body)
    clean_body_str, _ = scrub_pii(raw_body_str)
    logger.info(f"Received raw event payload: {clean_body_str}")

    # Extract Pub/Sub message body and subscription path
    message = body.get("message", body)
    raw_subscription = body.get("subscription")

    # Gotcha handling: Normalize fully-qualified subscription path down to short name
    short_subscription = normalize_subscription_name(raw_subscription)
    message_id = (
        message.get("messageId", "msg-001") if isinstance(message, dict) else "msg-001"
    )

    # Construct readable session ID using normalized subscription name
    session_id = f"{short_subscription}-{message_id}"
    user_id = f"pubsub-subscriber-{short_subscription}"

    # Extract message data payload (base64 string or dict)
    message_data = (
        message.get("data", message) if isinstance(message, dict) else message
    )

    logger.info(
        f"Normalized subscription: '{short_subscription}', Session ID: '{session_id}'"
    )

    runner: Runner = app.state.runner
    session = await runner.session_service.create_session(
        app_name=adk_app.name,
        user_id=user_id,
        session_id=session_id,
    )

    # Input message passed to workflow
    input_text = json.dumps({"data": message_data})
    user_content = types.Content(
        role="user", parts=[types.Part.from_text(text=input_text)]
    )

    outputs = []
    contents = []

    async for event in runner.run_async(
        user_id=user_id,
        session_id=session.id,
        new_message=user_content,
    ):
        if event.output is not None:
            outputs.append(event.output)
            logger.info(f"Workflow event output: {event.output}")

        # Extract text messages or RequestInput HITL interrupt messages
        if event.content and event.content.parts:
            for part in event.content.parts:
                if getattr(part, "text", None):
                    contents.append(part.text)
                    logger.info(f"Workflow content message: {part.text}")
                elif getattr(part, "function_call", None):
                    fc = part.function_call
                    if getattr(fc, "name", "") == "adk_request_input" and isinstance(
                        fc.args, dict
                    ):
                        msg = fc.args.get("message")
                        if msg:
                            contents.append(msg)
                            logger.info(f"RequestInput interrupt prompt: {msg}")

    # Filter outputs to show decision status if present, avoiding duplicate raw inputs
    final_outputs = [o for o in outputs if isinstance(o, dict) and "status" in o]
    if not final_outputs and outputs:
        final_outputs = [outputs[-1]]

    return {
        "status": "processed",
        "subscription": short_subscription,
        "session_id": session.id,
        "outputs": final_outputs,
        "messages": contents,
    }


@app.post("/stream_reasoning_engine")
@app.post("/reasoning_engine")
@app.post("/api/stream_reasoning_engine")
@app.post("/api/reasoning_engine")
@app.post("/run_sse")
@app.post("/run")
async def handle_reasoning_engine_playground(request: Request) -> Any:
    """Handles Agent Platform Playground queries sent to /api/stream_reasoning_engine, /run_sse, etc."""
    try:
        body = await request.json()
    except Exception:
        body = {}

    # Probe / health check handling: if body is empty or null, return immediately
    if not body:
        return {"status": "ok", "service": "ambient-expense-agent"}

    # Extract user_id and session_id from body or input dictionary (supports camelCase and snake_case)
    input_dict = body.get("input") if isinstance(body.get("input"), dict) else {}

    user_id = str(
        body.get("userId")
        or body.get("user_id")
        or input_dict.get("userId")
        or input_dict.get("user_id")
        or "playground-user"
    )
    raw_session_id = str(
        body.get("sessionId")
        or body.get("session_id")
        or input_dict.get("sessionId")
        or input_dict.get("session_id")
        or f"playground-{int(time.time())}"
    )
    # Sanitize session_id to conform to Vertex AI rules: lowercase letters, digits, and hyphens only
    session_id = re.sub(r"[^a-z0-9-]+", "-", raw_session_id.lower()).strip("-")
    if not session_id:
        session_id = f"session-{int(time.time())}"

    # Extract user input query text or structured resume message
    raw_message = (
        body.get("newMessage")
        or input_dict.get("newMessage")
        or input_dict.get("message")
        or body.get("message")
        or body.get("input")
        or body.get("query")
        or body
    )

    user_content: types.Content | None = None

    # Check if raw_message is an ADK Content / function_response envelope
    if isinstance(raw_message, dict) and ("parts" in raw_message or "role" in raw_message):
        try:
            user_content = types.Content.model_validate(raw_message)
            if not user_content.role:
                user_content.role = "user"
        except Exception as val_err:
            logger.warning("Could not model_validate Content directly: %s; constructing manually", val_err)
            if "parts" in raw_message and isinstance(raw_message["parts"], list):
                parts = []
                for p in raw_message["parts"]:
                    if isinstance(p, dict) and "function_response" in p:
                        fr = p["function_response"]
                        parts.append(
                            types.Part(
                                function_response=types.FunctionResponse(
                                    id=fr.get("id", "human_decision"),
                                    name=fr.get("name", "adk_request_input"),
                                    response=fr.get("response", {}),
                                )
                            )
                        )
                    elif isinstance(p, dict) and "text" in p:
                        parts.append(types.Part.from_text(text=str(p["text"])))
                user_content = types.Content(role=raw_message.get("role", "user"), parts=parts)

    is_resume = False
    if user_content and user_content.parts:
        for p in user_content.parts:
            if getattr(p, "function_response", None):
                is_resume = True
                break

    if user_content is None:
        if isinstance(raw_message, dict):
            input_text_str = str(
                raw_message.get("data", raw_message.get("message", json.dumps(raw_message)))
            )
        else:
            input_text_str = str(raw_message)

        input_text = json.dumps({"data": input_text_str})
        user_content = types.Content(
            role="user", parts=[types.Part.from_text(text=input_text)]
        )

    runner: Runner = app.state.runner

    # 1. Resolve session ownership: if the session exists in VertexAiSessionService,
    # adopt the session's actual owner user_id (e.g. 'vais-query-reasoning-engine' or authenticated user)
    # to avoid 400 ALREADY_EXISTS / ValueError ownership mismatch errors.
    if hasattr(runner.session_service, "_get_api_client"):
        try:
            engine_id = (
                getattr(runner.session_service, "_agent_engine_id", None)
                or (
                    runner.session_service._get_reasoning_engine_id(adk_app.name)
                    if hasattr(runner.session_service, "_get_reasoning_engine_id")
                    else None
                )
            )
            if engine_id:
                async with runner.session_service._get_api_client() as api_client:
                    res_name = f"reasoningEngines/{engine_id}/sessions/{session_id}"
                    try:
                        sess_res = await api_client.agent_engines.sessions.get(name=res_name)
                        if sess_res and isinstance(getattr(sess_res, "user_id", None), str):
                            user_id = sess_res.user_id
                            logger.info(
                                f"Adopting existing session owner user_id '{user_id}' for session '{session_id}'"
                            )
                    except Exception:
                        pass
        except Exception as owner_err:
            logger.debug(f"Error checking session owner on Vertex AI: {owner_err}")

    # 2. Get existing session or create a new session using the exact target session_id.
    # Always preserve session_id across multi-turn chats so subsequent messages stay in the same session.
    try:
        session = await runner.session_service.get_session(
            app_name=adk_app.name,
            user_id=user_id,
            session_id=session_id,
        )
    except Exception as get_err:
        logger.info(
            f"get_session failed for '{session_id}' with user_id '{user_id}': {get_err}"
        )
        session = None

    if not session:
        try:
            session = await runner.session_service.create_session(
                app_name=adk_app.name,
                user_id=user_id,
                session_id=session_id,
            )
        except Exception as create_err:
            logger.warning(
                f"create_session failed for '{session_id}': {create_err}. Retrying get_session with default owner."
            )
            # Session already exists under another user_id that couldn't be fetched prior
            session = await runner.session_service.get_session(
                app_name=adk_app.name,
                user_id="vais-query-reasoning-engine",
                session_id=session_id,
            )
            if session:
                user_id = "vais-query-reasoning-engine"

    async def json_stream() -> AsyncIterator[str]:
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session.id,
            new_message=user_content,
        ):
            # Check for actual user-facing text content
            has_text = False
            if hasattr(event, "content") and event.content and event.content.parts:
                for p in event.content.parts:
                    if getattr(p, "text", None):
                        has_text = True
                        break

            # Suppress internal node execution events that have no user-facing text.
            # This completely eliminates empty assistant chat bubbles (#2, #5) in the Playground UI.
            if not has_text:
                continue

            has_status = (
                hasattr(event, "output")
                and isinstance(event.output, dict)
                and "status" in event.output
            )

            event_payload: dict[str, Any] = {}
            if hasattr(event, "content") and event.content:
                event_payload["content"] = event.content.model_dump(
                    mode="json", exclude_none=True
                )
            if has_status and hasattr(event, "output") and event.output:
                event_payload["output"] = event.output
            if hasattr(event, "author") and event.author:
                event_payload["author"] = event.author

            yield f"{json.dumps(event_payload)}\n"

    return StreamingResponse(json_stream(), media_type="application/json")


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Ambient Expense Approval Service on port 8080...")
    uvicorn.run(app, host="0.0.0.0", port=8080)
