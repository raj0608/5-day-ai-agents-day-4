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
    """Handles Agent Platform Playground queries sent to /api/stream_reasoning_engine."""
    try:
        body = await request.json()
    except Exception:
        body = {}

    # Extract user input query text from Playground payload formats
    input_data = body.get("input", body.get("message", body.get("query", body)))
    if isinstance(input_data, dict):
        input_text_str = str(
            input_data.get("message", input_data.get("data", json.dumps(input_data)))
        )
    else:
        input_text_str = str(input_data)

    user_id = str(body.get("user_id", "playground-user"))
    session_id = str(body.get("session_id", f"playground-{int(time.time())}"))

    runner: Runner = app.state.runner
    try:
        session = await runner.session_service.get_session(
            app_name=adk_app.name,
            user_id=user_id,
            session_id=session_id,
        )
    except Exception:
        session = None

    if not session:
        session = await runner.session_service.create_session(
            app_name=adk_app.name,
            user_id=user_id,
            session_id=session_id,
        )

    input_text = json.dumps({"data": input_text_str})
    user_content = types.Content(
        role="user", parts=[types.Part.from_text(text=input_text)]
    )

    async def json_stream() -> AsyncIterator[str]:
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session.id,
            new_message=user_content,
        ):
            event_payload: dict[str, Any] = {}
            if hasattr(event, "content") and event.content:
                event_payload["content"] = event.content.model_dump(
                    mode="json", exclude_none=True
                )
            if hasattr(event, "output") and event.output:
                event_payload["output"] = event.output
            if hasattr(event, "author") and event.author:
                event_payload["author"] = event.author

            yield f"{json.dumps(event_payload)}\n"

    return StreamingResponse(json_stream(), media_type="application/json")


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Ambient Expense Approval Service on port 8080...")
    uvicorn.run(app, host="0.0.0.0", port=8080)
