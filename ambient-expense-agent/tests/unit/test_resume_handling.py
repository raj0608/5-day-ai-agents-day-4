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

"""Test that handle_reasoning_engine_playground properly extracts session_id and resumes without creating new sessions."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.requests import Request
from google.genai import types


@pytest.mark.asyncio
async def test_resume_function_response_preserves_session_id():
    """Verify that handle_reasoning_engine_playground preserves session_id and passes function_response to runner."""
    from app.fast_api_app import handle_reasoning_engine_playground, app

    # Build mock request body as sent by Vertex AI Reasoning Engine stream_query
    payload = {
        "class_method": "stream_query",
        "input": {
            "user_id": "default-user",
            "session_id": "existing-session-abc",
            "message": {
                "role": "user",
                "parts": [
                    {
                        "function_response": {
                            "id": "human_decision",
                            "name": "adk_request_input",
                            "response": {
                                "approved": True,
                                "human_decision": "approve",
                            },
                        }
                    }
                ],
            },
        },
    }

    # Mock request
    mock_req = MagicMock(spec=Request)
    mock_req.json = AsyncMock(return_value=payload)

    # Mock runner
    mock_runner = MagicMock()
    mock_session = MagicMock()
    mock_session.id = "existing-session-abc"
    mock_runner.session_service.get_session = AsyncMock(return_value=mock_session)

    captured_args = {}

    async def mock_run_async(user_id, session_id, new_message):
        captured_args["user_id"] = user_id
        captured_args["session_id"] = session_id
        captured_args["new_message"] = new_message
        yield MagicMock(content=None, output={"status": "HUMAN_APPROVED"}, author=None)

    mock_runner.run_async = mock_run_async
    app.state.runner = mock_runner

    # Invoke handler
    response = await handle_reasoning_engine_playground(mock_req)
    assert response is not None

    # Consume the streaming response generator
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk)

    # Assertions
    assert captured_args["session_id"] == "existing-session-abc"
    assert captured_args["user_id"] == "default-user"

    # Verify that new_message is a types.Content with a function_response part (NOT a text string!)
    msg = captured_args["new_message"]
    assert isinstance(msg, types.Content)
    assert len(msg.parts) == 1
    assert msg.parts[0].function_response is not None
    assert msg.parts[0].function_response.name == "adk_request_input"
    assert msg.parts[0].function_response.response == {
        "approved": True,
        "human_decision": "approve",
    }


@pytest.mark.asyncio
async def test_playground_expense_query_creates_user_text():
    """Verify that regular user text or JSON payload from playground is correctly wrapped."""
    from app.fast_api_app import handle_reasoning_engine_playground, app

    payload = {
        "input": {
            "message": json.dumps({"amount": 45.50, "submitter": "alice@example.com", "category": "Meals"}),
        }
    }

    mock_req = MagicMock(spec=Request)
    mock_req.json = AsyncMock(return_value=payload)

    mock_runner = MagicMock()
    mock_session = MagicMock()
    mock_session.id = "new-session"
    mock_runner.session_service.get_session = AsyncMock(return_value=None)
    mock_runner.session_service.create_session = AsyncMock(return_value=mock_session)

    captured_args = {}

    async def mock_run_async(user_id, session_id, new_message):
        captured_args["user_id"] = user_id
        captured_args["session_id"] = session_id
        captured_args["new_message"] = new_message
        yield MagicMock(content=None, output={"status": "AUTO_APPROVED"}, author=None)

    mock_runner.run_async = mock_run_async
    app.state.runner = mock_runner

    response = await handle_reasoning_engine_playground(mock_req)
    assert response is not None

    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk)

    msg = captured_args["new_message"]
    assert isinstance(msg, types.Content)
    assert len(msg.parts) == 1
    assert msg.parts[0].text is not None
    parsed = json.loads(msg.parts[0].text)
    assert "data" in parsed

