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

"""Automated tests for Manager Dashboard FastAPI service and endpoints."""

from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_dashboard_html_rendering(client):
    """Test GET / renders dashboard HTML with Google Fonts, glassmorphism, drawer, and action buttons."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    html = response.text

    # Typography check
    assert "Outfit" in html
    assert "Inter" in html

    # Styling and elements check
    assert "backdrop-filter" in html
    assert "Manager Approval Portal" in html
    assert "cards-container" in html
    assert "modal-overlay" in html
    assert "drawer" in html
    assert "btn-approve" in html
    assert "btn-reject" in html
    assert "Compliance Review & Audit Detail" in html


def test_health_endpoint(client):
    """Test GET /api/health endpoint."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "project" in data
    assert "agent_runtime_id" in data


@pytest.mark.asyncio
async def test_api_pending_identification():
    """Test GET /api/pending identifies unresolved adk_request_input and excludes resolved calls."""
    with patch("main.session_service") as mock_session_service:
        # Mock two sessions:
        # Session 1: has unresolved adk_request_input
        # Session 2: has adk_request_input that is already resolved with function_response

        mock_session_info1 = MagicMock()
        mock_session_info1.id = "session-unresolved-123"
        mock_session_info1.user_id = "default-user"
        mock_session_info1.last_update_time = 2000000000.0

        mock_session_info2 = MagicMock()
        mock_session_info2.id = "session-resolved-456"
        mock_session_info2.user_id = "default-user"
        mock_session_info2.last_update_time = 2000000000.0

        mock_list_resp = MagicMock()
        mock_list_resp.sessions = [mock_session_info1, mock_session_info2]
        mock_session_service.list_sessions = AsyncMock(return_value=mock_list_resp)

        # Full session 1 (unresolved)
        full_s1 = MagicMock()
        full_s1.id = "session-unresolved-123"
        full_s1.user_id = "default-user"
        full_s1.last_update_time = 2000000000.0
        full_s1.state = {
            "expense": {
                "amount": 450.00,
                "submitter": "bob@example.com",
                "category": "Equipment",
                "description": "Developer monitor",
                "date": "2026-08-23",
            },
            "security_check": {
                "is_prompt_injection": False,
                "injection_reasons": [],
                "redacted_categories": [],
            },
            "risk_review": {
                "risk_level": "MEDIUM",
                "risk_factors": ["High dollar amount"],
                "recommended_action": "REVIEW",
            },
        }

        # Event with adk_request_input call
        fc_part1 = MagicMock()
        fc_part1.function_call.name = "adk_request_input"
        fc_part1.function_call.id = "human_decision"
        fc_part1.function_call.args = {"message": "Human approval required for expense >= $100.00"}
        fc_part1.function_response = None

        ev1 = MagicMock()
        ev1.content.parts = [fc_part1]
        ev1.long_running_tool_ids = ["human_decision"]
        full_s1.events = [ev1]

        # Full session 2 (resolved)
        full_s2 = MagicMock()
        full_s2.id = "session-resolved-456"
        full_s2.user_id = "default-user"
        full_s2.last_update_time = 2000000000.0
        full_s2.state = {"expense": {"amount": 200.0}}

        fc_part2 = MagicMock()
        fc_part2.function_call.name = "adk_request_input"
        fc_part2.function_call.id = "human_decision"
        fc_part2.function_call.args = {"message": "Approval required"}
        fc_part2.function_response = None

        resp_part2 = MagicMock()
        resp_part2.function_call = None
        resp_part2.function_response.name = "adk_request_input"
        resp_part2.function_response.id = "human_decision"

        ev2_call = MagicMock()
        ev2_call.content.parts = [fc_part2]
        ev2_call.long_running_tool_ids = ["human_decision"]

        ev2_resp = MagicMock()
        ev2_resp.content.parts = [resp_part2]
        ev2_resp.long_running_tool_ids = None

        full_s2.events = [ev2_call, ev2_resp]

        async def get_session_mock(app_name, user_id, session_id):
            if session_id == "session-unresolved-123":
                return full_s1
            return full_s2

        mock_session_service.get_session = AsyncMock(side_effect=get_session_mock)

        # Call endpoint
        client = TestClient(app)
        response = client.get("/api/pending")
        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "success"
        assert data["count"] == 1
        pending = data["pending"]
        assert len(pending) == 1
        item = pending[0]
        assert item["session_id"] == "session-unresolved-123"
        assert item["interrupt_id"] == "human_decision"
        assert item["expense"]["amount"] == 450.00
        assert item["expense"]["submitter"] == "bob@example.com"
        assert item["risk_review"]["risk_level"] == "MEDIUM"


@pytest.mark.asyncio
async def test_api_action_resume_payload_format():
    """Test POST /api/action/{session_id} passes the exact resume payload format with user_id strictly default-user."""
    client = TestClient(app)

    with patch("main.reasoning_engines.ReasoningEngine") as MockEngine:
        mock_engine_instance = MagicMock()
        MockEngine.return_value = mock_engine_instance
        mock_engine_instance.resource_name = "projects/test/locations/us-east1/reasoningEngines/4144947730482987008"

        # Mock stream_query_reasoning_engine response
        mock_chunk = MagicMock()
        # mock parsed json iterator
        with patch("main._utils.yield_parsed_json", return_value=[
            {
                "output": {"status": "HUMAN_APPROVED", "decision": "Approved by manager"},
                "content": {"parts": [{"text": "✅ EXPENSE DECISION [HUMAN_APPROVED]: Approved by human reviewer for $450.00 expense."}]}
            }
        ]):
            mock_engine_instance.execution_api_client.stream_query_reasoning_engine.return_value = [mock_chunk]

            response = client.post(
                "/api/action/test-session-789",
                json={
                    "action": "approve",
                    "interrupt_id": "human_decision",
                    "user_id": "default-user",
                }
            )

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert data["session_id"] == "test-session-789"
            assert "Approved" in data["decision_message"]

            # Verify the SDK request parameters
            called_req = mock_engine_instance.execution_api_client.stream_query_reasoning_engine.call_args[1]["request"]
            assert called_req.class_method == "stream_query"
            assert called_req.input["user_id"] == "default-user"
            assert called_req.input["session_id"] == "test-session-789"

            # Verify exact resume payload structure
            msg = called_req.input["message"]
            assert msg["role"] == "user"
            assert len(msg["parts"]) == 1
            fn_resp = msg["parts"][0]["function_response"]
            assert fn_resp["name"] == "adk_request_input"
            assert fn_resp["id"] == "human_decision"
            assert fn_resp["response"]["approved"] is True
