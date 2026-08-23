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

import json
from unittest.mock import MagicMock

from expense_agent.agent import parse_expense_node, parse_expense_payload


def test_double_wrapped_python_repr_envelope_unwraps_and_routes_correctly():
    """Test feeding parse_expense_payload and parse_expense_node the exact double-wrapped Python-repr envelope verbatim.

    Envelope structure:
    {"data": "{'parts': [{'text': '<the JSON I typed>'}], 'role': 'user'}"}
    """
    json_inner = json.dumps(
        {
            "amount": 100.00,
            "submitter": "bob@example.com",
            "category": "Travel",
            "description": "Taxi",
            "date": "2026-08-23",
        }
    )
    # Construct exact Python-repr string envelope with single quotes
    python_repr_str = f"{{'parts': [{{'text': '{json_inner}\\n'}}], 'role': 'user'}}"
    double_wrapped_payload = {"data": python_repr_str}

    expense, _categories = parse_expense_payload(double_wrapped_payload)

    # 1. Unwrapping check
    assert expense.amount == 100.00
    assert expense.submitter == "bob@example.com"

    # 2. Routing check: amount == 100.00 >= 100.0 MUST route to security_check
    ctx = MagicMock()
    ctx.state = {}
    event = parse_expense_node(ctx, double_wrapped_payload)
    assert event.actions.route == "security_check"
