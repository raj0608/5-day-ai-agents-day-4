import base64
import json

import pytest

from expense_agent.agent import ParseFailure, parse_expense_payload, scrub_pii
from expense_agent.schemas import ExpenseReport


def test_scrub_pii_ssn():
    text = "John Doe expense with SSN 123-45-6789 for dinner"
    sanitized, categories = scrub_pii(text)
    assert "[REDACTED_SSN]" in sanitized
    assert "123-45-6789" not in sanitized
    assert "SSN" in categories


def test_scrub_pii_credit_card():
    text = "Card payment 4532-1234-5678-9012 for hotel"
    sanitized, categories = scrub_pii(text)
    assert "[REDACTED_CREDIT_CARD]" in sanitized
    assert "4532-1234-5678-9012" not in sanitized
    assert "CREDIT_CARD" in categories


def test_parse_expense_payload_json():
    raw = json.dumps(
        {
            "amount": 120.50,
            "submitter": "alice@example.com",
            "category": "Travel",
            "description": "Flight booking SSN 987-65-4321",
            "date": "2026-08-23",
        }
    )
    expense, categories = parse_expense_payload(raw)
    assert isinstance(expense, ExpenseReport)
    assert expense.amount == 120.50
    assert expense.submitter == "alice@example.com"
    assert "[REDACTED_SSN]" in expense.description
    assert "SSN" in categories


def test_parse_expense_payload_base64_pubsub():
    payload_dict = {
        "amount": 45.00,
        "submitter": "bob@example.com",
        "category": "Meals",
        "description": "Team lunch",
        "date": "2026-08-23",
    }
    encoded = base64.b64encode(json.dumps(payload_dict).encode("utf-8")).decode("utf-8")
    expense, _categories = parse_expense_payload(encoded)
    assert expense.amount == 45.00
    assert expense.submitter == "bob@example.com"


def test_parse_expense_payload_invalid_type():
    with pytest.raises(ParseFailure):
        parse_expense_payload(12345)
