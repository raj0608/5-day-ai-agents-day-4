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

"""ADK 2.0 Graph Workflow for Ambient Expense Approval with Security Controls."""

import base64
import json
import os
import re
from typing import Any, AsyncGenerator

from dotenv import load_dotenv

load_dotenv()
if os.getenv("GEMINI_API_KEY"):
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "false"

from google.adk.agents import LlmAgent
from google.adk.agents.context import Context
from google.adk.apps import App, ResumabilityConfig
from google.adk.events.event import Event
from google.adk.events.request_input import RequestInput
from google.adk.workflow import START, Workflow
from google.genai import types

from . import config
from .schemas import ExpenseReport, RiskReview, SecurityCheckResult


def parse_expense_payload(raw_input: Any) -> tuple[ExpenseReport, list[str]]:
    """Parse raw payload from plain JSON or Base64 Pub/Sub event format, immediately scrubbing PII."""
    text = None
    data = None

    if hasattr(raw_input, "parts") and raw_input.parts:
        text = raw_input.parts[0].text
    elif isinstance(raw_input, str):
        text = raw_input
    elif isinstance(raw_input, dict):
        data = raw_input
    else:
        data = {"data": raw_input}

    if text:
        try:
            data = json.loads(text)
        except Exception:
            data = {"data": text}

    payload = data.get("data", data) if isinstance(data, dict) else data

    # Decode Base64 string if Pub/Sub event format is present
    if isinstance(payload, str):
        try:
            decoded_bytes = base64.b64decode(payload)
            payload = json.loads(decoded_bytes.decode("utf-8"))
        except Exception:
            try:
                payload = json.loads(payload)
            except Exception:
                pass

    if isinstance(payload, dict):
        raw_desc = str(payload.get("description", ""))
        sanitized_desc, redacted_categories = scrub_pii(raw_desc)
        expense = ExpenseReport(
            amount=float(payload.get("amount", 0.0)),
            submitter=str(payload.get("submitter", "Unknown")),
            category=str(payload.get("category", "General")),
            description=sanitized_desc,
            date=str(payload.get("date", "")),
        )
        return expense, redacted_categories
    raise ValueError(f"Could not parse valid expense report payload from input: {raw_input}")


def scrub_pii(text: str) -> tuple[str, list[str]]:
    """Scrub SSN and Credit Card numbers from text, returning sanitized text and list of redacted categories."""
    redacted_categories = []

    # 1. Standard 9-digit SSN format: XXX-XX-XXXX, XXX XX XXXX, or 9 raw digits
    ssn_std = r"\b\d{3}[-\s]\d{2}[-\s]\d{4}\b|\b\d{9}\b"
    if re.search(ssn_std, text):
        redacted_categories.append("SSN")
        text = re.sub(ssn_std, "[REDACTED_SSN]", text)

    # 2. SSN keyword followed by continuous digits or spaced/hyphenated numbers
    ssn_kw = r"(?i)\bssn\b\s*(?:number|id|#)?\s*(?:is|=|:)?\s*[\d\s-]{3,15}\b"
    if re.search(ssn_kw, text):
        if "SSN" not in redacted_categories:
            redacted_categories.append("SSN")
        text = re.sub(ssn_kw, "SSN [REDACTED_SSN]", text)

    # 3. Credit Card pattern (matches 13 to 19 digit card sequences)
    cc_pattern = r"\b(?:\d[ -]*?){13,19}\b"
    if re.search(cc_pattern, text):
        if "CREDIT_CARD" not in redacted_categories:
            redacted_categories.append("CREDIT_CARD")
        text = re.sub(cc_pattern, "[REDACTED_CREDIT_CARD]", text)

    return text, redacted_categories


def detect_prompt_injection(text: str) -> tuple[bool, list[str]]:
    """Detect prompt injection attempt indicators targeting LLM instruction override or policy bypass."""
    injection_patterns = [
        (r"bypass\b", "Policy bypass attempt"),
        (r"ignore\b.*?\b(instruction|rule|system|prompt)\b", "Instruction override attempt"),
        (r"override\b", "System override attempt"),
        (r"auto[- ]?approve\b", "Forced auto-approval attempt"),
        (r"system\s*prompt", "System prompt reference"),
        (r"disregard\b", "Disregard instruction attempt"),
        (r"say\s+approved", "Forced approval instruction"),
    ]
    detected = []
    text_lower = text.lower()
    for pattern, reason in injection_patterns:
        if re.search(pattern, text_lower):
            detected.append(reason)
    return (len(detected) > 0), detected


def parse_expense_node(ctx: Context, node_input: Any) -> Event:
    """Node 1: Parse input expense, immediately scrub PII, store state, and route to auto_approve or security_check."""
    if ctx.state.get("expense"):
        expense = ExpenseReport(**ctx.state["expense"])
        redacted_categories = ctx.state.get("redacted_categories", [])
    else:
        expense, redacted_categories = parse_expense_payload(node_input)

    # Route decision based on configurable dollar threshold
    if expense.amount < config.AUTO_APPROVE_THRESHOLD:
        route = "auto_approve"
    else:
        route = "security_check"

    return Event(
        output=expense.model_dump(),
        route=route,
        state={
            "expense": expense.model_dump(),
            "redacted_categories": redacted_categories,
        },
    )


def security_checkpoint_node(ctx: Context, node_input: dict[str, Any]) -> Event:
    """Node 2: Security Checkpoint. Performs prompt injection defense and defense-in-depth second pass PII check."""
    expense_data = ctx.state.get("expense", node_input)
    raw_desc = str(expense_data.get("description", ""))
    existing_redacted = ctx.state.get("redacted_categories", [])

    # Defense-in-depth second pass PII check
    sanitized_desc, new_redacted = scrub_pii(raw_desc)
    expense_data["description"] = sanitized_desc
    redacted_categories = list(dict.fromkeys(existing_redacted + new_redacted))

    # Check for Prompt Injection
    is_injection, injection_reasons = detect_prompt_injection(raw_desc)

    sec_result = SecurityCheckResult(
        is_prompt_injection=is_injection,
        injection_reasons=injection_reasons,
        redacted_categories=redacted_categories,
        sanitized_description=sanitized_desc,
    )

    state_delta = {
        "expense": expense_data,
        "security_check": sec_result.model_dump(),
        "redacted_categories": redacted_categories,
    }

    if is_injection:
        # Route directly to human review, bypassing the LLM
        return Event(
            output=expense_data,
            route="security_flagged",
            state=state_delta,
        )
    else:
        # Clean expense - continue to LLM reviewer
        return Event(
            output=expense_data,
            route="llm_review",
            state=state_delta,
        )


def auto_approve_node(node_input: dict[str, Any]) -> Event:
    """Node 3A: Auto-approve expenses under $100 instantly without LLM involvement."""
    expense = ExpenseReport(**node_input)
    result = {
        "status": "AUTO_APPROVED",
        "amount": expense.amount,
        "submitter": expense.submitter,
        "category": expense.category,
        "description": expense.description,
        "date": expense.date,
        "reason": (
            f"Expense amount (${expense.amount:.2f}) is under auto-approval "
            f"threshold (${config.AUTO_APPROVE_THRESHOLD:.2f}). Approved automatically."
        ),
    }
    content = types.Content(
        role="model",
        parts=[
            types.Part.from_text(
                text=(
                    f"⚡ AUTO-APPROVED: Expense of ${expense.amount:.2f} for '{expense.description}' "
                    f"submitted by {expense.submitter} was automatically approved."
                )
            )
        ],
    )
    return Event(output=result, content=content)


# Node 3B: LLM Risk Review for clean expenses >= $100
llm_risk_review = LlmAgent(
    name="llm_risk_review",
    model=config.MODEL_NAME,
    instruction=(
        "You are a financial risk evaluation assistant reviewing high-value expense reports.\n"
        "Evaluate the expense report details (submitter, amount, category, description, date).\n"
        "Identify potential risk factors (e.g. high dollar amount, unusual items, policy concerns).\n"
        "Generate a structured RiskReview output containing risk level, factors, alert summary, and recommendation."
    ),
    output_schema=RiskReview,
    output_key="risk_review",
)


async def human_approval_node(ctx: Context, node_input: dict[str, Any]) -> AsyncGenerator[Any, None]:
    """Node 4: Pause workflow with RequestInput for human approval on expenses >= $100 or security events."""
    expense_data = ctx.state.get("expense", {})
    sec_check = ctx.state.get("security_check", {})
    amount = expense_data.get("amount", 0.0)
    submitter = expense_data.get("submitter", "Unknown")
    description = expense_data.get("description", "")
    redacted_cats = ctx.state.get("redacted_categories", [])

    # Extract decision from ctx.resume_inputs when resuming
    human_reply = None
    if ctx.resume_inputs and "human_decision" in ctx.resume_inputs:
        human_reply = str(ctx.resume_inputs["human_decision"]).strip()
    elif ctx.resume_inputs:
        for v in ctx.resume_inputs.values():
            if isinstance(v, (str, dict)):
                human_reply = str(v).strip()
                break

    # Request human input if not yet answered
    if not human_reply:
        is_injection = sec_check.get("is_prompt_injection", False)

        if is_injection:
            # Prompt injection security alert layout (LLM bypassed)
            reasons = ", ".join(sec_check.get("injection_reasons", []))
            redacted_str = f"\n🔒 Redacted PII Categories: {', '.join(redacted_cats)}" if redacted_cats else ""

            message = (
                f"🚨 CRITICAL SECURITY ALERT: SUSPECTED PROMPT INJECTION DETECTED 🚨\n"
                f"--------------------------------------------------\n"
                f"The LLM reviewer was BYPASSED to defend system rules.\n\n"
                f"• Submitter: {submitter}\n"
                f"• Amount: ${amount:.2f}\n"
                f"• Category: {expense_data.get('category', 'N/A')}\n"
                f"• Cleaned Description: {description}\n"
                f"• Security Trigger: {reasons}{redacted_str}\n\n"
                f"Please inspect carefully and respond with 'approve' or 'reject'."
            )
        else:
            # Normal LLM Risk Review layout
            risk_alert = node_input.get("alert_summary", "High-value expense requires manual review.")
            risk_level = node_input.get("risk_level", "MEDIUM")
            risk_factors = ", ".join(node_input.get("risk_factors", [])) or "None identified"
            rec = node_input.get("recommended_action", "REVIEW")
            redacted_str = f"\n🔒 Redacted PII Categories: {', '.join(redacted_cats)}" if redacted_cats else ""

            message = (
                f"⚠️ HUMAN APPROVAL REQUIRED (Expense >= ${config.AUTO_APPROVE_THRESHOLD:.2f})\n"
                f"• Submitter: {submitter}\n"
                f"• Amount: ${amount:.2f}\n"
                f"• Category: {expense_data.get('category', 'N/A')}\n"
                f"• Description: {description}\n"
                f"• Date: {expense_data.get('date', 'N/A')}{redacted_str}\n\n"
                f"🔍 LLM Risk Review [{risk_level}]: {risk_alert}\n"
                f"• Risk Factors: {risk_factors}\n"
                f"• AI Recommendation: {rec}\n\n"
                f"Please respond with 'approve' or 'reject' to finalize this expense."
            )

        yield RequestInput(interrupt_id="human_decision", message=message)
        return

    # Process human approval/rejection response
    is_approved = "approve" in human_reply.lower()

    if sec_check.get("is_prompt_injection", False):
        status = "SECURITY_APPROVED" if is_approved else "SECURITY_REJECTED"
        decision_summary = "Human reviewer decision on security-flagged expense"
    else:
        status = "HUMAN_APPROVED" if is_approved else "HUMAN_REJECTED"
        decision_summary = "Approved by human reviewer" if is_approved else "Rejected by human reviewer"

    final_result = {
        "status": status,
        "decision": decision_summary,
        "human_response": human_reply,
        "expense": expense_data,
        "security_check": sec_check,
        "risk_review": node_input if not sec_check.get("is_prompt_injection") else None,
    }

    status_icon = "✅" if is_approved else "❌"
    content = types.Content(
        role="model",
        parts=[
            types.Part.from_text(
                text=f"{status_icon} EXPENSE DECISION [{status}]: {decision_summary} for ${amount:.2f} expense ('{description}' by {submitter})."
            )
        ],
    )
    yield Event(output=final_result, content=content)


# Construct the ADK 2.0 Workflow Graph with Security Controls
root_agent = Workflow(
    name="ambient_expense_agent",
    description=(
        "Ambient expense approval graph workflow with threshold auto-approval, "
        "PII redaction, prompt injection defense, and LLM risk review with HITL approval."
    ),
    edges=[
        (START, parse_expense_node),
        (parse_expense_node, {
            "auto_approve": auto_approve_node,
            "security_check": security_checkpoint_node,
        }),
        (security_checkpoint_node, {
            "llm_review": llm_risk_review,
            "security_flagged": human_approval_node,
        }),
        (llm_risk_review, human_approval_node),
    ],
)

app = App(
    root_agent=root_agent,
    name="ambient_expense_app",
    resumability_config=ResumabilityConfig(is_resumable=True),
)
