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

import ast
import base64
import json
import logging
import os
import re
from collections.abc import AsyncGenerator
from typing import Any

from dotenv import load_dotenv

load_dotenv()
if os.getenv("GEMINI_API_KEY"):
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "false"

from google.adk.agents import LlmAgent
from google.adk.agents.context import Context
from google.adk.apps import App, ResumabilityConfig
from google.adk.events.event import Event
from google.adk.events.request_input import RequestInput
from google.adk.workflow import START, FunctionNode, Workflow
from google.genai import types
from pydantic import ValidationError

from . import config
from .schemas import ExpenseReport, RiskReview, SecurityCheckResult


class ParseFailure(Exception):
    """Raised when an inbound event payload cannot be unwrapped, validated, or parsed as a valid ExpenseReport."""

    pass


def unwrap_payload_layers(raw_input: Any) -> Any:
    """Iteratively unwrap nesting layers (dict, ADK Content/Part, Python repr string, double-escaped JSON)."""
    current = raw_input
    for _ in range(10):
        model_dump_fn = getattr(current, "model_dump", None)
        if callable(model_dump_fn):
            current = model_dump_fn()
            continue

        if isinstance(current, dict):
            # Unpack common wrapper keys
            for key in ("data", "input", "message", "payload", "event", "content"):
                if key in current and len(current) == 1:
                    current = current[key]
                    break
            else:
                if (
                    "parts" in current
                    and isinstance(current["parts"], list)
                    and current["parts"]
                ):
                    first_part = current["parts"][0]
                    if isinstance(first_part, dict):
                        current = first_part.get("text", first_part)
                    else:
                        current = getattr(first_part, "text", str(first_part))
                    continue
                else:
                    break
            continue
        elif isinstance(current, str):
            curr_str = current.strip()

            # 1. Strip markdown code fences if present (```json ... ``` or ``` ... ```)
            fence_match = re.match(
                r"^```(?:json)?\s*([\s\S]*?)\s*```$", curr_str, re.IGNORECASE
            )
            if fence_match:
                candidate = fence_match.group(1).strip()
                if candidate != current:
                    current = candidate
                    continue

            # 2. ADK Part representation in string
            if "Part(" in curr_str or "parts=" in curr_str:
                match = re.search(
                    r"text=\s*[\x27\"](.*?)[\x27\"](?:\s*\)|\s*,|\s*$)",
                    curr_str,
                    re.DOTALL,
                )
                if match:
                    current = match.group(1)
                    continue

            # 3. Direct JSON loads
            try:
                res = json.loads(curr_str)
                if res != current:
                    current = res
                    continue
            except Exception:
                pass

            # 4. AST literal eval (for Python dict reprs with single quotes)
            try:
                res = ast.literal_eval(curr_str)
                if res != current:
                    current = res
                    continue
            except Exception:
                pass

            # 5. Escaped quote cleanup
            if '\\"' in curr_str or "\\\\" in curr_str:
                cleaned = curr_str.replace('\\"', '"').replace("\\\\", "\\")
                try:
                    res = json.loads(cleaned)
                    if res != current:
                        current = res
                        continue
                except Exception:
                    pass
                try:
                    res = ast.literal_eval(cleaned)
                    if res != current:
                        current = res
                        continue
                except Exception:
                    pass

            # 6. Base64 decoding
            try:
                decoded_bytes = base64.b64decode(curr_str)
                res = json.loads(decoded_bytes.decode("utf-8"))
                if res != current:
                    current = res
                    continue
            except Exception:
                pass

            # 7. Extract outermost JSON substring if surrounded by prose
            if "{" in curr_str and "}" in curr_str:
                start_idx = curr_str.find("{")
                end_idx = curr_str.rfind("}") + 1
                candidate = curr_str[start_idx:end_idx].strip()
                if candidate and candidate != curr_str:
                    try:
                        res = json.loads(candidate)
                        current = res
                        continue
                    except Exception:
                        pass
                    try:
                        res = ast.literal_eval(candidate)
                        current = res
                        continue
                    except Exception:
                        pass

            break
        else:
            break
    return current


def parse_expense_payload(raw_input: Any) -> tuple[ExpenseReport, list[str]]:
    """Parse raw payload from plain JSON, dict, ADK Content object, or Python repr string envelope.

    Preserves submitter, category, description, and date even when amount or date is missing/invalid.
    Raises ParseFailure only if payload cannot be unwrapped into a dictionary.
    """
    payload_layer = unwrap_payload_layers(raw_input)

    if not isinstance(payload_layer, dict):
        raise ParseFailure(
            f"Final payload layer is not a dictionary: {type(payload_layer).__name__}"
        )

    # 1. Extract raw field values
    raw_amount = payload_layer.get("amount")
    raw_submitter = payload_layer.get("submitter")
    raw_category = payload_layer.get("category")
    raw_desc = payload_layer.get("description", "")
    raw_date = payload_layer.get("date")

    # 2. Scrub PII across all text fields
    sanitized_desc, desc_pii = scrub_pii(str(raw_desc))
    sanitized_submitter, sub_pii = scrub_pii(
        str(raw_submitter) if raw_submitter is not None else ""
    )
    sanitized_category, cat_pii = scrub_pii(
        str(raw_category) if raw_category is not None else ""
    )
    sanitized_date, date_pii = scrub_pii(
        str(raw_date) if raw_date is not None else ""
    )

    redacted_categories = list(dict.fromkeys(desc_pii + sub_pii + cat_pii + date_pii))

    validation_error: str | None = None
    parsed_amount: float = 0.0

    # 3. Amount parsing and validation
    if raw_amount is None or (isinstance(raw_amount, str) and not raw_amount.strip()):
        validation_error = "MISSING_AMOUNT"
        parsed_amount = 0.0
    else:
        cleaned_amount = raw_amount
        if isinstance(raw_amount, str):
            cleaned_amount_str = raw_amount.replace("$", "").replace(",", "").strip()
            try:
                cleaned_amount = float(cleaned_amount_str)
            except ValueError:
                validation_error = "INVALID_AMOUNT"
                cleaned_amount = 0.0

        if isinstance(cleaned_amount, (int, float)):
            parsed_amount = float(cleaned_amount)
            if parsed_amount < 0.0:
                validation_error = "INVALID_AMOUNT"
            elif parsed_amount == 0.0 and validation_error is None:
                validation_error = "ZERO_AMOUNT"
        else:
            validation_error = "INVALID_AMOUNT"
            parsed_amount = 0.0

    # 4. Submitter validation
    submitter_final = sanitized_submitter.strip()
    if not submitter_final:
        submitter_final = "Unknown"
        if validation_error is None:
            validation_error = "MISSING_SUBMITTER"

    # 5. Category validation
    category_final = sanitized_category.strip()
    if not category_final:
        category_final = "Uncategorized"
        if validation_error is None:
            validation_error = "MISSING_CATEGORY"

    # 6. Date validation
    date_final = sanitized_date.strip()
    if not date_final:
        date_final = ""
        if validation_error is None:
            validation_error = "MISSING_DATE"

    expense = ExpenseReport(
        amount=parsed_amount,
        submitter=submitter_final,
        category=category_final,
        description=sanitized_desc,
        date=date_final,
        validation_error=validation_error,
    )

    return expense, redacted_categories


def scrub_pii(text: str) -> tuple[str, list[str]]:
    """Scrub SSN, Credit Card, and Phone numbers from text, returning sanitized text and list of redacted categories."""
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

    # 4. Phone number pattern (matches bare NNN-NNN-NNNN, (NNN) NNN-NNNN, 1-NNN-NNN-NNNN)
    phone_pattern = r"(?:\+?1[-\s.]?)?\(?\b\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b"
    if re.search(phone_pattern, text):
        if "PHONE_NUMBER" not in redacted_categories:
            redacted_categories.append("PHONE_NUMBER")
        text = re.sub(phone_pattern, "[REDACTED_PHONE]", text)

    return text, redacted_categories


def detect_prompt_injection(text: str) -> tuple[bool, list[str]]:
    """Detect prompt injection attempt indicators targeting LLM instruction override or policy bypass."""
    injection_patterns = [
        (r"bypass\b", "Policy bypass attempt"),
        (
            r"ignore\b.*?\b(instructions?|rules?|systems?|prompts?)\b",
            "Instruction override attempt",
        ),
        (r"override\b", "System override attempt"),
        (r"auto[- ]?approve\b", "Forced auto-approval attempt"),
        (r"system\s*prompts?", "System prompt reference"),
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
    """Node 1: Parse input expense, immediately scrub PII, detect multi-field prompt injection, and route safely."""
    raw_text = str(node_input)
    is_inj_raw, inj_reasons_raw = detect_prompt_injection(raw_text)

    try:
        expense, redacted_categories = parse_expense_payload(node_input)
    except ParseFailure as parse_err:
        logging.warning(
            "ParseFailure encountered in parse_expense_node: %s", str(parse_err)
        )
        sanitized_raw, extra_redacted = scrub_pii(raw_text)
        all_redacted = list(dict.fromkeys(["UNPARSABLE_PAYLOAD", *extra_redacted]))
        if is_inj_raw:
            all_redacted.append("PROMPT_INJECTION")

        # Non-content description summary kept free of raw input text
        dummy_expense = {
            "amount": 0.0,
            "submitter": "Unknown",
            "category": "Unparsable",
            "description": f"Malformed payload: {type(parse_err).__name__}",
            "date": "",
            "validation_error": "UNPARSABLE_PAYLOAD",
        }

        sec_result = SecurityCheckResult(
            is_prompt_injection=is_inj_raw,
            injection_reasons=inj_reasons_raw if is_inj_raw else [],
            redacted_categories=all_redacted,
            sanitized_description=dummy_expense["description"],
        )

        state_delta = {
            "expense": dummy_expense,
            "security_check": sec_result.model_dump(),
            "redacted_categories": all_redacted,
            "raw_payload_debug": sanitized_raw,
        }

        # Fail closed to security_check node
        return Event(
            output=dummy_expense,
            route="security_check",
            state=state_delta,
        )

    # Multi-field Prompt Injection Detection across all extracted fields
    inj_targets = [
        raw_text,
        expense.description,
        expense.category,
        expense.submitter,
        expense.date,
    ]
    is_injection = is_inj_raw
    injection_reasons = list(inj_reasons_raw)
    for field_val in inj_targets:
        if field_val:
            inj_detected, reasons = detect_prompt_injection(str(field_val))
            if inj_detected:
                is_injection = True
                injection_reasons.extend(reasons)
    injection_reasons = list(dict.fromkeys(injection_reasons))

    if is_injection and "PROMPT_INJECTION" not in redacted_categories:
        redacted_categories.append("PROMPT_INJECTION")

    sec_result = SecurityCheckResult(
        is_prompt_injection=is_injection,
        injection_reasons=injection_reasons,
        redacted_categories=redacted_categories,
        sanitized_description=expense.description,
    )

    state_delta = {
        "expense": expense.model_dump(),
        "security_check": sec_result.model_dump(),
        "redacted_categories": redacted_categories,
    }

    # Strict Routing Policy:
    # Auto-approve ONLY if:
    # - No validation errors (validation_error is None)
    # - Strict positive threshold: 0.0 < amount < 100.0
    # - No PII detected
    # - No prompt injection detected
    if (
        expense.validation_error is None
        and not is_injection
        and len(redacted_categories) == 0
        and 0.0 < expense.amount < config.AUTO_APPROVE_THRESHOLD
    ):
        route = "auto_approve"
    else:
        route = "security_check"

    return Event(
        output=expense.model_dump(),
        route=route,
        state=state_delta,
    )


def security_checkpoint_node(ctx: Context, node_input: dict[str, Any]) -> Event:
    """Node 2: Security Checkpoint. Performs prompt injection defense and defense-in-depth second pass PII check."""
    expense_data = ctx.state.get("expense", node_input)
    raw_desc = str(expense_data.get("description", ""))
    raw_cat = str(expense_data.get("category", ""))
    raw_sub = str(expense_data.get("submitter", ""))
    existing_redacted = ctx.state.get("redacted_categories", [])
    existing_sec = ctx.state.get("security_check", {})

    # Defense-in-depth second pass PII check
    sanitized_desc, new_redacted = scrub_pii(raw_desc)
    expense_data["description"] = sanitized_desc
    redacted_categories = list(dict.fromkeys(existing_redacted + new_redacted))

    # Check for Prompt Injection across all fields AND existing security_check state from parse node
    all_reasons = list(existing_sec.get("injection_reasons", []))
    is_injection = existing_sec.get("is_prompt_injection", False)

    for field_text in (raw_desc, raw_cat, raw_sub):
        if field_text:
            inj_detected, reasons = detect_prompt_injection(field_text)
            if inj_detected:
                is_injection = True
                all_reasons.extend(reasons)
    all_reasons = list(dict.fromkeys(all_reasons))

    if is_injection and "PROMPT_INJECTION" not in redacted_categories:
        redacted_categories.append("PROMPT_INJECTION")

    sec_result = SecurityCheckResult(
        is_prompt_injection=is_injection,
        injection_reasons=all_reasons,
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


async def human_approval_node(
    ctx: Context, node_input: dict[str, Any]
) -> AsyncGenerator[Any, None]:
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
        val_error = expense_data.get("validation_error")

        if is_injection:
            # Prompt injection security alert layout (LLM bypassed)
            reasons = ", ".join(sec_check.get("injection_reasons", []))
            redacted_str = (
                f"\n🔒 Redacted PII Categories: {', '.join(redacted_cats)}"
                if redacted_cats
                else ""
            )

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
            # Determine precise header based on review trigger
            if val_error == "INVALID_AMOUNT":
                header_text = f"⚠️ HUMAN REVIEW REQUIRED (Invalid Expense Amount: ${amount:.2f})"
            elif val_error == "MISSING_AMOUNT":
                header_text = "⚠️ HUMAN REVIEW REQUIRED (Missing Required Field: Amount)"
            elif val_error == "ZERO_AMOUNT":
                header_text = "⚠️ HUMAN REVIEW REQUIRED (Zero-Dollar Expense)"
            elif val_error == "MISSING_DATE":
                header_text = "⚠️ HUMAN REVIEW REQUIRED (Missing Required Field: Date)"
            elif val_error == "MISSING_SUBMITTER":
                header_text = "⚠️ HUMAN REVIEW REQUIRED (Missing Required Field: Submitter)"
            elif val_error == "MISSING_CATEGORY":
                header_text = "⚠️ HUMAN REVIEW REQUIRED (Missing Required Field: Category)"
            elif (
                expense_data.get("category") == "Unparsable"
                or "UNPARSABLE_PAYLOAD" in redacted_cats
                or val_error == "UNPARSABLE_PAYLOAD"
            ):
                header_text = "⚠️ HUMAN REVIEW REQUIRED (Unparsable payload — could not validate)"
            elif len(redacted_cats) > 0 and amount < config.AUTO_APPROVE_THRESHOLD:
                header_text = "⚠️ HUMAN APPROVAL REQUIRED (Sensitive PII Detected)"
            elif len(redacted_cats) > 0 and amount >= config.AUTO_APPROVE_THRESHOLD:
                header_text = (
                    f"⚠️ HUMAN APPROVAL REQUIRED (Expense >= ${config.AUTO_APPROVE_THRESHOLD:.2f} & Sensitive PII Detected)"
                )
            else:
                header_text = f"⚠️ HUMAN APPROVAL REQUIRED (Expense >= ${config.AUTO_APPROVE_THRESHOLD:.2f})"

            risk_alert = node_input.get(
                "alert_summary", "Manual review required by policy."
            )
            risk_level = node_input.get("risk_level", "MEDIUM")
            risk_factors = (
                ", ".join(node_input.get("risk_factors", [])) or "None identified"
            )
            rec = node_input.get("recommended_action", "REVIEW")
            redacted_str = (
                f"\n🔒 Redacted PII Categories: {', '.join(redacted_cats)}"
                if redacted_cats
                else ""
            )

            message = (
                f"{header_text}\n"
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

        # Emit model text content so chat UIs (such as Vertex AI Playground) render the message
        alert_content = types.Content(
            role="model",
            parts=[types.Part.from_text(text=message)],
        )
        yield Event(content=alert_content, author="ambient_expense_agent")
        yield RequestInput(interrupt_id="human_decision", message=message)
        return

    # Process human approval/rejection response
    is_approved = (
        "approve" in human_reply.lower() and "reject" not in human_reply.lower()
    )

    # Prevent prompt injection attacks inside resume_inputs from spoofing human approval
    is_inj_reply, _ = detect_prompt_injection(human_reply)
    if is_inj_reply:
        is_approved = False
        decision_summary = (
            "Security rejection: Resume input contained suspected prompt injection"
        )
        status = "SECURITY_REJECTED"
    elif sec_check.get("is_prompt_injection", False):
        status = "SECURITY_APPROVED" if is_approved else "SECURITY_REJECTED"
        decision_summary = "Human reviewer decision on security-flagged expense"
    else:
        status = "HUMAN_APPROVED" if is_approved else "HUMAN_REJECTED"
        decision_summary = (
            "Approved by human reviewer"
            if is_approved
            else "Rejected by human reviewer"
        )

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


resumable_human_approval_node = FunctionNode(
    func=human_approval_node,
    name="human_approval_node",
    rerun_on_resume=True,
)

# Construct the ADK 2.0 Workflow Graph with Security Controls
root_agent = Workflow(
    name="ambient_expense_agent",
    description=(
        "Ambient expense approval graph workflow with threshold auto-approval, "
        "PII redaction, prompt injection defense, and LLM risk review with HITL approval."
    ),
    edges=[
        (START, parse_expense_node),
        (
            parse_expense_node,
            {
                "auto_approve": auto_approve_node,
                "security_check": security_checkpoint_node,
            },
        ),
        (
            security_checkpoint_node,
            {
                "llm_review": llm_risk_review,
                "security_flagged": resumable_human_approval_node,
            },
        ),
        (llm_risk_review, resumable_human_approval_node),
    ],
)

app = App(
    root_agent=root_agent,
    name="ambient_expense_app",
    resumability_config=ResumabilityConfig(is_resumable=True),
)
