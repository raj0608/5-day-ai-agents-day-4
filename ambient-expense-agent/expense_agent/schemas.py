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

"""Pydantic data schemas for expense reporting, risk evaluation, and security checkpointing."""

from pydantic import BaseModel, Field


class ExpenseReport(BaseModel):
    """Normalized expense report data structure."""

    amount: float = Field(default=0.0, description="Expense total in USD")
    submitter: str = Field(
        default="Unknown", description="Person who submitted the expense"
    )
    category: str = Field(
        default="Uncategorized",
        description="Expense category (e.g. Travel, Meals, Supplies)",
    )
    description: str = Field(
        default="", description="Description/justification of the expense"
    )
    date: str = Field(
        default="", description="Date of the expense in YYYY-MM-DD format"
    )
    validation_error: str | None = Field(
        default=None,
        description="Validation error code if any (e.g. MISSING_AMOUNT, INVALID_AMOUNT)",
    )


class SecurityCheckResult(BaseModel):
    """Security checkpoint analysis output schema."""

    is_prompt_injection: bool = Field(
        default=False, description="True if prompt injection was detected"
    )
    injection_reasons: list[str] = Field(
        default_factory=list, description="Detected prompt injection indicators"
    )
    redacted_categories: list[str] = Field(
        default_factory=list,
        description="Categories of PII redacted (e.g. SSN, CREDIT_CARD)",
    )
    sanitized_description: str = Field(
        ..., description="Scrubbed description with PII masked"
    )


class RiskReview(BaseModel):
    """LLM risk evaluation output schema."""

    risk_level: str = Field(
        ..., description="Risk assessment level: LOW, MEDIUM, or HIGH"
    )
    risk_factors: list[str] = Field(
        default_factory=list,
        description="List of identified risk factors or observations",
    )
    alert_summary: str = Field(
        ..., description="Concise summary alert for the human reviewer"
    )
    recommended_action: str = Field(
        ..., description="Recommended approval decision: APPROVE or REJECT"
    )
