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

"""Trace generator script running synthetic scenarios through local ADK runner and serializing traces."""

import asyncio
import json
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
if os.getenv("GEMINI_API_KEY"):
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "false"

from google.adk.runners import InMemoryRunner
from google.genai import types
from expense_agent.agent import app, scrub_pii

DATASET_PATH = Path("tests/eval/datasets/basic-dataset.json")
OUTPUT_PATH = Path("artifacts/traces/generated_traces.json")


async def generate_all_traces():
    """Runs all dataset evaluation scenarios, intercepting HITL approvals automatically."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset file not found at: {DATASET_PATH}")

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    eval_cases = dataset.get("eval_cases", [])
    runner = InMemoryRunner(app=app)
    output_cases = []

    print(f"Generating traces for {len(eval_cases)} evaluation cases...")

    for idx, case in enumerate(eval_cases):
        case_id = case.get("eval_case_id", f"case_{idx+1}")
        prompt_text = case["prompt"]["parts"][0]["text"]
        print(f"\n--- Running Case {idx+1}/{len(eval_cases)}: '{case_id}' ---")

        session = await runner.session_service.create_session(
            app_name=app.name, user_id=f"eval_user_{case_id}"
        )

        turns_events = []
        sanitized_prompt_text, _ = scrub_pii(prompt_text)
        user_content = types.Content(role="user", parts=[types.Part.from_text(text=prompt_text)])
        turns_events.append({
            "author": "user",
            "content": {"role": "user", "parts": [{"text": sanitized_prompt_text}]}
        })

        interrupt_msg = None
        final_text = ""

        # Step 1: Submit initial expense report event with 429 rate limit backoff retry
        for attempt in range(5):
            try:
                async for event in runner.run_async(
                    user_id=f"eval_user_{case_id}",
                    session_id=session.id,
                    new_message=user_content,
                ):
                    if event.content and event.content.parts:
                        for part in event.content.parts:
                            if getattr(part, "text", None):
                                final_text += part.text + "\n"
                                turns_events.append({
                                    "author": "ambient_expense_agent",
                                    "content": {"role": "model", "parts": [{"text": part.text}]}
                                })
                            elif getattr(part, "function_call", None):
                                fc = part.function_call
                                if getattr(fc, "name", "") == "adk_request_input" and isinstance(fc.args, dict):
                                    interrupt_msg = fc.args.get("message", "")
                                    turns_events.append({
                                        "author": "ambient_expense_agent",
                                        "content": {
                                            "role": "model",
                                            "parts": [{
                                                "function_call": {
                                                    "name": "adk_request_input",
                                                    "args": fc.args
                                                }
                                            }]
                                        }
                                    })
                break
            except Exception as e:
                if ("RESOURCE_EXHAUSTED" in str(e) or "429" in str(e)) and attempt < 4:
                    print(f"  [Rate limit 429] Waiting 5s before retry (attempt {attempt+2}/5)...")
                    await asyncio.sleep(5)
                else:
                    raise e

        # Step 2: Intercept HITL human decision if human approval was requested
        if interrupt_msg:
            is_security_alert = "CRITICAL SECURITY ALERT" in interrupt_msg or "PROMPT INJECTION" in interrupt_msg
            decision = "reject" if is_security_alert else "approve"
            print(f"  [HITL Intercepted] Prompt Injection Alert={is_security_alert} -> Decision: '{decision}'")

            turns_events.append({
                "author": "human_approver",
                "content": {
                    "role": "user",
                    "parts": [{
                        "function_response": {
                            "name": "adk_request_input",
                            "response": {"human_decision": decision}
                        }
                    }]
                }
            })

            resume_msg = types.Content(
                role="user",
                parts=[
                    types.Part.from_function_response(
                        name="adk_request_input", response={"human_decision": decision}
                    )
                ]
            )

            for attempt in range(5):
                try:
                    async for event in runner.run_async(
                        user_id=f"eval_user_{case_id}",
                        session_id=session.id,
                        new_message=resume_msg,
                    ):
                        if event.content and event.content.parts:
                            for part in event.content.parts:
                                if getattr(part, "text", None):
                                    final_text += part.text + "\n"
                                    turns_events.append({
                                        "author": "ambient_expense_agent",
                                        "content": {"role": "model", "parts": [{"text": part.text}]}
                                    })
                    break
                except Exception as e:
                    if ("RESOURCE_EXHAUSTED" in str(e) or "429" in str(e)) and attempt < 4:
                        print(f"  [Rate limit 429] Waiting 5s before retry (attempt {attempt+2}/5)...")
                        await asyncio.sleep(5)
                    else:
                        raise e

        output_cases.append({
            "eval_case_id": case_id,
            "prompt": {
                "role": "user",
                "parts": [{"text": sanitized_prompt_text}]
            },
            "responses": [
                {
                    "response": {
                        "role": "model",
                        "parts": [{"text": final_text.strip() or "Workflow completed."}]
                    }
                }
            ],
            "agent_data": {
                "agents": {
                    "ambient_expense_agent": {
                        "agent_id": "ambient_expense_agent",
                        "agent_type": "Workflow"
                    }
                },
                "turns": [
                    {
                        "turn_index": 0,
                        "events": turns_events
                    }
                ]
            }
        })

        # Pause 2s between cases to respect free tier rate limit (15 RPM)
        await asyncio.sleep(2)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"eval_cases": output_cases}, f, indent=2)

    print(f"\n✅ Successfully generated {len(output_cases)} traces -> {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(generate_all_traces())
