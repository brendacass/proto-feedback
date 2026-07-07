import base64
import json
from pathlib import Path

import anthropic

SYSTEM_PROMPT = """\
You are a UX feedback analyst. You will receive a reviewer's verbal comment about a UI prototype screen.

Your job is to return a JSON object with this exact structure (no markdown, no extra text):
{
  "type": "<one of: usability, visual_hierarchy, accessibility, navigation, content, performance, positive_note, other>",
  "severity": "<one of: low, medium, high, critical — use null for positive_note>",
  "ui_element": "<specific element being discussed, e.g. 'submit button', 'navigation bar', 'form field'>",
  "issue": "<concise description of the problem or observation, one sentence>",
  "recommendation": "<actionable suggestion to address the issue, or null for positive_note>",
  "sentiment": "<one of: positive, neutral, negative>"
}\
"""

SYSTEM_PROMPT_VISION = SYSTEM_PROMPT + "\n\nA screenshot of the screen is also provided — use it to identify the specific UI element being discussed."


def _encode_image(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def _parse_json(raw: str) -> dict:
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def analyze_item(
    screenshot_path: Path,
    transcript: str,
    client: anthropic.Anthropic,
) -> dict | None:
    """Call Claude Vision to produce structured feedback for one screenshot+transcript pair."""
    try:
        image_data = _encode_image(screenshot_path)
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            system=SYSTEM_PROMPT_VISION,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": f'Reviewer comment: "{transcript}"',
                        },
                    ],
                }
            ],
        )
        return _parse_json(message.content[0].text.strip())
    except Exception:
        return None


def analyze_item_lm_studio(
    screenshot_path: Path,
    transcript: str,
    base_url: str,
    model: str,
    use_vision: bool = False,
) -> dict | None:
    """
    Call an LM Studio model via its OpenAI-compatible API.
    use_vision=True sends the screenshot as a base64 image (requires a vision-capable model).
    """
    try:
        from openai import OpenAI

        client = OpenAI(base_url=base_url, api_key="lm-studio")
        system = SYSTEM_PROMPT_VISION if use_vision else SYSTEM_PROMPT

        if use_vision:
            image_data = _encode_image(screenshot_path)
            user_content = [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_data}"},
                },
                {"type": "text", "text": f'Reviewer comment: "{transcript}"'},
            ]
        else:
            user_content = f'Reviewer comment: "{transcript}"'

        response = client.chat.completions.create(
            model=model,
            max_tokens=2048,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
        )
        return _parse_json(response.choices[0].message.content.strip())
    except Exception:
        return None
