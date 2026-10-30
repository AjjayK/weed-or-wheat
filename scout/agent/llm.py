"""Thin wrapper over the Responses API for Muse Spark: one call = one LlmTurn."""
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from scout.retry import BadRequest, with_retry


class LlmError(RuntimeError):
    """Muse Spark answered, but the response did not complete (failed, incomplete, or cancelled)."""


@dataclass
class ToolCall:
    call_id: str
    name: str
    arguments: dict


@dataclass
class LlmTurn:
    response_id: str
    calls: list[ToolCall]
    text: str
    # The response's output as input items that can be replayed without previous_response_id: function calls with
    # their raw argument strings, and assistant text. Reasoning items are left out.
    output_items: list[dict] = field(default_factory=list)


def is_lost_response(error: Exception) -> bool:
    """Meta sometimes forgets a response a request chains to: 400 "referenced response not found or expired"."""
    return isinstance(error, BadRequest) and "referenced response not found" in str(error).lower()


def user_message(text: str, image_urls: Sequence[str] = (), file: tuple[str, str] | None = None) -> dict:
    """A user-role message. Images and files are only processed in user-role messages."""
    content: list[dict] = [{"type": "input_text", "text": text}]
    content += [{"type": "input_image", "image_url": url} for url in image_urls]
    if file is not None:
        content.append({"type": "input_file", "filename": file[0], "file_data": file[1]})
    return {"type": "message", "role": "user", "content": content}


class Llm:
    def __init__(self, create: Callable[..., Any], model: str, sleep: Callable[[float], None] = time.sleep):
        self._create = create
        self.model = model
        self._sleep = sleep

    @classmethod
    def from_settings(cls, settings) -> "Llm":
        import httpx
        from openai import OpenAI

        # max_retries=0 leaves with_retry as the only retry layer; connect=5 fails fast on an unreachable host.
        client = OpenAI(base_url=settings.base_url, api_key=settings.api_key,
                        timeout=httpx.Timeout(180.0, connect=5.0), max_retries=0)
        return cls(client.responses.create, settings.agent_model)

    def turn(self, input, instructions: str | None = None, tools: list[dict] | None = None,
             previous_response_id: str | None = None, text_format: dict | None = None) -> LlmTurn:
        kwargs: dict[str, Any] = {"model": self.model, "input": input}
        if instructions:
            kwargs["instructions"] = instructions
        if tools:
            kwargs["tools"] = tools
        if previous_response_id:
            kwargs["previous_response_id"] = previous_response_id
        if text_format:
            kwargs["text"] = {"format": text_format}
        response = with_retry(lambda: self._create(**kwargs), sleep=self._sleep)
        status = getattr(response, "status", None)
        if status in ("failed", "incomplete", "cancelled"):
            detail = getattr(response, "incomplete_details", None) or getattr(response, "error", None)
            raise LlmError(f"Muse Spark response {response.id} was {status}: {detail}")
        calls, items = [], []
        for item in response.output:
            kind = getattr(item, "type", None)
            if kind == "message" and getattr(item, "role", "assistant") == "assistant":
                text = "".join(part.text for part in item.content or [] if getattr(part, "type", None) == "output_text")
                if text:
                    items.append({"role": "assistant", "content": text})
            elif kind == "function_call":
                try:
                    arguments = json.loads(item.arguments or "{}")
                except json.JSONDecodeError:
                    arguments = None
                if not isinstance(arguments, dict):  # tools are called with **arguments, so only an object is usable
                    arguments = {"__invalid__": item.arguments}
                calls.append(ToolCall(item.call_id, item.name, arguments))
                items.append({"type": "function_call", "call_id": item.call_id, "name": item.name,
                              "arguments": item.arguments or "{}"})
        return LlmTurn(response.id, calls, getattr(response, "output_text", "") or "", items)
