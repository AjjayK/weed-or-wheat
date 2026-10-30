"""SAM 3.1 requests over the Meta Model API: concept prompts and box/point prompts, cached per image."""
import time
from typing import Callable, Sequence

from meta_sam_parser import FramePrompt, PromptBox, PromptObject, PromptPoint, build_object_prompt

from scout.app.ingest import StopImage
from scout.models import Box
from scout.retry import with_retry
from scout.vision.sam_parse import SamRecord, parse_sam_output


class SamClient:
    def __init__(self, send: Callable[[dict], str], model: str = "sam-3.1", sleep: Callable[[float], None] = time.sleep):
        self._send = send
        self._model = model
        self._sleep = sleep
        self._cache: dict[tuple[str, str], str] = {}
        self._parsed: dict[tuple[str, str], list[SamRecord]] = {}

    @classmethod
    def from_settings(cls, settings) -> "SamClient":
        import httpx
        from openai import OpenAI

        # max_retries=0 leaves with_retry as the only retry layer; connect=5 fails fast on an unreachable host.
        client = OpenAI(base_url=settings.base_url, api_key=settings.api_key,
                        timeout=httpx.Timeout(120.0, connect=5.0), max_retries=0)
        return cls(lambda payload: client.responses.create(**payload).output_text, model=settings.sam_model)

    def fetch_raw(self, image: StopImage, text: str) -> str:
        key = (image.hash, text)
        if key not in self._cache:
            payload = {
                "model": self._model,
                "input": [{
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": text},
                        {"type": "input_image", "image_url": image.work_data_url},
                    ],
                }],
                "metadata": {"include_confidence": "true", "mask_encoding": "one_bit"},
            }
            self._cache[key] = with_retry(lambda: self._send(payload), sleep=self._sleep)
        return self._cache[key]

    def _records(self, image: StopImage, text: str) -> list[SamRecord]:
        """Parsed records are cached too, since decoding masks costs seconds of CPU."""
        key = (image.hash, text)
        if key not in self._parsed:
            self._parsed[key] = parse_sam_output(self.fetch_raw(image, text))
        return self._parsed[key]

    def segment_concept(self, image: StopImage, phrase: str) -> list[SamRecord]:
        return self._records(image, phrase.strip())

    def segment_at(self, image: StopImage, box: Box | None = None,
                   points: Sequence[tuple[int, int, bool]] = ()) -> list[SamRecord]:
        prompt = FramePrompt(
            frame=0,
            box=PromptBox(*box) if box else None,
            points=[PromptPoint(int(x), int(y), label="positive" if positive else "negative") for x, y, positive in points],
        )
        text = build_object_prompt(size=image.size, objects=[PromptObject(id=1, prompts=[prompt])])
        return self._records(image, text)
