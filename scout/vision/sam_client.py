"""SAM 3.1 requests over the Meta Model API: concept prompts and box/point prompts, cached per image."""
import logging
import threading
import time
from collections import OrderedDict
from typing import Callable, Sequence

from meta_sam_parser import FramePrompt, PromptBox, PromptObject, PromptPoint, build_object_prompt

from scout.app.ingest import StopImage
from scout.models import Box
from scout.retry import with_retry
from scout.vision.sam_parse import SamRecord, parse_sam_output

log = logging.getLogger(__name__)


class SamClient:
    def __init__(self, send: Callable[[dict], str], model: str = "sam-3.1", sleep: Callable[[float], None] = time.sleep,
                 max_entries: int = 128):
        self._send = send
        self._model = model
        self._sleep = sleep
        # One client serves every walk for the life of the process, so both caches are bounded LRUs.
        self._max_entries = max_entries
        self._lock = threading.Lock()  # guards dict ops only; requests and decoding run unlocked
        self._cache: OrderedDict[tuple[str, str], str] = OrderedDict()
        self._parsed: OrderedDict[tuple[str, str], list[SamRecord]] = OrderedDict()

    @classmethod
    def from_settings(cls, settings) -> "SamClient":
        import httpx
        from openai import OpenAI

        # max_retries=0 leaves with_retry as the only retry layer; connect=5 fails fast on an unreachable host.
        client = OpenAI(base_url=settings.base_url, api_key=settings.api_key,
                        timeout=httpx.Timeout(120.0, connect=5.0), max_retries=0)

        def send(payload: dict) -> str:
            response = client.responses.create(**payload)
            if not response.output_text:
                log.warning("SAM reply had no output_text (status=%s, items=%s)", response.status,
                            [getattr(item, "type", "?") for item in response.output])
            return response.output_text

        return cls(send, model=settings.sam_model)

    def _get(self, store: OrderedDict, key):
        with self._lock:
            if key in store:
                store.move_to_end(key)
                return store[key]
        return None

    def _put(self, store: OrderedDict, key, value):
        with self._lock:
            store[key] = value
            store.move_to_end(key)
            while len(store) > self._max_entries:
                store.popitem(last=False)
        return value

    def fetch_raw(self, image: StopImage, text: str) -> str:
        key = (image.hash, text)
        raw = self._get(self._cache, key)
        if raw is None:
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
            raw = self._put(self._cache, key, with_retry(lambda: self._send(payload), sleep=self._sleep))
        return raw

    def _records(self, image: StopImage, text: str) -> list[SamRecord]:
        """Parsed records are cached too, since decoding masks costs seconds of CPU."""
        key = (image.hash, text)
        records = self._get(self._parsed, key)
        if records is None:
            records = self._put(self._parsed, key, parse_sam_output(self.fetch_raw(image, text)))
        return records

    def segment_concept(self, image: StopImage, phrase: str) -> list[SamRecord]:
        return self._records(image, phrase.strip())

    def segment_at(self, image: StopImage, box: Box | None = None,
                   points: Sequence[tuple[int, int, bool]] = ()) -> list[SamRecord]:
        prompt = FramePrompt(
            frame=0,
            box=PromptBox(*(int(v) for v in box)) if box else None,
            points=[PromptPoint(int(x), int(y), label="positive" if positive else "negative") for x, y, positive in points],
        )
        text = build_object_prompt(size=image.size, objects=[PromptObject(id=1, prompts=[prompt])])
        return self._records(image, text)
