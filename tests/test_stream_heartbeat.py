"""SSE heartbeat tests: empty-delta frames keep silent-but-productive stream
spans (buffered tool-call text) visible to downstream progress counters, while
a genuinely quiet producer never heartbeats (hang detection stays with the
gateway's stall cap)."""

import asyncio
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from common import model
from endpoints.OAI.utils import chat_completion
from endpoints.OAI.utils.chat_completion import _compose_serialize_stream_heartbeat


class DummyDisconnectHandler:
    async def cleanup(self):
        pass


class DummyRequestData:
    n = 1
    stream_options = None

    def model_copy(self, deep=False):
        return self

    def model_dump(self, mode=None):
        return {}


def request_with_id(request_id="request-id"):
    return SimpleNamespace(state=SimpleNamespace(id=request_id))


def frame_json(frame):
    return json.loads(frame)


def is_heartbeat(frame):
    if frame == "[DONE]":
        return False
    choice = frame_json(frame)["choices"][0]
    return choice.get("delta") == {} and not choice.get("finish_reason")


class HeartbeatComposerTests(unittest.TestCase):
    def test_heartbeat_is_empty_delta_without_finish_or_usage(self):
        s = frame_json(_compose_serialize_stream_heartbeat("req", 3, "model"))

        self.assertEqual(s["id"], "chatcmpl-req")
        self.assertEqual(s["object"], "chat.completion.chunk")
        self.assertEqual(s["model"], "model")
        self.assertEqual(s["choices"], [{"index": 3, "delta": {}, "finish_reason": None}])
        self.assertIsNone(s["choices"][0]["finish_reason"])
        self.assertNotIn("usage", s)


class HeartbeatStreamTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._original_container = model.container
        model.container = SimpleNamespace(
            reasoning=False, harmony=False, muse_glimmer=False
        )

    def tearDown(self):
        model.container = self._original_container

    async def _run(self, collector, interval, timeout=2.0):
        frames = []

        async def collect():
            async for frame in chat_completion.stream_generate_chat_completion(
                "prompt",
                None,
                DummyRequestData(),
                request_with_id(),
                Path("model"),
                DummyDisconnectHandler(),
            ):
                frames.append(frame)

        with patch.object(chat_completion, "_chat_stream_collector", collector):
            with patch.object(chat_completion, "STREAM_HEARTBEAT_INTERVAL_S", interval):
                try:
                    await asyncio.wait_for(collect(), timeout=timeout)
                except asyncio.TimeoutError:
                    pass
        return frames

    async def test_tool_span_heartbeats_every_interval(self):
        async def collector(idx, gen_queue, *args, **kwargs):
            # Tool-call span: generation events arrive steadily but every
            # frame is empty (tool text is buffered until finish).
            for _ in range(10):
                await gen_queue.put({"index": idx, "delta_tool_calls": ""})
                await asyncio.sleep(0.03)
            await gen_queue.put(
                {
                    "index": idx,
                    "delta_tool_calls": [{"id": "t"}],
                    "finish_reason": "tool_calls",
                }
            )

        frames = await self._run(collector, interval=0.06)
        beats = [f for f in frames if is_heartbeat(f)]
        self.assertGreaterEqual(len(beats), 2)
        # Cadence pin: not one per empty event (10) — roughly one per interval
        self.assertLessEqual(len(beats), 8)
        for b in beats:
            data = frame_json(b)
            self.assertEqual(data["choices"][0]["index"], 0)
            self.assertIsNone(data["choices"][0]["finish_reason"])
            self.assertNotIn("usage", data)
        # The span still ends with the real tool-call finish frame
        finishes = [
            f
            for f in frames
            if f != "[DONE]" and frame_json(f)["choices"][0].get("finish_reason")
        ]
        self.assertEqual(len(finishes), 1)
        self.assertEqual(frames[-1], "[DONE]")

    async def test_content_stream_emits_no_heartbeats(self):
        async def collector(idx, gen_queue, *args, **kwargs):
            for i in range(10):
                await gen_queue.put({"index": idx, "delta_content": f"tok{i} "})
                await asyncio.sleep(0.03)
            await gen_queue.put(
                {"index": idx, "delta_content": "", "finish_reason": "stop"}
            )

        frames = await self._run(collector, interval=0.02)
        beats = [f for f in frames if is_heartbeat(f)]
        self.assertEqual(beats, [])
        content = [
            f
            for f in frames
            if f != "[DONE]" and frame_json(f)["choices"][0]["delta"].get("content")
        ]
        self.assertEqual(len(content), 10)

    async def test_quiet_producer_never_heartbeats(self):
        started = asyncio.Event()

        async def collector(idx, gen_queue, *args, **kwargs):
            await gen_queue.put({"index": idx, "delta_content": "x"})
            started.set()
            await asyncio.sleep(5)  # genuine stall: no further events

        frames = await self._run(collector, interval=0.05, timeout=0.3)
        self.assertTrue(started.is_set())
        beats = [f for f in frames if is_heartbeat(f)]
        self.assertEqual(beats, [])
        self.assertEqual(len(frames), 1)  # the one real frame, then silence


if __name__ == "__main__":
    unittest.main()
