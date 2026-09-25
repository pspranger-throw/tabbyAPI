"""KV cache save/restore endpoint tests (save-session P2) — offline, no GPU, no model.

Gate the tabbyAPI side of kv-persistence/tabby-save-session-plan.md P2 against a stubbed
generator (the engine mechanics are gated by exllamav3's `test_kv_save_restore.py` (P1) and the
pre-signed P3 rig drills):

* `POST /v1/cache/save` policy matrix: 409 busy-skip (D12 no waits), 422 refused/skipped
  (nothing written, existing set kept — including the zero-stash "no dead sets" skip),
  500/504 definitive machinery failure, 200 saved with save_ms/counts/bytes (D8 parity).
* `GET /v1/cache/status`: drill-assertable restore + last-save records (no log-grep).
* Eager load-time restore is exception-isolated fail-closed-to-COLD (G4) and quarantines a
  rejected set (`*.rejected-<ts>`) instead of deleting it.

Async style follows tests/test_context_length_errors.py (`unittest.IsolatedAsyncioTestCase` —
the suite runs without a pytest async plugin).
"""

import asyncio
import os
import shutil
import tempfile
import unittest

import pytest

pytest.importorskip("exllamav3")

from fastapi import HTTPException  # noqa: E402

from backends.exllamav3.model import ExllamaV3Container  # noqa: E402
from common import model as model_module  # noqa: E402
from common.config_models import KvSaveConfig, TabbyConfigModel  # noqa: E402
from common.tabby_config import config  # noqa: E402
from endpoints.core.router import kv_cache_save, kv_cache_status  # noqa: E402


class FakeGenerator:
    """Stands in for the engine AsyncGenerator's P2 surface."""

    def __init__(
        self,
        save_result = None,
        save_error = None,
        restore_result = None,
        restore_error = None,
        save_delay = 0.0,
    ):
        self.error = None
        self.save_error = save_error
        self.save_delay = save_delay
        self.restore_error = restore_error
        self.save_calls = []
        self.restore_calls = []
        self.save_result = save_result if save_result is not None else {
            "dir": "/tmp/kv/session",
            "n_pages": 512,
            "n_stashes": 4,
            "stash_keys": [],
            "checks": {"chain_complete": True, "anchor_in_chain": True},
            "meta": {
                "files": {
                    "pages.bin": {"sha256": "aa", "size": 3145728},
                    "chain.json": {"sha256": "bb", "size": 2048},
                    "stash-0.bin": {"sha256": "cc", "size": 262144},
                }
            },
            "deepest_anchor_page_idx": 500,
        }
        self.restore_result = restore_result if restore_result is not None else {
            "pages_restored": 512,
            "stashes_restored": 4,
            "deepest_anchor_page_idx": 500,
            "stash_keys": [],
            "meta": {},
        }

    def request_save(self, store, stash_budget_mb = None):
        """Sync call returning a Future — mirrors AsyncGenerator.request_save's real surface.
        An `async def` fake has different cancel semantics than production (P2 review R7/F7)."""
        self.save_calls.append((store, stash_budget_mb))
        loop = asyncio.get_running_loop()
        fut = loop.create_future()

        def _complete():
            if fut.done():
                return
            if self.save_error is not None:
                fut.set_exception(self.save_error)
            else:
                fut.set_result(self.save_result)

        loop.call_later(self.save_delay, _complete)
        return fut

    def restore_state(self, store):
        self.restore_calls.append(store)
        if self.restore_error is not None:
            raise self.restore_error
        return self.restore_result


def make_container(generator = None):
    container = ExllamaV3Container.__new__(ExllamaV3Container)
    container.active_job_ids = {}
    container.load_lock = asyncio.Lock()
    container.load_condition = asyncio.Condition()
    container.generator = generator
    container._kv_last_save = None
    container._kv_restore = None
    return container


class _KvCaseBase(unittest.IsolatedAsyncioTestCase):
    """Temp store + global config/container patching, restored on teardown."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix = "kvsave-p2-test-")
        self.store = os.path.join(self.tmp, "session")
        self._prev_kv = config.kv_save
        self._prev_container = model_module.container

    def tearDown(self):
        config.kv_save = self._prev_kv
        model_module.container = self._prev_container
        shutil.rmtree(self.tmp, ignore_errors = True)

    def configure(self, store = True, budget = 256):
        config.kv_save = KvSaveConfig(
            store_dir = self.store if store else None, stash_budget_mb = budget
        )

    def unconfigure(self):
        config.kv_save = KvSaveConfig()

    def _make_store(self):
        """A store dir that reads as a candidate set (non-empty, with a meta.json)."""
        os.makedirs(self.store, exist_ok = True)
        with open(os.path.join(self.store, "meta.json"), "w") as f:
            f.write("{}")


class KvSavePolicyTests(_KvCaseBase):
    async def test_not_configured_refuses_and_keeps_set(self):
        self.unconfigure()
        gen = FakeGenerator()
        container = make_container(gen)
        code, record = await container.kv_save()
        self.assertEqual(code, 422)
        self.assertEqual(record["status"], "refused")
        self.assertIn("not configured", record["reason"])
        self.assertEqual(gen.save_calls, [])

    async def test_no_generator_refuses_and_keeps_set(self):
        self.configure()
        container = make_container(None)
        code, record = await container.kv_save()
        self.assertEqual(code, 422)
        self.assertEqual(record["status"], "refused")
        self.assertIn("no generator", record["reason"])

    async def test_latched_generator_refuses_and_keeps_set(self):
        self.configure()
        gen = FakeGenerator()
        gen.error = RuntimeError("latched")
        container = make_container(gen)
        code, record = await container.kv_save()
        self.assertEqual(code, 422)
        self.assertEqual(record["status"], "refused")
        self.assertIn("latched", record["reason"])
        self.assertEqual(gen.save_calls, [])

    async def test_busy_skips_immediately_without_waiting(self):
        self.configure()
        gen = FakeGenerator(save_delay = 5.0)
        container = make_container(gen)
        container.active_job_ids["req-1"] = object()
        code, record = await container.kv_save()
        self.assertEqual(code, 409)
        self.assertEqual(record["status"], "busy")
        self.assertIn("1 active job", record["reason"])
        self.assertEqual(gen.save_calls, [], "D12: a busy save must not queue a snapshot")

    async def test_saved_200_carries_d8_observability(self):
        self.configure(budget = 256)
        gen = FakeGenerator()
        container = make_container(gen)
        code, record = await container.kv_save()
        self.assertEqual(code, 200)
        self.assertEqual(record["status"], "saved")
        self.assertEqual(record["n_pages"], 512)
        self.assertEqual(record["n_stashes"], 4)
        self.assertEqual(record["bytes"], 3145728 + 2048 + 262144)
        self.assertIsNotNone(record["save_ms"])
        self.assertGreaterEqual(record["save_ms"], 0)
        self.assertEqual(record["store_dir"], self.store)
        self.assertEqual(gen.save_calls, [(self.store, 256)],
                         "store + configured stash budget must be passed")
        self.assertEqual(container.kv_status()["last_save"]["status"], "saved")

    async def test_zero_stash_is_skipped_not_dead_set(self):
        """User decision 2026-09-24: zero-stash saves = skip + loud log, no dead sets."""
        self.configure()
        gen = FakeGenerator(save_result = {
            "dir": None, "skipped": "zero-stash", "n_pages": 3, "n_stashes": 0,
            "stash_keys": [], "checks": None, "meta": None, "deepest_anchor_page_idx": -1,
        })
        container = make_container(gen)
        code, record = await container.kv_save()
        self.assertEqual(code, 422)
        self.assertEqual(record["status"], "skipped")
        self.assertEqual(record["reason"], "zero-stash")
        self.assertEqual(record["n_pages"], 3)
        self.assertEqual(record["n_stashes"], 0)
        self.assertEqual(container.kv_status()["last_save"]["reason"], "zero-stash")

    async def test_engine_exception_is_definitive_failure(self):
        self.configure()
        gen = FakeGenerator(save_error = RuntimeError("io exploded"))
        container = make_container(gen)
        code, record = await container.kv_save()
        self.assertEqual(code, 500)
        self.assertEqual(record["status"], "error")
        self.assertIn("io exploded", record["reason"])

    async def test_save_timeout_is_definitive_failure(self):
        self.configure()
        gen = FakeGenerator(save_delay = 5.0)
        container = make_container(gen)
        container.KV_SAVE_TIMEOUT_S = 0.05
        code, record = await container.kv_save()
        self.assertEqual(code, 504)
        self.assertEqual(record["status"], "error")
        self.assertIn("timed out", record["reason"])

    async def test_save_releases_and_wakes_condition_waiters(self):
        """The save path must pair the load_lock release with a load_condition notify like every
        other lock holder: a generation request parked during a save must wake when the save ends
        (P2 review R1/F1/E1 — 4/4 blind eyes). Pre-fix this hangs until an unrelated notify."""
        self.configure()
        gen = FakeGenerator(save_delay = 0.2)
        container = make_container(gen)
        save_task = asyncio.create_task(container.kv_save())
        await asyncio.sleep(0.05)
        self.assertTrue(container.load_lock.locked(), "kv_save must hold the lock while saving")

        woken = asyncio.Event()

        async def waiter():
            async with container.load_condition:
                await container.load_condition.wait_for(lambda: not container.load_lock.locked())
            woken.set()

        waiter_task = asyncio.create_task(waiter())
        await asyncio.sleep(0.05)  # let the waiter park on the held lock
        code, record = await save_task
        self.assertEqual(code, 200)
        await asyncio.wait_for(woken.wait(), timeout = 2.0)
        await waiter_task

    async def test_generator_vanished_under_lock_refuses_not_errors(self):
        """A generator that disappears between the unlocked view and the lock (unload/swap holds
        the lock while nulling it) must yield the polite 422, not a spurious definitive 500
        (P2 review R2 — destructive under the P3 driver policy)."""
        self.configure()
        gen = FakeGenerator()
        container = make_container(gen)
        orig_acquire = container.load_lock.acquire

        async def acquire_then_null():
            res = await orig_acquire()
            container.generator = None
            return res

        container.load_lock.acquire = acquire_then_null
        code, record = await container.kv_save()
        self.assertEqual(code, 422)
        self.assertEqual(record["status"], "refused")
        self.assertIn("no generator", record["reason"])
        self.assertEqual(gen.save_calls, [])

    async def test_save_endpoint_maps_status_codes(self):
        self.configure()
        gen = FakeGenerator(save_error = RuntimeError("boom"))
        model_module.container = make_container(gen)
        with self.assertRaises(HTTPException) as ctx:
            await kv_cache_save()
        self.assertEqual(ctx.exception.status_code, 500)

        gen2 = FakeGenerator(save_result = {"skipped": "zero-stash", "n_pages": 0, "n_stashes": 0})
        model_module.container = make_container(gen2)
        with self.assertRaises(HTTPException) as ctx2:
            await kv_cache_save()
        self.assertEqual(ctx2.exception.status_code, 422)
        self.assertIn("zero-stash", ctx2.exception.detail)

        model_module.container = None
        with self.assertRaises(HTTPException) as ctx3:
            await kv_cache_save()
        self.assertEqual(ctx3.exception.status_code, 422)


class KvRestoreTests(_KvCaseBase):
    def test_restore_not_configured(self):
        self.unconfigure()
        gen = FakeGenerator()
        container = make_container(gen)
        container._kv_restore_on_create()
        self.assertEqual(gen.restore_calls, [])
        self.assertEqual(container.kv_status()["restore"]["reason"], "not-configured")

    def test_restore_no_store_present_is_cold_start(self):
        self.configure()
        gen = FakeGenerator()
        container = make_container(gen)
        container._kv_restore_on_create()
        self.assertEqual(gen.restore_calls, [])
        rec = container.kv_status()["restore"]
        self.assertFalse(rec["attempted"])
        self.assertEqual(rec["reason"], "no-store")

    def test_restore_ok_records_counts(self):
        self.configure()
        self._make_store()
        gen = FakeGenerator()
        container = make_container(gen)
        container._kv_restore_on_create()
        self.assertEqual(gen.restore_calls, [self.store])
        rec = container.kv_status()["restore"]
        self.assertTrue(rec["ok"])
        self.assertEqual(rec["pages_restored"], 512)
        self.assertEqual(rec["stashes_restored"], 4)
        self.assertEqual(rec["deepest_anchor_page_idx"], 500)

    def test_restore_empty_dir_is_no_store_not_rejected(self):
        """An existing-but-empty store dir (e.g. pre-created by the driver) is not a candidate
        set: no-store, no quarantine (P2 review R5)."""
        self.configure()
        os.makedirs(self.store)  # empty on purpose
        gen = FakeGenerator()
        container = make_container(gen)
        container._kv_restore_on_create()
        self.assertEqual(gen.restore_calls, [])
        rec = container.kv_status()["restore"]
        self.assertFalse(rec["attempted"])
        self.assertEqual(rec["reason"], "no-store")
        self.assertTrue(os.path.isdir(self.store), "an empty dir must not be quarantined")
        self.assertEqual([p for p in os.listdir(self.tmp) if ".rejected-" in p], [])

    def test_restore_failure_is_fail_closed_and_quarantines_set(self):
        self.configure()
        self._make_store()
        gen = FakeGenerator(restore_error = RuntimeError("digest mismatch"))
        container = make_container(gen)
        container._kv_restore_on_create()  # must not raise — the id keeps serving
        rec = container.kv_status()["restore"]
        self.assertFalse(rec["ok"])
        self.assertTrue(rec["attempted"])
        self.assertTrue(rec["reason"].startswith("rejected:"))
        self.assertIn("digest mismatch", rec["reason"])
        # The live path is vacated (no re-queue of the same failure) but the set is kept
        self.assertFalse(os.path.isdir(self.store))
        rejected = [p for p in os.listdir(self.tmp) if ".rejected-" in p]
        self.assertEqual(len(rejected), 1)


class KvStatusSurfaceTests(_KvCaseBase):
    async def test_status_endpoint_reports_records(self):
        self.configure()
        gen = FakeGenerator()
        container = make_container(gen)
        container._kv_restore_on_create()  # production order: load (restore pass) precedes saves
        await container.kv_save()
        model_module.container = container
        status = await kv_cache_status()
        self.assertTrue(status.configured)
        self.assertEqual(status.store_dir, self.store)
        self.assertEqual(status.stash_budget_mb, 256)
        self.assertTrue(status.generator_loaded)
        self.assertEqual(status.last_save.status, "saved")
        self.assertEqual(status.restore.reason, "no-store")

    async def test_status_endpoint_without_container(self):
        self.configure()
        model_module.container = None
        status = await kv_cache_status()
        self.assertTrue(status.configured)
        self.assertFalse(status.generator_loaded)
        self.assertIsNone(status.last_save)


class KvSaveConfigTests(unittest.TestCase):
    def test_kv_save_config_parses_from_section(self):
        cfg = TabbyConfigModel.model_validate({
            "kv_save": {"store_dir": "/tmp/kv-cache-tabby/test-id", "stash_budget_mb": 256},
        })
        self.assertEqual(cfg.kv_save.store_dir, "/tmp/kv-cache-tabby/test-id")
        self.assertEqual(cfg.kv_save.stash_budget_mb, 256)

    def test_kv_save_config_defaults_off(self):
        cfg = TabbyConfigModel.model_validate({})
        self.assertIsNone(cfg.kv_save.store_dir)
        self.assertEqual(cfg.kv_save.stash_budget_mb, 512)


if __name__ == "__main__":
    unittest.main()
