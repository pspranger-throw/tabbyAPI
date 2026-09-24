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
"""

import asyncio
import os

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

    async def request_save(self, store, stash_budget_mb = None):
        self.save_calls.append((store, stash_budget_mb))
        if self.save_delay:
            await asyncio.sleep(self.save_delay)
        if self.save_error is not None:
            raise self.save_error
        return self.save_result

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


@pytest.fixture
def kv_config(tmp_path, monkeypatch):
    kv = KvSaveConfig(store_dir = str(tmp_path / "session"), stash_budget_mb = 256)
    monkeypatch.setattr(config, "kv_save", kv)
    return kv


@pytest.fixture
def kv_unconfigured(monkeypatch):
    monkeypatch.setattr(config, "kv_save", KvSaveConfig())
    return config.kv_save


# --------------------------------------------------------------------------- #
# Save policy matrix                                                            #
# --------------------------------------------------------------------------- #


async def test_not_configured_refuses_and_keeps_set(kv_unconfigured):
    gen = FakeGenerator()
    container = make_container(gen)
    code, record = await container.kv_save()
    assert code == 422
    assert record["status"] == "refused"
    assert "not configured" in record["reason"]
    assert gen.save_calls == []


async def test_no_generator_refuses_and_keeps_set(kv_config):
    container = make_container(None)
    code, record = await container.kv_save()
    assert code == 422
    assert record["status"] == "refused"
    assert "no generator" in record["reason"]


async def test_latched_generator_refuses_and_keeps_set(kv_config):
    gen = FakeGenerator()
    gen.error = RuntimeError("latched")
    container = make_container(gen)
    code, record = await container.kv_save()
    assert code == 422
    assert record["status"] == "refused"
    assert "latched" in record["reason"]
    assert gen.save_calls == []


async def test_busy_skips_immediately_without_waiting(kv_config):
    gen = FakeGenerator(save_delay = 5.0)
    container = make_container(gen)
    container.active_job_ids["req-1"] = object()
    code, record = await container.kv_save()
    assert code == 409
    assert record["status"] == "busy"
    assert "1 active job" in record["reason"]
    assert gen.save_calls == [], "D12: a busy save must not queue a snapshot"


async def test_saved_200_carries_d8_observability(kv_config):
    gen = FakeGenerator()
    container = make_container(gen)
    code, record = await container.kv_save()
    assert code == 200
    assert record["status"] == "saved"
    assert record["n_pages"] == 512
    assert record["n_stashes"] == 4
    assert record["bytes"] == 3145728 + 2048 + 262144
    assert record["save_ms"] is not None and record["save_ms"] >= 0
    assert record["store_dir"] == kv_config.store_dir
    assert gen.save_calls == [(kv_config.store_dir, 256)], "store + configured stash budget must be passed"
    # The status surface mirrors the outcome for drill asserts
    assert container.kv_status()["last_save"]["status"] == "saved"


async def test_zero_stash_is_skipped_not_dead_set(kv_config):
    """User decision 2026-09-24: zero-stash saves = skip + loud log, no dead sets."""
    gen = FakeGenerator(save_result = {
        "dir": None, "skipped": "zero-stash", "n_pages": 3, "n_stashes": 0,
        "stash_keys": [], "checks": None, "meta": None, "deepest_anchor_page_idx": -1,
    })
    container = make_container(gen)
    code, record = await container.kv_save()
    assert code == 422
    assert record["status"] == "skipped"
    assert record["reason"] == "zero-stash"
    assert record["n_pages"] == 3
    assert record["n_stashes"] == 0
    assert container.kv_status()["last_save"]["reason"] == "zero-stash"


async def test_engine_exception_is_definitive_failure(kv_config):
    gen = FakeGenerator(save_error = RuntimeError("io exploded"))
    container = make_container(gen)
    code, record = await container.kv_save()
    assert code == 500
    assert record["status"] == "error"
    assert "io exploded" in record["reason"]


async def test_save_timeout_is_definitive_failure(kv_config, monkeypatch):
    gen = FakeGenerator(save_delay = 5.0)
    container = make_container(gen)
    monkeypatch.setattr(container, "KV_SAVE_TIMEOUT_S", 0.05)
    code, record = await container.kv_save()
    assert code == 504
    assert record["status"] == "error"
    assert "timed out" in record["reason"]


async def test_save_endpoint_maps_status_codes(kv_config, monkeypatch):
    gen = FakeGenerator(save_error = RuntimeError("boom"))
    container = make_container(gen)
    monkeypatch.setattr(model_module, "container", container)
    with pytest.raises(HTTPException) as exc:
        await kv_cache_save()
    assert exc.value.status_code == 500

    gen2 = FakeGenerator(save_result = {"skipped": "zero-stash", "n_pages": 0, "n_stashes": 0})
    container2 = make_container(gen2)
    monkeypatch.setattr(model_module, "container", container2)
    with pytest.raises(HTTPException) as exc2:
        await kv_cache_save()
    assert exc2.value.status_code == 422
    assert "zero-stash" in exc2.value.detail


# --------------------------------------------------------------------------- #
# Load-time restore (G4: exception-isolated, fail-closed to COLD)               #
# --------------------------------------------------------------------------- #


def test_restore_not_configured(kv_unconfigured):
    gen = FakeGenerator()
    container = make_container(gen)
    container._kv_restore_on_create()
    assert gen.restore_calls == []
    assert container.kv_status()["restore"]["reason"] == "not-configured"


def test_restore_no_store_present_is_cold_start(kv_config):
    gen = FakeGenerator()
    container = make_container(gen)
    container._kv_restore_on_create()
    assert gen.restore_calls == []
    rec = container.kv_status()["restore"]
    assert rec["attempted"] is False
    assert rec["reason"] == "no-store"


def test_restore_ok_records_counts(kv_config):
    os.makedirs(kv_config.store_dir, exist_ok = True)
    gen = FakeGenerator()
    container = make_container(gen)
    container._kv_restore_on_create()
    assert gen.restore_calls == [kv_config.store_dir]
    rec = container.kv_status()["restore"]
    assert rec["ok"] is True
    assert rec["pages_restored"] == 512
    assert rec["stashes_restored"] == 4
    assert rec["deepest_anchor_page_idx"] == 500


def test_restore_failure_is_fail_closed_and_quarantines_set(kv_config):
    os.makedirs(kv_config.store_dir, exist_ok = True)
    gen = FakeGenerator(restore_error = RuntimeError("digest mismatch"))
    container = make_container(gen)
    container._kv_restore_on_create()  # must not raise — the id keeps serving
    rec = container.kv_status()["restore"]
    assert rec["ok"] is False
    assert rec["attempted"] is True
    assert rec["reason"].startswith("rejected:")
    assert "digest mismatch" in rec["reason"]
    # The live path is vacated (no re-queue of the same failure) but the set is kept for forensics
    assert not os.path.isdir(kv_config.store_dir)
    rejected = [p for p in os.listdir(os.path.dirname(kv_config.store_dir)) if ".rejected-" in p]
    assert len(rejected) == 1


# --------------------------------------------------------------------------- #
# Status surface + config parsing                                               #
# --------------------------------------------------------------------------- #


async def test_status_endpoint_reports_records(kv_config, monkeypatch):
    gen = FakeGenerator()
    container = make_container(gen)
    await container.kv_save()
    monkeypatch.setattr(model_module, "container", container)
    status = await kv_cache_status()
    assert status.configured is True
    assert status.store_dir == kv_config.store_dir
    assert status.stash_budget_mb == 256
    assert status.generator_loaded is True
    assert status.last_save.status == "saved"
    assert status.restore is not None


async def test_status_endpoint_without_container(kv_config, monkeypatch):
    monkeypatch.setattr(model_module, "container", None)
    status = await kv_cache_status()
    assert status.configured is True
    assert status.generator_loaded is False
    assert status.last_save is None


def test_kv_save_config_parses_from_section():
    cfg = TabbyConfigModel.model_validate({
        "kv_save": {"store_dir": "/tmp/kv-cache-tabby/test-id", "stash_budget_mb": 256},
    })
    assert cfg.kv_save.store_dir == "/tmp/kv-cache-tabby/test-id"
    assert cfg.kv_save.stash_budget_mb == 256


def test_kv_save_config_defaults_off():
    cfg = TabbyConfigModel.model_validate({})
    assert cfg.kv_save.store_dir is None
    assert cfg.kv_save.stash_budget_mb == 512
