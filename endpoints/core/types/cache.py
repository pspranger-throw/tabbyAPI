from typing import Optional
from pydantic import BaseModel, Field


class KvSaveRecord(BaseModel):
    """Outcome of one POST /v1/cache/save attempt (also the `last_save` status field)."""

    status: Optional[str] = Field(
        None,
        description="saved | skipped | busy | refused | error",
    )
    reason: Optional[str] = Field(None, description="Machine-readable skip/refuse/failure reason")
    store_dir: Optional[str] = Field(None, description="Final store directory written or targeted")
    stash_budget_mb: Optional[int] = Field(None, description="Configured recurrent-checkpoint budget")
    save_ms: Optional[float] = Field(None, description="Wall time of the snapshot, in milliseconds")
    n_pages: int = Field(0, description="KV pages captured")
    n_stashes: int = Field(0, description="Recurrent (GDN) checkpoints captured")
    bytes: int = Field(0, description="Total bytes published (sum of the payload file sizes "
                                      "recorded in meta['files']; excludes meta.json itself)")
    at: Optional[float] = Field(None, description="unix timestamp of the attempt")


class KvRestoreRecord(BaseModel):
    """Outcome of the eager load-time restore (the `restore` status field)."""

    attempted: bool = Field(False, description="Whether a store was present to restore")
    ok: bool = Field(False, description="Whether restore succeeded (else serving COLD)")
    reason: Optional[str] = Field(None, description="not-configured | no-store | rejected: <detail>")
    store_dir: Optional[str] = Field(None, description="Store directory targeted")
    pages_restored: int = Field(0, description="KV pages restored into the live pool")
    stashes_restored: int = Field(0, description="Recurrent (GDN) checkpoints restored")
    deepest_anchor_page_idx: int = Field(-1, description="Deepest anchor page index in the set")
    at: Optional[float] = Field(None, description="unix timestamp of the attempt")


class KvCacheStatusResponse(BaseModel):
    """KV save/restore state — the drill-assertable surface (no log-grep needed)."""

    configured: bool = Field(False, description="kv_save.store_dir is set in this id's config")
    store_dir: Optional[str] = Field(None, description="Configured store directory")
    stash_budget_mb: Optional[int] = Field(None, description="Configured stash budget")
    generator_loaded: bool = Field(False, description="A generator exists (model loaded)")
    restore: Optional[KvRestoreRecord] = Field(None, description="Load-time restore outcome")
    last_save: Optional[KvSaveRecord] = Field(None, description="Most recent save attempt")
