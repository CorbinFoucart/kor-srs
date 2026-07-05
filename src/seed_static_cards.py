#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import os
from typing import Tuple

from sqlalchemy.orm import sessionmaker

import llm_query
import srs_db
from add_word import define_vocab_raw, build_definition_block, generate_and_insert

import json
import time
from pathlib import Path
from threading import Lock

LATENCY_LOG_PATH = Path("openai_latency_log.jsonl")
_LATENCY_LOCK = Lock()


def log_latency(
    *,
    request_type: str, 
    model: str,
    latency_s: float,
    extra: dict | None = None,
):
    record = {
        "ts": time.time(),
        "request_type": request_type,
        "model": model,
        "latency_s": latency_s,
    }
    if extra:
        record.update(extra)

    with _LATENCY_LOCK:
        with LATENCY_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


async def run_with_queue_timing(
    *,
    request_type: str,
    api_sem: asyncio.Semaphore,
    fn,                 # sync function to run in a thread
    fn_kwargs: dict,
    model: str,
    log_extra: dict,
):
    t0_total = time.perf_counter()

    async with api_sem:
        t0_api = time.perf_counter()
        result = await asyncio.to_thread(fn, **fn_kwargs)
        api_dt = time.perf_counter() - t0_api

    total_dt = time.perf_counter() - t0_total
    queue_wait = total_dt - api_dt

    log_latency(
        request_type=request_type,
        model=model,
        latency_s=api_dt,
        extra={**log_extra, "queue_wait_s": queue_wait, "total_s": total_dt},
    )
    return result



async def seed_one_word(
    word: str,
    *,
    prompts: dict,
    SessionLocal,
    api_sem: asyncio.Semaphore,
    n_variants: int,
    model: str = llm_query.BUNDLE_MODEL,
) -> Tuple[str, bool, str]:
    """
    Returns (word, ok, message).
    """
    try:
        # (1) define vocab (rate-limited) + log API latency and queue wait
        raw_def = await run_with_queue_timing(
            request_type="define_vocab",
            api_sem=api_sem,
            fn=define_vocab_raw,
            fn_kwargs={"prompts": prompts, "word": word},
            model=model,
            log_extra={"word": word},
        )

        parsed = llm_query.parse_vocab_block(raw_def)
        if not parsed.get("ok"):
            return (word, False, f"define_vocab parse failed: {parsed.get('error')}")

        def_block = build_definition_block(parsed)

        # (2) generate + translate + upsert (single function, run in thread)
        async with api_sem:
            with SessionLocal() as sess:
                ok, msg = await asyncio.to_thread(
                    generate_and_insert, sess,
                    prompts=prompts, word=word,
                    def_block=def_block, n_variants=n_variants,
                    model=model,
                )

        return (word, ok, msg)

    except Exception as e:
        return (word, False, str(e))


def _existing_headwords(SessionLocal) -> set[str]:
    """Return the set of headwords already in the DB."""
    with SessionLocal() as sess:
        items = sess.query(srs_db.Item).filter(
            srs_db.Item.item_type == "card",
            srs_db.Item.external_id.isnot(None),
            srs_db.Item.deleted_at.is_(None),
        ).all()
        headwords = set()
        for item in items:
            parsed = srs_db.parse_external_id(item.external_id)
            if parsed:
                headwords.add(parsed[0])
        return headwords


async def main_async(args):
    prompts = llm_query.load_prompts(Path(args.prompts))

    # words
    if args.words:
        words = args.words
    else:
        from test_vocab_bank import vocab
        words = [w for (w, _def) in vocab.items()]

    # DB init/reset
    if args.reset and os.path.exists(args.db):
        os.remove(args.db)

    engine = srs_db.make_engine(args.db)
    srs_db.init_db(engine)
    SessionLocal = sessionmaker(bind=engine, future=True)

    # skip words that already exist in the DB
    existing = _existing_headwords(SessionLocal)
    skipped = [w for w in words if w in existing]
    words = [w for w in words if w not in existing]
    for w in skipped:
        print(f"[SKIP] {w}: already in database")

    if not words:
        print("\nNo new words to add.")
        return

    api_sem = asyncio.Semaphore(args.max_inflight)

    tasks = [
        asyncio.create_task(
            seed_one_word(
                w,
                prompts=prompts,
                SessionLocal=SessionLocal,
                api_sem=api_sem,
                n_variants=args.n_variants,
                model=args.model,
            )
        )
        for w in words
    ]

    ok = 0
    for coro in asyncio.as_completed(tasks):
        word, success, msg = await coro
        if success:
            ok += 1
            print(f"[OK]  {word}: {msg}")
        else:
            print(f"[ERR] {word}: {msg}")

    print(f"\nDone: {ok}/{len(words)} new words added. {len(skipped)} skipped.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="test_srs.sqlite")
    ap.add_argument("--prompts", default=str(Path(__file__).parent / "prompts.yaml"))
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--max-inflight", type=int, default=5, help="Max concurrent OpenAI calls")
    ap.add_argument("--n-variants", type=int, default=8)
    ap.add_argument("--words", nargs="*", default=None)
    ap.add_argument("--model", default=llm_query.BUNDLE_MODEL, help="Model for bundle generation")
    args = ap.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()

