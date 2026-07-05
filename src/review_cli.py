#!/usr/bin/env python3
"""
review_cli.py

Anki-style review CLI.  Cycles through items provided by an external SRS API.
The review loop is fully decoupled — it only depends on a ReviewItem protocol
and an on_grade callback; it knows nothing about the database or SRS internals.

Controls:
    space   – reveal answer
    1-6     – grade the item  (1=F|C … 6=No Idea)
    q       – quit
"""

from __future__ import annotations

import math
import sys
import time
import tty
import termios
from dataclasses import dataclass
from enum import Enum
from typing import Protocol


# ── item protocol (what the review loop needs) ──────────────────────────

class ReviewItem(Protocol):
    @property
    def item_id(self) -> int: ...
    @property
    def front(self) -> str: ...
    @property
    def back(self) -> str: ...
    @property
    def requires_input(self) -> bool: ...
    @property
    def expected_input(self) -> str | None: ...
    @property
    def review_mode(self) -> str: ...  # "intro" or "review"
    @property
    def translation_en(self) -> str: ...
    @property
    def skill_type(self) -> str: ...
    @property
    def difficulty(self) -> float: ...


# ── grade enum ───────────────────────────────────────────────────────────

class Grade(Enum):
    FLUENT_CORRECT    = 1   # F|C — fluent, effortless recall
    NONFLUENT_CORRECT = 2   # N|C — correct but had to think
    HARD_CORRECT      = 3   # H|C — correct but hard to retrieve
    EASY_WRONG        = 4
    HARD_WRONG        = 5
    NO_IDEA           = 6

    @property
    def label(self) -> str:
        _LABELS = {1: "F|C", 2: "N|C", 3: "H|C", 4: "Easy|W", 5: "Hard|W", 6: "No|Idea"}
        return _LABELS[self.value]

    @property
    def is_correct(self) -> bool:
        return self in (Grade.FLUENT_CORRECT, Grade.NONFLUENT_CORRECT, Grade.HARD_CORRECT)

    @property
    def key(self) -> str:
        return str(self.value)


KEY_TO_GRADE = {g.key: g for g in Grade}


# ── SRS protocol (what the CLI needs from any SRS backend) ─────────────

@dataclass
class SessionCounts:
    unseen: int             # words in new pool (never seen)
    learning: int           # words in learning queue
    learning_capacity: int  # max learning queue size
    reviews_due: int        # review-queue skills that are due right now
    reviews_locked: int = 0 # skills past interval but sibling-locked
    # phase-specific due counts
    due_maintenance: int = 0
    due_acquiring: int = 0
    due_repair: int = 0
    # progress tracking
    target_done: int = 0    # graduated this session
    target_total: int = 0   # total target new words
    total_graded: int = 0
    correct_count: int = 0


class SRSProvider(Protocol):
    def session_counts(self) -> SessionCounts: ...
    def is_session_complete(self) -> bool: ...
    def seconds_until_next_due(self) -> float: ...
    def next_due_item(self) -> ReviewItem | None: ...
    def check_answer(self, item: ReviewItem, answer: str) -> bool: ...
    def submit_intro(self, item: ReviewItem) -> None: ...
    def submit_review(self, item: ReviewItem, grade: Grade) -> None: ...
    def edit_and_skip(self, item: ReviewItem, new_front: str, new_back: str) -> None: ...
    def delete_variant_and_skip(self, item: ReviewItem) -> None: ...
    def add_variant_and_skip(self, item: ReviewItem, new_front: str, new_back: str) -> None: ...
    def quarantine_and_skip(self, item: ReviewItem) -> None: ...


# ── mock SRS for testing ───────────────────────────────────────────────

@dataclass(frozen=True)
class DummyItem:
    item_id: int
    front: str
    back: str
    requires_input: bool = False
    expected_input: str | None = None
    review_mode: str = "review"
    translation_en: str = ""
    skill_type: str = ""
    difficulty: float = 0.0


class MockSRS:
    def __init__(self) -> None:
        self._counts = SessionCounts(unseen=20, learning=3, learning_capacity=3, reviews_due=10, target_total=20)
        self._due_items: list[DummyItem] = [
            DummyItem(
                item_id=1,
                front="What is the Korean word for 'occurrence'?",
                back="발생",
                requires_input=True,
                expected_input="발생",
            ),
        ] + [
            DummyItem(item_id=i, front=f"Question {i}", back=f"Answer {i}")
            for i in range(2, 11)
        ]

    def session_counts(self) -> SessionCounts:
        return self._counts

    def is_session_complete(self) -> bool:
        return not self._due_items

    def seconds_until_next_due(self) -> float:
        return 0.0

    def next_due_item(self) -> DummyItem | None:
        if not self._due_items:
            return None
        return self._due_items[0]

    def check_answer(self, item: ReviewItem, answer: str) -> bool:
        if isinstance(item, DummyItem) and item.expected_input is not None:
            return answer.strip() == item.expected_input
        return False

    def submit_intro(self, item: ReviewItem) -> None:
        pass

    def submit_review(self, item: ReviewItem, grade: Grade) -> None:
        if grade.is_correct:
            self._due_items = [i for i in self._due_items if i.item_id != item.item_id]
            self._counts.reviews -= 1
        else:
            # wrong: rotate to the back of the due queue
            removed = [i for i in self._due_items if i.item_id == item.item_id][0]
            self._due_items = [i for i in self._due_items if i.item_id != item.item_id]
            self._due_items.append(removed)

    def edit_and_skip(self, item: ReviewItem, new_front: str, new_back: str) -> None:
        pass

    def delete_variant_and_skip(self, item: ReviewItem) -> None:
        pass

    def add_variant_and_skip(self, item: ReviewItem, new_front: str, new_back: str) -> None:
        pass

    def quarantine_and_skip(self, item: ReviewItem) -> None:
        pass


# ── terminal helpers ────────────────────────────────────────────────────

def read_key() -> str:
    """Read a single keypress in raw mode."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


def clear_screen() -> None:
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


# ── ANSI color helpers ─────────────────────────────────────────────────

GREY  = "\033[90m"
LIGHT_BLUE = "\033[38;2;42;105;232m"
RESET = "\033[0m"

def grey(text: str) -> str:
    return f"{GREY}{text}{RESET}"

def light_blue(text: str) -> str:
    return f"{LIGHT_BLUE}{text}{RESET}"

import re as _re
_HINT_RE = _re.compile(r'\s*(\[hint:.*\])\s*$')
_BRACKET_RE = _re.compile(r'\[\[(.+?)\]\]')

def _split_hint(front: str) -> tuple[str, str]:
    """Split trailing '[hint: ...]' from a front string. Returns (sentence, hint)."""
    m = _HINT_RE.search(front)
    if m:
        return front[:m.start()].rstrip(), m.group(1)
    return front, ""


BRIGHT_BLUE = "\033[38;2;3;136;252m"

def bright_blue(text: str) -> str:
    return f"{BRIGHT_BLUE}{text}{RESET}"


def _highlight_brackets(text: str) -> str:
    """Replace [[word]] with bright-blue colored word (no brackets)."""
    return _BRACKET_RE.sub(lambda m: bright_blue(m.group(1)), text)


def _print_front(item: ReviewItem, indent: str = "  ") -> None:
    """Print the card front with cosmetic formatting per skill type."""
    if "prod" in item.skill_type:
        sentence, hint = _split_hint(item.front)
        print(f"{indent}{sentence}")
        if hint:
            print(grey(f"{indent}{hint}"))
        if item.translation_en:
            print(grey(f"{indent}{item.translation_en}"))
    elif "recog" in item.skill_type:
        print(f"{indent}{_highlight_brackets(item.front)}")
    else:
        print(f"{indent}{item.front}")
    if item.review_mode == "review" and item.difficulty >= 0.4:
        print(grey(f"{indent}>> Write down your answer first"))


def _progress_bar(done: int, total: int, width: int = 20) -> str:
    if total == 0:
        return f"[{'─' * width}] 0/0"
    filled = round(width * done / total)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {done}/{total}"


# ── review loop ─────────────────────────────────────────────────────────

GRADE_PROMPT = "  ".join(f"{g.value}) {g.label}" for g in Grade)


def _input_prefilled(prompt: str, prefill: str) -> str:
    """input() but with the text buffer pre-filled so the user can edit in place."""
    import readline
    def hook():
        readline.insert_text(prefill)
        readline.redisplay()
    readline.set_startup_hook(hook)
    try:
        return input(prompt)
    finally:
        readline.set_startup_hook()


def _edit_card(srs: SRSProvider, item: ReviewItem) -> None:
    """Prompt user to edit front/back in-place, then tell the SRS to save and skip."""
    clear_screen()
    print("  === Edit Card ===\n")
    new_front = _input_prefilled("  front: ", item.front)
    new_back  = _input_prefilled("  back:  ", item.back)

    srs.edit_and_skip(item, new_front, new_back)
    print("\n  Saved. Card skipped (no grade).")
    time.sleep(0.5)


def _delete_variant(srs: SRSProvider, item: ReviewItem) -> None:
    """Confirm and delete the current variant, then skip."""
    clear_screen()
    print("  === Delete Variant ===\n")
    print(f"  front: {item.front}")
    print(f"  back:  {item.back}\n")
    print(grey("  Delete this variant? [y/n]"))
    key = read_key()
    if key == "y":
        srs.delete_variant_and_skip(item)
        print("\n  Deleted. Card skipped (no grade).")
    else:
        print("\n  Cancelled.")
    time.sleep(0.5)


def _add_variant(srs: SRSProvider, item: ReviewItem) -> None:
    """Prompt for a new variant, add it to the same item, then skip."""
    clear_screen()
    print("  === Add Variant ===\n")
    new_front = input("  front: ")
    new_back  = input("  back:  ")
    if not new_front.strip() and not new_back.strip():
        print("\n  Empty variant — cancelled.")
        time.sleep(0.5)
        return
    srs.add_variant_and_skip(item, new_front, new_back)
    print("\n  Added. Card skipped (no grade).")
    time.sleep(0.5)


def _quarantine_word(srs: SRSProvider, item: ReviewItem) -> None:
    """Confirm and quarantine (suspend) the entire lexeme, then skip."""
    clear_screen()
    print("  === Quarantine Word ===\n")
    print(f"  front: {item.front}")
    print(f"  back:  {item.back}\n")
    print(grey("  Quarantine this word? [y/n]"))
    key = read_key()
    if key == "y":
        srs.quarantine_and_skip(item)
        print("\n  Quarantined. Word suspended from all reviews.")
    else:
        print("\n  Cancelled.")
    time.sleep(0.5)


def _format_countdown(seconds: float) -> str:
    """Format seconds as dd:hh:mm:ss, dropping leading zero segments."""
    total = int(seconds)
    d, rem = divmod(total, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    if d > 0:
        return f"{d}d {h:02d}:{m:02d}:{s:02d}"
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _print_status_bar(srs: SRSProvider) -> None:
    c = srs.session_counts()
    print(grey(f"  Due now:  {c.reviews_due}"))
    print(grey(f"  Target:   {_progress_bar(c.target_done, c.target_total)}  Learning: {c.learning}/{c.learning_capacity}"))
    print(grey("=" * 50))


def _print_summary(srs: SRSProvider, elapsed: float) -> None:
    c = srs.session_counts()
    clear_screen()
    _print_status_bar(srs)
    mins, secs = divmod(int(elapsed), 60)
    accuracy = f"{c.correct_count / c.total_graded * 100:.0f}%" if c.total_graded else "—"
    print()
    print(grey("  ── Session Summary ──"))
    print(grey(f"  Cards graded:    {c.total_graded}"))
    print(grey(f"  Accuracy:        {accuracy}"))
    print(grey(f"  Words graduated: {c.target_done}"))
    print(grey(f"  Time:            {mins}m {secs:02d}s"))
    print()


def _show_reading_question(prompt: str) -> None:
    """Generate and display an LLM reading comprehension question, wait for space to dismiss."""
    import llm_query

    clear_screen()
    print(grey("  Generating reading question...\n"))
    llm_query.query_api(prompt, verbose=False)

    print(grey("\n  [space] Return to review"))
    while True:
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        if ch in (" ", "\x03"):
            break


def run_review(srs: SRSProvider, *, reading_prompt: str = "") -> None:
    t0 = time.monotonic()

    while True:
        item = srs.next_due_item()
        if item is None:
            if srs.is_session_complete():
                break
            # items on cooldown — show a waiting screen
            wait = srs.seconds_until_next_due()
            clear_screen()
            _print_status_bar(srs)
            print(grey(f"\n  Next due in {_format_countdown(wait)}"))
            if reading_prompt:
                print(grey("  [r] Reading question"))
            print(grey("  [q] Quit"))
            # poll: check for keypress while waiting
            deadline = time.monotonic() + wait
            while time.monotonic() < deadline:
                fd = sys.stdin.fileno()
                old = termios.tcgetattr(fd)
                try:
                    tty.setraw(fd)
                    import select as _sel
                    ready, _, _ = _sel.select([fd], [], [], 0.5)
                    if ready:
                        ch = sys.stdin.read(1)
                        if ch in ("q", "\x03"):
                            termios.tcsetattr(fd, termios.TCSADRAIN, old)
                            _print_summary(srs, time.monotonic() - t0)
                            return
                        if ch == "r" and reading_prompt:
                            termios.tcsetattr(fd, termios.TCSADRAIN, old)
                            _show_reading_question(reading_prompt)
                            break  # re-enter main loop (may have become due)
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old)
                # update countdown display
                remaining = max(0, deadline - time.monotonic())
                clear_screen()
                _print_status_bar(srs)
                print(grey(f"\n  Next due in {_format_countdown(remaining)}"))
                if reading_prompt:
                    print(grey("  [r] Reading question"))
                print(grey("  [q] Quit"))
            continue

        if item.review_mode == "intro":
            # ---- intro: show front + back, spacebar to continue ----
            clear_screen()
            _print_status_bar(srs)
            print(f"\n  {light_blue('[Intro]')}\n")
            _print_front(item)
            print()
            print(grey("─" * 50))
            print(f"\n  {item.back}")
            if item.translation_en and "prod" not in item.skill_type:
                print(grey(f"  {item.translation_en}"))
            print()
            print(grey("─" * 50))
            print(grey("  [e] Edit   [d] Delete   [a] Add   [x] Quarantine   [q] Quit"))
            print()
            print(grey("  [space] Next"))

            while True:
                key = read_key()
                if key == " ":
                    srs.submit_intro(item)
                    break
                if key == "e":
                    _edit_card(srs, item)
                    break
                if key == "d":
                    _delete_variant(srs, item)
                    break
                if key == "a":
                    _add_variant(srs, item)
                    break
                if key == "x":
                    _quarantine_word(srs, item)
                    break
                if key in ("q", "\x03"):
                    _print_summary(srs, time.monotonic() - t0)
                    return
            continue

        # ---- front side ----
        clear_screen()
        _print_status_bar(srs)
        if item.review_mode == "gate":
            print(f"\n  {grey('[Gate]')}")
        elif item.review_mode == "repair":
            print(f"\n  {grey('[Repair]')}")
        print()
        _print_front(item)
        print()
        print(grey("─" * 50))

        if item.requires_input:
            # typed-answer card
            answer = input("  Your answer: ")
            if answer.strip().lower() == "q":
                _print_summary(srs, time.monotonic() - t0)
                return
            correct = srs.check_answer(item, answer)

            clear_screen()
            _print_status_bar(srs)
            print()
            _print_front(item)
            print()
            print(grey("─" * 50))
            print(grey(f"\n  Your answer: {answer}"))
            if correct:
                print(grey("  >> Correct!"))
            else:
                print(grey(f"  >> Wrong — expected: {item.back}"))
            print(f"\n  {item.back}")
            if item.translation_en and "prod" not in item.skill_type:
                print(grey(f"  {item.translation_en}"))
            print()
            print(grey("─" * 50))
        else:
            # normal flashcard: spacebar to reveal
            print(grey("  [e] Edit   [d] Delete   [a] Add   [x] Quarantine   [q] Quit"))
            print()
            print(grey("  [space] Show answer"))

            skipped = False
            while True:
                key = read_key()
                if key == " ":
                    break
                if key == "e":
                    _edit_card(srs, item)
                    skipped = True
                    break
                if key == "d":
                    _delete_variant(srs, item)
                    skipped = True
                    break
                if key == "a":
                    _add_variant(srs, item)
                    skipped = True
                    break
                if key == "x":
                    _quarantine_word(srs, item)
                    skipped = True
                    break
                if key in ("q", "\x03"):
                    _print_summary(srs, time.monotonic() - t0)
                    return
            if skipped:
                continue

            clear_screen()
            _print_status_bar(srs)
            print()
            _print_front(item)
            print()
            print(grey("─" * 50))
            print(f"\n  {item.back}")
            if item.translation_en and "prod" not in item.skill_type:
                print(grey(f"  {item.translation_en}"))
            print()
            print(grey("─" * 50))

        # ---- grade ----
        print(grey("  [e] Edit   [d] Delete   [a] Add   [x] Quarantine   [q] Quit"))
        print()
        print(grey(f"  {GRADE_PROMPT}"))

        while True:
            key = read_key()
            if key in KEY_TO_GRADE:
                grade = KEY_TO_GRADE[key]
                srs.submit_review(item, grade)
                # wrong grade → retype prompt (fluency building)
                if not grade.is_correct:
                    print()
                    print(grey("  Type the answer to continue:"))
                    input("  ")
                break
            if key == "e":
                _edit_card(srs, item)
                break
            if key == "d":
                _delete_variant(srs, item)
                break
            if key == "a":
                _add_variant(srs, item)
                break
            if key == "x":
                _quarantine_word(srs, item)
                break
            if key in ("q", "\x03"):
                _print_summary(srs, time.monotonic() - t0)
                return

    _print_summary(srs, time.monotonic() - t0)


# ── entry point ─────────────────────────────────────────────────────────

def _show_pre_session_menu(session, prompts_path: str, n_variants: int) -> bool:
    """
    Pre-session menu loop. Returns True to start review, False to quit.
    Allows adding words before the session starts.
    """
    from pathlib import Path
    import srs_db
    import llm_query
    from add_word import word_exists, show_definition_and_confirm, generate_and_insert

    prompts = llm_query.load_prompts(Path(prompts_path))

    while True:
        seen, unseen = srs_db.classify_lexeme_groups(session)
        now = srs_db.now_utc()

        # compute due count and next-due time across all seen lexemes
        due_count = 0
        nearest_due = None
        for group in seen.values():
            dues = []
            for item in group["items"]:
                if item.srs_state is not None and item.srs_state.due_at is not None:
                    d = item.srs_state.due_at
                    if d.tzinfo is None:
                        d = d.replace(tzinfo=srs_db.UTC)
                    dues.append(d)
            if dues:
                earliest = min(dues)
                if earliest <= now:
                    due_count += 1
                if nearest_due is None or earliest < nearest_due:
                    nearest_due = earliest

        print(f"\n=== SRS Review ===")
        print(f"  Total lexemes:   {len(seen) + len(unseen)}")
        print(f"  Reviewing:       {len(seen)}")
        print(f"  Unseen:          {len(unseen)}")
        if due_count > 0:
            print(f"  Due now:         {due_count}")
        elif nearest_due is not None:
            gap = max(0.0, (nearest_due - now).total_seconds())
            print(f"  Due now:         0  (next in {_format_countdown(gap)})")
        print()
        print(grey("  [Enter] Start review"))
        print(grey("  [a]     Add a word"))
        print(grey("  [q]     Quit"))
        print()

        cmd = input("> ").strip().lower()

        if cmd == "q":
            return False
        elif cmd == "a":
            word = input("  Word to add: ").strip()
            if not word:
                continue
            if word_exists(session, word):
                print(f"  '{word}' already exists in the database.")
                continue

            confirmed, parsed, def_block = show_definition_and_confirm(prompts, word)
            if not confirmed:
                print("  Skipped.")
                continue

            ok, msg = generate_and_insert(
                session,
                prompts=prompts,
                word=word,
                def_block=def_block,
                n_variants=n_variants,
            )
            if ok:
                print(f"\n  {msg}")
            else:
                print(f"\n  Error: {msg}")
        elif cmd == "":
            return True


def main() -> None:
    import argparse
    from pathlib import Path
    from sqlalchemy.orm import Session as SASession
    import srs_db
    from lexeme_srs import LexemeSRS

    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="test_srs.sqlite", help="Path to SQLite DB")
    ap.add_argument("--target-new", type=int, default=0, help="Number of new lexemes to learn")
    ap.add_argument("--queue-size", type=int, default=3, help="Active learning queue size")
    ap.add_argument("--intro-examples", type=int, default=3, help="Number of intro examples per lexeme")
    ap.add_argument("--prompts", default=str(Path(__file__).parent / "prompts.yaml"), help="Path to prompts YAML")
    ap.add_argument("--n-variants", type=int, default=8, help="Number of card variants to generate")
    ap.add_argument("--reading-prompt", default=str(Path(__file__).parent / "reading_question.yaml"), help="Path to reading question YAML")
    args = ap.parse_args()

    engine = srs_db.make_engine(args.db)
    srs_db.init_db(engine)

    with SASession(engine) as session:
        # pre-session menu: add words here, before LexemeSRS init
        should_start = _show_pre_session_menu(session, args.prompts, args.n_variants)
        if not should_start:
            print("Bye.")
            return

        # initialize SRS after all word additions
        srs = LexemeSRS(
            session,
            target_new=args.target_new,
            queue_size=args.queue_size,
            intro_examples=args.intro_examples,
        )

        # load reading question prompt
        import yaml
        reading_prompt = ""
        rp_path = Path(args.reading_prompt)
        if rp_path.exists():
            with open(rp_path, encoding="utf-8") as f:
                reading_prompt = yaml.safe_load(f).get("prompt", "")

        counts = srs.session_counts()
        print(f"\n  Unseen: {counts.unseen}  |  Learning: {counts.learning}/{counts.learning_capacity}  |  Due now: {counts.reviews_due}")
        print()

        run_review(srs, reading_prompt=reading_prompt)


if __name__ == "__main__":
    main()
