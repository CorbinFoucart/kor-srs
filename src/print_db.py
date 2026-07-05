#!/usr/bin/env python3
"""Print all items stored in the SRS database, grouped by lexeme.
Shows all variants per card from the content JSON."""

import argparse
import json
from sqlalchemy.orm import Session

import srs_db
from lexeme_srs import LexemeSRS


def print_lexeme_groups(session: Session) -> None:
    groups = srs_db.group_items_by_lexeme(session)
    print(f"Lexemes in DB: {len(groups)}\n")

    for lexeme, group in sorted(groups.items()):
        print(f"  {lexeme}")
        for item in group["items"]:
            parsed = srs_db.parse_external_id(item.external_id)
            skill = parsed[1] if parsed else "?"
            content = item.content or {}
            variants = content.get("variants", [])
            print(f"    [{skill}]  id={item.id}  ({len(variants)} variants)")
            for i, v in enumerate(variants):
                print(f"      {i+1}) front: {v.get('front', '?')}")
                print(f"         back:  {v.get('back', '?')}")
                if v.get("translation_en"):
                    print(f"         en:    {v['translation_en']}")
        print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="test_srs.sqlite")
    args = ap.parse_args()
    engine = srs_db.make_engine(args.db)

    with Session(engine) as s:
        print_lexeme_groups(s)


if __name__ == "__main__":
    main()
