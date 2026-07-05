import os
import time
import yaml
from openai import OpenAI
from pathlib import Path
import json

import re
from typing import Tuple, Optional, Dict, List, Any
from pprint import pprint


# globals

class _LazyClient:
    """Defer OpenAI() construction until first use.

    Constructing the client at import time crashes the whole app (and even
    --help) when OPENAI_API_KEY is unset. This proxy builds the real client on
    first attribute access and raises a clear, actionable error if the key is
    missing. Call sites keep using ``client.chat...`` / ``client.responses...``.
    """

    _client = None

    def __getattr__(self, name):
        if _LazyClient._client is None:
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError(
                    "OPENAI_API_KEY is not set. Copy .env.example to .env, add "
                    "your OpenAI key, then `source .env` before running."
                )
            _LazyClient._client = OpenAI()
        return getattr(_LazyClient._client, name)


client = _LazyClient()

BASE_DIR: Path = Path(__file__).parent
PROMPTS: Path = BASE_DIR.joinpath("prompts.yaml")


def load_prompts(
    prompt_file: Path
) -> dict:
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompts = yaml.safe_load(f)
    return prompts


def query_api(
    input_str: str,
    system_prompt: str = "You are a helpful AI assistant",
    model="gpt-5-mini",
    verbose=True,
):
    if verbose:
        print("querying GPT...")

    start = time.time()
    stream = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": input_str
            }
        ],
        stream=True,
    )
    chunks = []
    for event in stream:
        # Text arrives in small chunks ("deltas")
        if event.type == "response.output_text.delta":
            chunks.append(event.delta)
            print(event.delta, end="", flush=True)
    print()  # newline after stream finishes
    end = time.time()
    elapsed = end - start
    print(f'elapsed: {elapsed:0.3f}')

    output_text = "".join(chunks)
    return output_text


def _parse_quiz_output(text: str) -> Tuple[str, str, Optional[str]]:
    """
    Parse quiz output into:
      - question (str)
      - answer (single letter 'a'..'e', lowercase)
      - explanation (str or None)

    Accepts answer formats like:
      ANSWER: a
      ANSWER: (a)
      ANSWER - A
      answer: (E)
    Accepts EXPLANATION / Explanation / explanation.
    """
    text = text.strip()

    # --- ANSWER: accept optional parentheses and loose punctuation ---
    # Matches lines like:
    #   ANSWER: a
    #   ANSWER: (a)
    #   ANSWER - A
    #   Answer – (e)
    answer_match = re.search(
        r"""(?im)            # case-insensitive, multiline
        ^\s*answer\s*        # 'answer' at start of a line
        [:\-–]\s*            # :, -, or en-dash
        \(?\s*([a-e])\s*\)?  # optional ( ), capture a-e
        \s*$                 # end of line
        """,
        text,
        re.VERBOSE,
    )
    if not answer_match:
        raise ValueError("Could not find an ANSWER line with a letter a-e")

    answer = answer_match.group(1).lower()

    # --- QUESTION = everything before the ANSWER line ---
    question = text[:answer_match.start()].strip()

    # --- EXPLANATION (optional, case-insensitive, multiline) ---
    explanation_match = re.search(
        r"(?is)^\s*explanation\s*[:\-–]\s*(.*)$",
        text,
        re.MULTILINE,
    )
    explanation = explanation_match.group(1).strip() if explanation_match else None
    return question, answer, explanation



def parse_quiz_output(text: str) -> Dict[str, Optional[str]]:
    """
    Attempts to parse quiz output.
    If parsing fails, returns raw text instead of raising.
    """
    try:
        question, answer, explanation = _parse_quiz_output(text)
        return {
            "ok": True,
            "question": question,
            "answer": answer,
            "explanation": explanation,
            "raw": None,
        }
    except Exception as e:
        return {
            "ok": False,
            "question": None,
            "answer": None,
            "explanation": None,
            "raw": text,
            "error": str(e),
        }


def parse_vocab_block(text: str) -> Dict[str, Any]:
    """
    Parse model output in the format:

    TERM: <term>

    DEF_EN:
    <english definition>

    DEF_KO:
    <korean definition>

    EXAMPLES:
    1. <korean sentence>
    2. <korean sentence>
    ...

    Returns dict:
      {
        "ok": bool,
        "term": str,
        "def_en": str,
        "def_ko": str,
        "examples": List[str],
        "raw": str,          # original text (always included)
        "error": Optional[str]
      }
    """
    raw = text
    t = text.strip()

    out: Dict[str, Any] = {
        "ok": False,
        "term": "",
        "def_en": "",
        "def_ko": "",
        "examples": [],
        "raw": raw,
        "error": None,
    }

    try:
        # TERM: ...
        m_term = re.search(r"(?im)^\s*TERM\s*:\s*(.+?)\s*$", t)
        if not m_term:
            raise ValueError("Missing 'TERM:' line")
        out["term"] = m_term.group(1).strip()

        # DEF_EN block (from DEF_EN: to next header or end)
        m_def_en = re.search(
            r"(?is)^\s*DEF_EN\s*:\s*\n(.*?)(?=^\s*DEF_KO\s*:|^\s*EXAMPLES\s*:|\Z)",
            t,
            re.MULTILINE,
        )
        out["def_en"] = (m_def_en.group(1).strip() if m_def_en else "")

        # DEF_KO block
        m_def_ko = re.search(
            r"(?is)^\s*DEF_KO\s*:\s*\n(.*?)(?=^\s*EXAMPLES\s*:|\Z)",
            t,
            re.MULTILINE,
        )
        out["def_ko"] = (m_def_ko.group(1).strip() if m_def_ko else "")

        # EXAMPLES block: collect numbered lines after EXAMPLES:
        m_examples = re.search(r"(?im)^\s*EXAMPLES\s*:\s*$", t)
        if not m_examples:
            raise ValueError("Missing 'EXAMPLES:' header")

        after = t[m_examples.end():]

        # Grab lines like "1. ...." or "2) ...." or "3 : ...."
        examples: List[str] = []
        for line in after.splitlines():
            line = line.strip()
            if not line:
                continue
            m_line = re.match(r"^\s*(\d+)\s*[\.\)\:]\s*(.+?)\s*$", line)
            if m_line:
                examples.append(m_line.group(2).strip())

        if len(examples) < 2:
            raise ValueError(f"Expected 2+ examples, got {len(examples)}")

        out["examples"] = examples
        out["ok"] = True
        return out

    except Exception as e:
        out["error"] = str(e)
        return out

def test_grade_output(prompts):
    sample_output = "오늘 버스 정류장에 갔는데 사람이 정말 많았어요."
    instruction = prompts["system_prompts"]["grade_output"]
    query = prompts["queries"]["grade_output"]
    query += f" {sample_output}"

    start = time.time()
    _ = query_api(query, instruction)
    end = time.time()
    print(f'query time: {end - start}')


def test_create_paragraph(prompts):
    vocab_word = "발생하다"
    instruction = prompts["system_prompts"]["create_text"]
    query = prompts["queries"]["create_paragraph"]
    query += f" {vocab_word}"

    start = time.time()
    _ = query_api(query, instruction)
    end = time.time()
    print(f'query time: {end - start}')


def test_create_story(prompts):
    vocab_word = "발생하다"
    instruction = prompts["system_prompts"]["create_text"]
    query = prompts["queries"]["create_story"]
    query += f" {vocab_word}"

    start = time.time()
    _ = query_api(query, instruction)
    end = time.time()
    print(f'query time: {end - start}')


def test_create_mc(prompts):
    vocab_word = "발생하다"
    instruction = prompts["system_prompts"]["create_MC_tuned"]
    query = prompts["queries"]["create_vocab_multiple_choice"]
    query += f" {vocab_word}"

    start = time.time()
    raw_output_text = query_api(query, instruction)
    end = time.time()
    print(f'query time: {end - start}')

    result = parse_quiz_output(raw_output_text)

    if result["ok"]:
        pass
    else:
        print("Could not parse quiz format.")
    pprint(result)

# -----------------------------
# Shared lexeme/diagnostics schema fragments
# -----------------------------

LEXEME_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["headword", "pos", "sense_id", "gloss_en", "gloss_ko", "notes"],
    "properties": {
        "headword": {"type": "string"},
        "pos": {"type": "string", "enum": ["noun", "verb", "adj", "adv", "bound_noun", "phrase", "other"]},
        "sense_id": {"type": "string"},
        "gloss_en": {"type": "string"},
        "gloss_ko": {"type": "string"},
        "notes": {"type": "string"},
    },
}

DIAGNOSTICS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["used_examples", "skipped_examples", "why_skipped", "warnings"],
    "properties": {
        "used_examples": {"type": "array", "items": {"type": "integer"}},
        "skipped_examples": {"type": "array", "items": {"type": "integer"}},
        "why_skipped": {"type": "array", "items": {"type": "string"}},
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
}

# -----------------------------
# Bundle schemas (per prompt)
# -----------------------------

# One bundled cloze_prod card with 5–10 variants.
SRS_SCHEMA_CLOZE_PROD_BUNDLE = {
    "name": "srs_cloze_prod_bundle",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["lexeme", "cards", "diagnostics"],
        "properties": {
            "lexeme": LEXEME_SCHEMA,
            "cards": {
                "type": "array",
                "minItems": 1,
                "maxItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["card_id", "type", "tags", "variants"],
                    "properties": {
                        "card_id": {"type": "string"},
                        "type": {"type": "string", "const": "cloze_prod"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "variants": {
                            "type": "array",
                            "minItems": 5,
                            "maxItems": 10,
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["front", "back", "accepted_answers", "hint_en", "source_example_index"],
                                "properties": {
                                    "front": {"type": "string"},
                                    "back": {"type": "string"},
                                    "hint_en": {"type": "string"},
                                    "accepted_answers": {
                                        "type": "array",
                                        "minItems": 1,
                                        "maxItems": 1,
                                        "items": {"type": "string"},
                                    },
                                    "source_example_index": {"type": "integer"},
                                },
                            },
                        },
                    },
                },
            },
            "diagnostics": DIAGNOSTICS_SCHEMA,
        },
    },
    "strict": True,
}

# One bundled cloze_recog card with 5–10 variants.
SRS_SCHEMA_CLOZE_RECOG_BUNDLE = {
    "name": "srs_cloze_recog_bundle",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["lexeme", "cards", "diagnostics"],
        "properties": {
            "lexeme": LEXEME_SCHEMA,
            "cards": {
                "type": "array",
                "minItems": 1,
                "maxItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["card_id", "type", "tags", "variants"],
                    "properties": {
                        "card_id": {"type": "string"},
                        "type": {"type": "string", "const": "cloze_recog"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "variants": {
                            "type": "array",
                            "minItems": 5,
                            "maxItems": 10,
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["front", "back", "source_example_index"],
                                "properties": {
                                    "front": {"type": "string"},
                                    "back": {"type": "string"},
                                    "source_example_index": {"type": "integer"},
                                },
                            },
                        },
                    },
                },
            },
            "diagnostics": DIAGNOSTICS_SCHEMA,
        },
    },
    "strict": True,
}

# One bundled diff card with 15–25 variants.
SRS_SCHEMA_DIFF_BUNDLE = {
    "name": "srs_diff_bundle",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["word_a", "word_b", "cards", "diagnostics"],
        "properties": {
            "word_a": {"type": "string"},
            "word_b": {"type": "string"},
            "cards": {
                "type": "array",
                "minItems": 1,
                "maxItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["card_id", "type", "variants"],
                    "properties": {
                        "card_id": {"type": "string"},
                        "type": {"type": "string", "const": "diff"},
                        "variants": {
                            "type": "array",
                            "minItems": 8,
                            "maxItems": 15,
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["front", "back", "correct_choice", "translation_en"],
                                "properties": {
                                    "front": {"type": "string"},
                                    "back": {"type": "string"},
                                    "correct_choice": {"type": "string", "enum": ["a", "b"]},
                                    "translation_en": {"type": "string"},
                                },
                            },
                        },
                    },
                },
            },
            "diagnostics": DIAGNOSTICS_SCHEMA,
        },
    },
    "strict": True,
}

# -----------------------------
# Helper: run one prompt kind
# -----------------------------

BUNDLE_MODEL = "gpt-4.1-mini"  # default model for bundle generation (Korean sentences)


def _run_bundle(prompts, *, kind: str, def_block: str, n_variants: int = 8, model: str = BUNDLE_MODEL):
    """
    kind: "create_static_cloze_prod_bundle" or "create_static_cloze_recog_bundle"
    """
    instruction = prompts["system_prompts"][kind]
    query = prompts["queries"][kind]

    user_prompt = query.format(
        definition_block=def_block,
        n_variants=n_variants,
    )

    schema = {
        "create_static_cloze_prod_bundle": SRS_SCHEMA_CLOZE_PROD_BUNDLE,
        "create_static_cloze_recog_bundle": SRS_SCHEMA_CLOZE_RECOG_BUNDLE,
    }[kind]

    start = time.time()
    resp = client.chat.completions.create(
        model=model,
        # temperature=0.2,
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_schema", "json_schema": schema},
    )
    elapsed = time.time() - start
    print(f"{kind} query time: {elapsed:0.3f}s")

    payload = json.loads(resp.choices[0].message.content)

    # Extra sanity checks beyond JSON schema (optional but helpful)
    card = payload["cards"][0]
    if kind.endswith("cloze_prod_bundle"):
        # ensure each front has exactly one blank and accepted_answers == [back]
        for v in card["variants"]:
            assert v["front"].count("(   )") == 1, v["front"]
            assert v["accepted_answers"] == [v["back"]], (v["accepted_answers"], v["back"])
    return payload


def _run_diff_bundle(
    prompts,
    *,
    word_a: str,
    def_a: str,
    word_b: str,
    def_b: str,
    n_variants: int = 10,
    model: str = BUNDLE_MODEL,
):
    """Generate a diff bundle for a confusable word pair."""
    instruction = prompts["system_prompts"]["create_diff_bundle"]
    query = prompts["queries"]["create_diff_bundle"]

    user_prompt = query.format(
        word_a=word_a,
        def_a=def_a,
        word_b=word_b,
        def_b=def_b,
        n_variants=n_variants,
    )

    start = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_schema", "json_schema": SRS_SCHEMA_DIFF_BUNDLE},
    )
    elapsed = time.time() - start
    print(f"create_diff_bundle query time: {elapsed:0.3f}s")

    payload = json.loads(resp.choices[0].message.content)

    # Sanity check: at least 30% of variants target each word
    card = payload["cards"][0]
    variants = card["variants"]
    a_count = sum(1 for v in variants if v["correct_choice"] == "a")
    b_count = len(variants) - a_count
    total = len(variants)
    if a_count < total * 0.3 or b_count < total * 0.3:
        print(f"WARNING: skewed diff distribution a={a_count} b={b_count} (total={total})")

    return payload


def test_create_static_cloze_prod_bundle(prompts):
    def_block = """
시끄럽다
English: to be noisy; to be loud or bothersome because of sound.
Korean: 소리나 말이 커서 불편하거나 귀찮다.
Examples:
1. 아이들이 놀이터에서 시끄럽게 논다.
2. 이 근처는 밤에도 차 소리가 시끄럽다.
3. 텔레비전 소리가 너무 시끄러워서 집중할 수 없다.
"""
    return _run_bundle(
        prompts,
        kind="create_static_cloze_prod_bundle",
        def_block=def_block,
        n_variants=8,
    )


def test_create_static_cloze_recog_bundle(prompts):
    def_block = """
발생하다
English: to occur; to happen; to arise
Korean: 어떤 일이 일어나거나 생기다
Examples:
1. 지진으로 큰 피해가 발생했다.
2. 회의 중 문제가 발생하면 알려 주세요.
3. 컴퓨터 오류로 데이터 손실이 발생할 수 있다.
4. 추가 비용이 발생했다.
"""
    return _run_bundle(
        prompts,
        kind="create_static_cloze_recog_bundle",
        def_block=def_block,
        n_variants=8,
    )


def main():
    prompts = load_prompts(PROMPTS)

    payload_prod = test_create_static_cloze_prod_bundle(prompts)
    payload_recog = test_create_static_cloze_recog_bundle(prompts)

    from pprint import pprint
    print("\n--- CLOZE_PROD BUNDLE ---")
    pprint(payload_prod)
    print("\n--- CLOZE_RECOG BUNDLE ---")
    pprint(payload_recog)

    breakpoint()


if __name__ == "__main__":
    main()


