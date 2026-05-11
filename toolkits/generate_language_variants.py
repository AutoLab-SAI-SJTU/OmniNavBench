#!/usr/bin/env python3
"""
Generate Language Variant Split (version4 style).
For each episode JSON, produce 4 variants in separate top-level directories:
  - original/       : original wording
  - concise/        : concise rewrite
  - verbose/        : verbose rewrite
  - first_person/   : first-person rewrite

Key differences from original script:
  1. Output structure: target/{variant}/{agent_type}/{scene_id}/final_episode_X.json
     instead of: target/{scene_type}/{agent_type}/{scene_id}/final_episode_X{suffix}.json
  2. Optionally strips top-level scene type directories (commercial_scenes, home_scenes, etc.)
  3. Matches version4_test directory structure

Strategy: Only rewrite sub_instructions via LLM, then concatenate to form main instruction.
This guarantees instruction == join(sub_instructions) always.

Usage:
    export API_KEY="your-key"

    # Keep scene type directories (commercial_scenes, home_scenes, etc.)
    python generate_variants_v4_style.py --source /path/to/version2_split/train --target /path/to/version4_train

    # Strip scene type directories (like version4_test)
    python generate_variants_v4_style.py --source /path/to/version2_split/train --target /path/to/version4_train --strip-scene-type

    # Parallel processing
    python generate_variants_v4_style.py --source ... --target ... --workers 4
"""

import os
import sys
import re
import json
import copy
import time
import argparse
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Config ───────────────────────────────────────────────────────────────────

API_URL = os.environ.get("API_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
API_KEY = os.environ.get("API_KEY", "")
MODEL = os.environ.get("MODEL", "qwen3-max-2026-01-23")

VARIANT_KEYS = ["original", "concise", "verbose", "first_person"]

FIRST_PERSON_PREFIX = "You are now the robot navigating. Here is what you need to do: "

# Scene type directories to strip (if --strip-scene-type is enabled)
SCENE_TYPE_DIRS = {"commercial_scenes", "home_scenes", "matterport_usd"}


# ── API ──────────────────────────────────────────────────────────────────────

def call_llm(prompt: str, max_retries: int = 3, timeout: int = 120) -> str | None:
    if not API_KEY:
        sys.exit("ERROR: Set API_KEY env var.")

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
    }

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(API_URL, headers=headers, json=body, timeout=timeout)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"    API error (attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                time.sleep(2 * attempt)
    return None


def parse_json_response(text: str) -> dict | None:
    if not text:
        return None

    # Strip <think>...</think> (qwen3 thinking mode)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # Strip markdown fences
    if "```" in text:
        lines = []
        inside = False
        for line in text.split("\n"):
            if line.strip().startswith("```"):
                inside = not inside
                continue
            if inside:
                lines.append(line)
        text = "\n".join(lines)

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None

    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError as e:
        print(f"    JSON parse error: {e}")
        return None


# ── Prompt ───────────────────────────────────────────────────────────────────

def build_prompt(sub_instructions: list[dict], objects: dict = None, room_zone: dict = None) -> str:
    """Only rewrite sub_instructions. Main instruction will be concatenated from results."""

    # Protected entities (skip "origin" which is just a coordinate label)
    protected = []
    if objects:
        protected.extend(k for k in objects.keys() if k != "origin")
    if room_zone:
        protected.extend(room_zone.keys())
    protected_str = ", ".join(f'"{p}"' for p in protected)

    sub_lines = [{"step": s["step"], "type": s["type"], "text": s["text"]} for s in sub_instructions]
    sub_json = json.dumps(sub_lines, ensure_ascii=False, indent=2)

    return f"""You are a navigation instruction rewriter. Rewrite the following sub_instructions into 3 style variants.

## PROTECTED ENTITIES (must appear EXACTLY as-is, never paraphrase or rephrase):
{protected_str}

## Original sub_instructions:
{sub_json}

## Type guide:
- SOCIAL: social behavior constraint — rewrite style only
- VLN: navigation — keep ALL direction words (left/right/straight/turn) and landmarks exactly
- OBJ: object search — keep object name and room name exactly
- EQA: question — keep question semantics and answer-relevant keywords identical

## Rewrite rules:

### concise:
- Remove filler: "please", "politely", "then", "after that"
- Keep every direction, protected entity, and action verb
- Each sub must be SHORTER than original

### verbose:
- Add ONLY transition phrases between clauses: "After that,", "Next,", "Once you arrive,"
- Do NOT invent details not in the original
- Each sub must be LONGER than original

### first_person:
- Rewrite to first person: "I need to...", "I should..."
- Keep all details identical

## Output:
Return ONLY this JSON. Exactly {len(sub_instructions)} sub_instructions per variant.
{{
  "concise": ["step0 text", "step1 text", ...],
  "verbose": ["step0 text", "step1 text", ...],
  "first_person": ["step0 text", "step1 text", ...]
}}

/no_think"""


# ── Core Logic ───────────────────────────────────────────────────────────────

def concat_instruction(sub_texts: list[str]) -> str:
    """Concatenate sub_instruction texts into a main instruction."""
    parts = []
    for text in sub_texts:
        text = text.strip()
        if not text:
            continue
        # Ensure proper sentence ending
        if text[-1] not in ".!?":
            text += "."
        parts.append(text)
    return " ".join(parts)


def extract_task(data: dict) -> dict | None:
    """Extract first scenario's task."""
    for scenario in data.get("scenarios", []):
        task = scenario.get("task")
        if task and task.get("navigation", {}).get("instruction"):
            return task
    return None


def apply_variant(data: dict, variant_key: str, variant_sub_texts: list[str]) -> dict:
    """
    Deep-copy data, write variant sub_texts into sub_instructions,
    then concatenate to form main instruction. Guarantees consistency.
    """
    new_data = copy.deepcopy(data)

    for scenario in new_data.get("scenarios", []):
        task = scenario.get("task")
        if not task:
            continue
        nav = task.get("navigation", {})
        subs = task.get("sub_instructions", [])

        if len(variant_sub_texts) != len(subs):
            print(f"    WARNING: sub count mismatch (got {len(variant_sub_texts)}, expected {len(subs)})")

        # Write back sub_instructions
        for i, sub in enumerate(subs):
            if i < len(variant_sub_texts):
                sub["text"] = variant_sub_texts[i]

        # Concatenate to main instruction (guaranteed consistency)
        main = concat_instruction(variant_sub_texts)
        if variant_key == "first_person":
            main = FIRST_PERSON_PREFIX + main
        nav["instruction"] = main

        break  # only first scenario

    return new_data


def compute_relative_path(src_path: Path, source_root: Path, strip_scene_type: bool) -> Path:
    """
    Compute the relative path for output, optionally stripping scene type directories.

    Examples:
        Input: commercial_scenes/car/SCENE_ID/final_episode_1.json

        strip_scene_type=False: commercial_scenes/car/SCENE_ID
        strip_scene_type=True:  car/SCENE_ID
    """
    rel_path = src_path.relative_to(source_root).parent

    if strip_scene_type and rel_path.parts:
        # Check if first part is a scene type directory
        if rel_path.parts[0] in SCENE_TYPE_DIRS:
            # Strip the first directory level
            rel_path = Path(*rel_path.parts[1:]) if len(rel_path.parts) > 1 else Path(".")

    return rel_path


def process_file(src_path: Path, file_num: int, total: int) -> bool:
    """
    Process one episode file → 4 variant files in separate directories.

    Output structure:
        target/original/{relative_path}/final_episode_X.json
        target/concise/{relative_path}/final_episode_X.json
        target/verbose/{relative_path}/final_episode_X.json
        target/first_person/{relative_path}/final_episode_X.json
    """
    stem = src_path.stem
    rel_display = src_path.relative_to(args.source)

    # Compute relative path (with optional scene type stripping)
    rel_path = compute_relative_path(src_path, args.source, args.strip_scene_type)

    # Check if all 4 variants already exist
    all_exist = all(
        (args.target / variant / rel_path / f"{stem}.json").exists()
        for variant in VARIANT_KEYS
    )
    if all_exist:
        print(f"[{file_num}/{total}] {rel_display} — skip (exists)")
        return True

    print(f"[{file_num}/{total}] {rel_display}", end=" ", flush=True)

    with open(src_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    task = extract_task(data)
    if not task:
        print("— skip (no instruction)")
        # Save original to all variant directories
        for variant in VARIANT_KEYS:
            _save_json(args.target / variant / rel_path / f"{stem}.json", data)
        return True

    nav = task["navigation"]
    sub_instructions = task.get("sub_instructions", [])
    objects = nav.get("objects", {})
    room_zone = nav.get("room_zone", {})

    # Call LLM — only rewrite sub_instructions
    prompt = build_prompt(sub_instructions, objects, room_zone)
    raw = call_llm(prompt)
    variants = parse_json_response(raw)

    if not variants:
        print("— API failed")
        # Save original to all variant directories
        for variant in VARIANT_KEYS:
            _save_json(args.target / variant / rel_path / f"{stem}.json", data)
        return False

    # Validate counts
    expected = len(sub_instructions)
    valid = True
    for key in ("concise", "verbose", "first_person"):
        got = len(variants.get(key, []))
        if got != expected:
            print(f"— {key} count mismatch ({got} vs {expected})", end=" ")
            valid = False

    if not valid:
        print("— skipping variants")
        # Save original to all variant directories
        for variant in VARIANT_KEYS:
            _save_json(args.target / variant / rel_path / f"{stem}.json", data)
        return False

    # Save 4 versions to separate directories
    # 1. Original
    _save_json(args.target / "original" / rel_path / f"{stem}.json", data)

    # 2-4. Variants
    for variant_key in ("concise", "verbose", "first_person"):
        sub_texts = variants.get(variant_key, [])
        new_data = apply_variant(data, variant_key, sub_texts)
        _save_json(args.target / variant_key / rel_path / f"{stem}.json", new_data)

    print("✓")
    return True


def _save_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ── File Discovery ───────────────────────────────────────────────────────────

def discover_files(source: Path) -> list[Path]:
    """Find all final_episode*.json recursively."""
    return sorted(source.rglob("final_episode*.json"))


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    global args
    parser = argparse.ArgumentParser(
        description="Generate navigation instruction language variants (version4 style)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Keep scene type directories
  python %(prog)s --source /path/to/version2_split/train --target /path/to/version4_train

  # Strip scene type directories (like version4_test)
  python %(prog)s --source /path/to/version2_split/train --target /path/to/version4_train --strip-scene-type

  # Parallel processing
  python %(prog)s --source ... --target ... --workers 4 --strip-scene-type
        """
    )
    parser.add_argument("--source", type=Path, required=True, help="Source directory containing episode files")
    parser.add_argument("--target", type=Path, required=True, help="Target directory for variant outputs")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers (default: 1)")
    parser.add_argument("--strip-scene-type", action="store_true",
                       help="Strip scene type directories (commercial_scenes, home_scenes, etc.) from output paths")
    parser.add_argument("--dry-run", action="store_true", help="List files without processing")
    args = parser.parse_args()

    files = discover_files(args.source)
    print(f"Found {len(files)} episode files under {args.source}")
    print(f"Strip scene type: {args.strip_scene_type}\n")

    if args.dry_run:
        print("Sample output paths:")
        for f in files[:5]:
            rel_path = compute_relative_path(f, args.source, args.strip_scene_type)
            print(f"  Source: {f.relative_to(args.source)}")
            for variant in VARIANT_KEYS:
                print(f"    → {variant}/{rel_path}/{f.name}")
            print()
        if len(files) > 5:
            print(f"  ... and {len(files) - 5} more files")
        return

    # Test API
    print("Testing API...", end=" ", flush=True)
    if call_llm("Say OK /no_think"):
        print("OK\n")
    else:
        print("FAILED")
        return

    args.target.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    success = 0

    if args.workers <= 1:
        for i, src in enumerate(files, 1):
            if process_file(src, i, len(files)):
                success += 1
            time.sleep(0.3)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {}
            for i, src in enumerate(files, 1):
                fut = pool.submit(process_file, src, i, len(files))
                futures[fut] = src
            for fut in as_completed(futures):
                if fut.result():
                    success += 1

    elapsed = time.time() - t0
    print(f"\nDone: {success}/{len(files)} files, {elapsed / 60:.1f} min")
    print(f"Output: {args.target}")
    print(f"\nDirectory structure:")
    for variant in VARIANT_KEYS:
        print(f"  {args.target}/{variant}/")


if __name__ == "__main__":
    main()
