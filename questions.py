"""
Question Database Manager — Loads, stores, and updates benchmark questions.

Questions are stored in questions_db.json (source of truth) and optionally
indexed in ChromaDB for semantic search and similarity queries.

Monthly update workflow:
  1. Edit questions_db.json directly (add/modify/remove questions)
  2. Run: python3 questions.py --validate    # check consistency
  3. Run: python3 questions.py --reindex     # rebuild ChromaDB index
  4. Run: python3 questions.py --stats       # show distribution
  5. Run: python3 questions.py --export-py   # regenerate Python checker code

Or programmatically:
  from questions import QuestionDB
  db = QuestionDB()
  questions = db.get_category("PHYSICS")
  easy = db.get_by_difficulty(1)
  all_q = db.get_all()
"""
from __future__ import annotations

import json
import re
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

DB_FILE = Path(__file__).parent / "questions_db.json"
CHROMA_DIR = Path(__file__).parent / "questions_chroma"


class QuestionDB:
    """Manages the benchmark question database."""

    def __init__(self, db_path: Path = DB_FILE):
        self.db_path = db_path
        self.data = self._load()
        self._chroma = None

    def _load(self) -> dict:
        if self.db_path.exists():
            return json.loads(self.db_path.read_text())
        return {"version": "4.0", "categories": {}, "total_questions": 0}

    def save(self):
        """Save changes back to JSON."""
        self.data["total_questions"] = sum(
            len(cat["questions"]) for cat in self.data["categories"].values()
        )
        self.data["total_categories"] = len(self.data["categories"])
        self.data["last_updated"] = datetime.now().strftime("%Y-%m-%d")
        self.db_path.write_text(json.dumps(self.data, indent=2, ensure_ascii=False))

    # ── Queries ──────────────────────────────────────────────────────────

    def get_all(self) -> Dict[str, list]:
        """Get all questions grouped by category."""
        return {cat: info["questions"]
                for cat, info in self.data["categories"].items()}

    def get_category(self, category: str) -> list:
        """Get all questions for a category."""
        return self.data["categories"].get(category, {}).get("questions", [])

    def get_by_difficulty(self, level: int) -> Dict[str, list]:
        """Get questions filtered by difficulty (1=easy, 2=moderate, 3=hard)."""
        result = {}
        for cat, info in self.data["categories"].items():
            filtered = [q for q in info["questions"] if q["difficulty"] == level]
            if filtered:
                result[cat] = filtered
        return result

    def get_categories(self) -> List[str]:
        return list(self.data["categories"].keys())

    def count(self) -> int:
        return sum(len(info["questions"])
                   for info in self.data["categories"].values())

    # ── Mutations ────────────────────────────────────────────────────────

    def add_question(self, category: str, question: dict):
        """Add a question to a category."""
        if category not in self.data["categories"]:
            self.data["categories"][category] = {
                "count": 0, "easy": 0, "moderate": 0, "hard": 0,
                "questions": [],
            }
        cat = self.data["categories"][category]
        cat["questions"].append(question)
        cat["count"] = len(cat["questions"])
        d = question.get("difficulty", 1)
        if d == 1: cat["easy"] += 1
        elif d == 2: cat["moderate"] += 1
        elif d == 3: cat["hard"] += 1

    def add_category(self, name: str, questions: list):
        """Add a new category with questions."""
        e = sum(1 for q in questions if q.get("difficulty") == 1)
        m = sum(1 for q in questions if q.get("difficulty") == 2)
        h = sum(1 for q in questions if q.get("difficulty") == 3)
        self.data["categories"][name] = {
            "count": len(questions),
            "easy": e, "moderate": m, "hard": h,
            "questions": questions,
        }

    def remove_category(self, name: str):
        self.data["categories"].pop(name, None)

    # ── Validation ───────────────────────────────────────────────────────

    def validate(self) -> list:
        """Validate the database. Returns list of issues."""
        issues = []
        seen_ids = set()

        for cat, info in self.data["categories"].items():
            qs = info.get("questions", [])
            if not qs:
                issues.append(f"{cat}: empty category")
                continue

            for q in qs:
                # Check required fields
                for field in ["id", "difficulty", "prompt", "max_tokens"]:
                    if field not in q:
                        issues.append(f"{cat}/{q.get('id','?')}: missing '{field}'")

                # Duplicate IDs
                qid = q.get("id", "")
                if qid in seen_ids:
                    issues.append(f"{cat}/{qid}: duplicate ID")
                seen_ids.add(qid)

                # Difficulty range
                d = q.get("difficulty", 0)
                if d not in (1, 2, 3):
                    issues.append(f"{cat}/{qid}: invalid difficulty {d}")

                # Empty prompt
                if not q.get("prompt", "").strip():
                    issues.append(f"{cat}/{qid}: empty prompt")

            # Check distribution
            e = sum(1 for q in qs if q.get("difficulty") == 1)
            m = sum(1 for q in qs if q.get("difficulty") == 2)
            h = sum(1 for q in qs if q.get("difficulty") == 3)
            if e < 3: issues.append(f"{cat}: only {e} easy questions (need 3+)")
            if m < 3: issues.append(f"{cat}: only {m} moderate questions (need 3+)")
            if h < 3: issues.append(f"{cat}: only {h} hard questions (need 3+)")

        return issues

    # ── ChromaDB indexing ────────────────────────────────────────────────

    def reindex_chroma(self):
        """Build/rebuild ChromaDB index for semantic search."""
        try:
            import chromadb
        except ImportError:
            print("chromadb not installed — pip install chromadb")
            return

        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))

        # Delete and recreate
        try:
            client.delete_collection("questions")
        except:
            pass
        col = client.create_collection("questions")

        ids, docs, metas = [], [], []
        for cat, info in self.data["categories"].items():
            for q in info["questions"]:
                ids.append(q["id"])
                docs.append(f"{q['prompt']} [{q.get('explanation', '')}]")
                metas.append({
                    "category": cat,
                    "difficulty": q["difficulty"],
                    "difficulty_label": q.get("difficulty_label", ""),
                    "explanation": q.get("explanation", ""),
                    "max_tokens": q.get("max_tokens", 100),
                })

        # Batch insert
        batch = 5000
        for i in range(0, len(ids), batch):
            col.add(
                ids=ids[i:i+batch],
                documents=docs[i:i+batch],
                metadatas=metas[i:i+batch],
            )
        print(f"Indexed {len(ids)} questions in ChromaDB at {CHROMA_DIR}")

    def search_similar(self, query: str, n: int = 5) -> list:
        """Find similar questions by semantic search."""
        if not self._chroma:
            try:
                import chromadb
                client = chromadb.PersistentClient(path=str(CHROMA_DIR))
                self._chroma = client.get_collection("questions")
            except:
                return []

        results = self._chroma.query(query_texts=[query], n_results=n)
        return list(zip(results["ids"][0], results["documents"][0],
                        results["metadatas"][0]))

    # ── Stats ────────────────────────────────────────────────────────────

    def print_stats(self):
        total = self.count()
        cats = len(self.data["categories"])
        print(f"\nQuestion Database: {total} questions, {cats} categories")
        print(f"Last updated: {self.data.get('last_updated', 'unknown')}")
        print(f"\n{'Category':30s} {'Total':>5} {'Easy':>5} {'Mod':>5} {'Hard':>5}")
        print("-" * 60)
        for cat, info in sorted(self.data["categories"].items()):
            print(f"  {cat:28s} {info['count']:>5} {info.get('easy',0):>5} "
                  f"{info.get('moderate',0):>5} {info.get('hard',0):>5}")
        print("-" * 60)
        print(f"  {'TOTAL':28s} {total:>5}")

    # ── Monthly Update Helper ────────────────────────────────────────────

    def monthly_update_template(self, category: str) -> str:
        """Generate a template for adding new questions to a category."""
        template = f"""
# Monthly update template for {category}
# Add 9 questions: 3 easy (d=1) + 3 moderate (d=2) + 3 hard (d=3)
# Paste into questions_db.json under categories.{category}.questions

[
  {{"id": "{category[:3].lower()}_new1", "difficulty": 1, "difficulty_label": "easy",
   "prompt": "YOUR EASY QUESTION HERE",
   "max_tokens": 30, "explanation": "What this tests"}},

  {{"id": "{category[:3].lower()}_new2", "difficulty": 1, "difficulty_label": "easy",
   "prompt": "YOUR EASY QUESTION HERE",
   "max_tokens": 30, "explanation": "What this tests"}},

  {{"id": "{category[:3].lower()}_new3", "difficulty": 1, "difficulty_label": "easy",
   "prompt": "YOUR EASY QUESTION HERE",
   "max_tokens": 30, "explanation": "What this tests"}},

  {{"id": "{category[:3].lower()}_new4", "difficulty": 2, "difficulty_label": "moderate",
   "prompt": "YOUR MODERATE QUESTION HERE",
   "max_tokens": 80, "explanation": "What this tests"}},

  {{"id": "{category[:3].lower()}_new5", "difficulty": 2, "difficulty_label": "moderate",
   "prompt": "YOUR MODERATE QUESTION HERE",
   "max_tokens": 80, "explanation": "What this tests"}},

  {{"id": "{category[:3].lower()}_new6", "difficulty": 2, "difficulty_label": "moderate",
   "prompt": "YOUR MODERATE QUESTION HERE",
   "max_tokens": 80, "explanation": "What this tests"}},

  {{"id": "{category[:3].lower()}_new7", "difficulty": 3, "difficulty_label": "hard",
   "prompt": "YOUR HARD QUESTION HERE",
   "max_tokens": 150, "explanation": "What this tests"}},

  {{"id": "{category[:3].lower()}_new8", "difficulty": 3, "difficulty_label": "hard",
   "prompt": "YOUR HARD QUESTION HERE",
   "max_tokens": 150, "explanation": "What this tests"}},

  {{"id": "{category[:3].lower()}_new9", "difficulty": 3, "difficulty_label": "hard",
   "prompt": "YOUR HARD QUESTION HERE",
   "max_tokens": 150, "explanation": "What this tests"}}
]
"""
        return template


# ── CLI ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Question Database Manager")
    p.add_argument("--validate", action="store_true", help="Validate database")
    p.add_argument("--stats", action="store_true", help="Show statistics")
    p.add_argument("--reindex", action="store_true", help="Rebuild ChromaDB index")
    p.add_argument("--search", type=str, help="Search for similar questions")
    p.add_argument("--template", type=str, help="Generate monthly update template for category")
    p.add_argument("--add-category", type=str, help="Add new empty category")
    args = p.parse_args()

    db = QuestionDB()

    if args.stats:
        db.print_stats()
    elif args.validate:
        issues = db.validate()
        if issues:
            print(f"\n{len(issues)} issues found:")
            for i in issues:
                print(f"  ✗ {i}")
        else:
            print("\n✓ Database is valid")
    elif args.reindex:
        db.reindex_chroma()
    elif args.search:
        results = db.search_similar(args.search)
        for qid, doc, meta in results:
            print(f"  [{meta['category']}/{meta['difficulty_label']}] {doc[:100]}")
    elif args.template:
        print(db.monthly_update_template(args.template))
    elif args.add_category:
        db.add_category(args.add_category, [])
        db.save()
        print(f"Added empty category: {args.add_category}")
    else:
        db.print_stats()
