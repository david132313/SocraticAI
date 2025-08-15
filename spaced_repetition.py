# spaced_repetition.py
import json
import os
import tempfile
import logging
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from anki_sm_2 import Scheduler, Card, Rating

SECONDS_PER_DAY = 86400
SCHEMA_VERSION = 1  # persisted JSON schema version


class SM2SpacedRepetition:
    """
    SM-2 spaced repetition scheduler that integrates with GraspPredictor.
    Each concept becomes an Anki card with intelligent review scheduling.
    """

    def __init__(self):
        self.scheduler = Scheduler()
        self.concept_cards: Dict[str, Card] = {}  # concept_name -> Card
        self.logger = logging.getLogger(__name__)
        self._lock = threading.Lock()

    def initialize_concept_card(self, concept_name: str, difficulty: float) -> None:
        """
        Create a new SM-2 card for a concept.
        (difficulty currently unused by the underlying scheduler)
        """
        # Idempotent: do not overwrite existing cards (preserves interval/ease)
        if concept_name in self.concept_cards:
            return
        card = Card(due=datetime.now(timezone.utc))  # new cards due now
        self.concept_cards[concept_name] = card

    def convert_to_sm2_rating(self, understanding_level: float, response_quality: str) -> Rating:
        """
        Convert grasp analysis to SM-2 rating scale.
        Args:
            understanding_level: 0.0-1.0 from analyze/rating pipeline
            response_quality: "low"/"medium"/"high"
        Returns:
            Rating: Again(1), Hard(2), Good(3), or Easy(4)
        """
        ul = max(0.0, min(1.0, float(understanding_level)))
        rq = (response_quality or "medium").lower()

        # Calibrated bands: slightly wider near the middle to reduce jumpiness
        if ul < 0.28:
            base = Rating.Again   # Forgot/confused
        elif ul < 0.58:
            base = Rating.Hard    # Struggled but got it
        elif ul < 0.78:
            base = Rating.Good    # Solid understanding
        else:
            base = Rating.Easy    # Mastered it

        # Adjust based on response quality (one-step up/down, within bounds)
        if rq == "high" and base != Rating.Again:
            return Rating(min(4, base.value + 1))
        if rq == "low" and base != Rating.Easy:
            return Rating(max(1, base.value - 1))
        return base

    def review_concept(
        self,
        concept_name: str,
        understanding_level: float,
        response_quality: str,
        review_datetime: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Review a concept and update its SM-2 schedule.

        Args:
            concept_name: Name of the concept being reviewed
            understanding_level: 0.0-1.0 performance score
            response_quality: "low"/"medium"/"high" response quality
            review_datetime: When the review happened (defaults to now UTC)

        Returns:
            Dict with review results and next due date (includes interval/ease_factor)
        """
        with self._lock:
            if concept_name not in self.concept_cards:
                # Initialize if concept doesn't exist
                self.initialize_concept_card(concept_name, 3.0)

            # Normalize inputs
            ul = max(0.0, min(1.0, float(understanding_level)))
            rq = (response_quality or "medium").lower()

            card = self.concept_cards[concept_name]
            rating = self.convert_to_sm2_rating(ul, rq)
            if review_datetime is None:
                review_datetime = datetime.now(timezone.utc)

            # Review the card with SM-2 algorithm
            updated_card, review_log = self.scheduler.review_card(
                card=card,
                rating=rating,
                review_datetime=review_datetime
            )

            # Update stored card
            self.concept_cards[concept_name] = updated_card

            # Calculate time until next review (cap to sane range for UI/reporting)
            now_utc = datetime.now(timezone.utc)
            time_until_due = updated_card.due - now_utc
            days_until_due = time_until_due.total_seconds() / SECONDS_PER_DAY
            days_until_due = max(-365.0, min(365.0, days_until_due))

            return {
                "concept": concept_name,
                "rating": rating.value,
                "rating_name": rating.name,
                "next_due": updated_card.due.isoformat(),
                "days_until_due": days_until_due,
                "review_timestamp": review_log.review_datetime.isoformat(),
                "interval": getattr(updated_card, "interval", 0),
                "ease_factor": getattr(updated_card, "ease_factor", 2.5),
            }

    def get_due_concepts(self, current_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Get all concepts that are due for review.

        Args:
            current_time: Time to check against (defaults to now UTC)

        Returns:
            List of concept dictionaries sorted deterministically by urgency
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        due_concepts: List[Dict[str, Any]] = []

        for concept_name, card in self.concept_cards.items():
            if card.due <= current_time:
                overdue_seconds = (current_time - card.due).total_seconds()
                overdue_days = overdue_seconds / SECONDS_PER_DAY
                overdue_days = max(0.0, min(365.0, overdue_days))  # bound for sanity

                due_concepts.append({
                    "concept": concept_name,
                    "due_date": card.due.isoformat(),
                    "overdue_days": overdue_days,
                    "urgency_score": overdue_days + 1.0,  # Higher = more urgent
                    "interval": getattr(card, "interval", 0),
                    "ease_factor": getattr(card, "ease_factor", 2.5),
                })

        # Deterministic ordering: highest urgency first, then earliest due_date, then name
        due_concepts.sort(
            key=lambda x: (-float(x["urgency_score"]), x["due_date"], x["concept"])
        )
        return due_concepts

    def save_to_file(self, filepath: str) -> None:
        """Save SM-2 scheduler and cards to JSON file atomically."""
        data = {
            "version": SCHEMA_VERSION,
            "scheduler": self.scheduler.to_dict(),
            "concept_cards": {
                concept: card.to_dict()
                for concept, card in self.concept_cards.items()
            },
        }

        tmpname = None
        try:
            # Ensure directory exists
            dir_ = os.path.dirname(filepath) or "."
            os.makedirs(dir_, exist_ok=True)

            # Atomic write
            with tempfile.NamedTemporaryFile(
                "w", delete=False, dir=dir_, suffix=".tmp", encoding="utf-8"
            ) as tf:
                json.dump(data, tf, indent=2)
                tmpname = tf.name
            os.replace(tmpname, filepath)
            tmpname = None  # replaced successfully

        except Exception as e:
            self.logger.error(f"Error saving SM-2 data to {filepath}: {e}")
        finally:
            # Clean up temp file if replace failed
            if tmpname and os.path.exists(tmpname):
                try:
                    os.remove(tmpname)
                except Exception:
                    pass

    def load_from_file(self, filepath: str) -> None:
        """Load SM-2 scheduler and cards from JSON file."""
        with self._lock:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Version tolerance
                version = int(data.get("version", 1))
                if version > SCHEMA_VERSION:
                    self.logger.warning(
                        f"SM-2 data version {version} is newer than supported {SCHEMA_VERSION}. "
                        "Attempting best-effort load."
                    )

                # Restore scheduler
                self.scheduler = Scheduler.from_dict(data.get("scheduler", {}))

                # Restore cards
                cards_in = data.get("concept_cards", {}) or {}
                restored: Dict[str, Card] = {}
                for concept, card_data in cards_in.items():
                    try:
                        restored[concept] = Card.from_dict(card_data)
                    except Exception as e:
                        self.logger.error(f"Could not load card for '{concept}': {e}")
                self.concept_cards = restored

            except FileNotFoundError:
                self.logger.warning(f"SM-2 data file not found: {filepath}")
                self.scheduler = Scheduler()
                self.concept_cards = {}
            except Exception as e:
                self.logger.error(f"Error loading SM-2 data from {filepath}: {e}")
                self.scheduler = Scheduler()
                self.concept_cards = {}

    def get_all_concepts_status(self, current_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Get comprehensive status of all concepts in the SM-2 system.

        Args:
            current_time: Time to check against (defaults to now UTC)

        Returns:
            List of concept status dictionaries
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        status_list: List[Dict[str, Any]] = []

        for concept_name, card in self.concept_cards.items():
            time_until_due = card.due - current_time
            days_until_due = time_until_due.total_seconds() / SECONDS_PER_DAY
            days_until_due = max(-365.0, min(365.0, days_until_due))

            # Determine status
            if days_until_due <= 0:
                status = "Due Now"
            elif days_until_due < 1:
                status = "Due Soon"
            elif days_until_due < 7:
                status = "This Week"
            else:
                status = "Future"

            status_list.append({
                "concept": concept_name,
                "status": status,
                "due_date": card.due.isoformat(),
                "days_until_due": round(days_until_due, 1),
                "interval": getattr(card, "interval", 0),
                "ease_factor": round(getattr(card, "ease_factor", 2.5), 2),
                "review_count": getattr(card, "reviews", 0),
                "lapses": getattr(card, "lapses", 0),
            })

        # Sort by due soonest (increasing days until due)
        status_list.sort(key=lambda x: (float(x["days_until_due"]), x["due_date"], x["concept"]))
        return status_list

    # ----- Utility helpers -----

    def peek_next_due(self, concept_name: str) -> Optional[str]:
        """Return ISO timestamp for next due date of a concept (no mutation)."""
        card = self.concept_cards.get(concept_name)
        return card.due.isoformat() if card else None


# Convenience function for creating the SM-2 system
def create_sm2_scheduler() -> SM2SpacedRepetition:
    """Create an SM2SpacedRepetition instance."""
    return SM2SpacedRepetition()


# explicit public surface
__all__ = [
    "SM2SpacedRepetition",
    "create_sm2_scheduler",
]
