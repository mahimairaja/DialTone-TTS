"""The reproducibility gate."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

__all__ = ["DEFAULT_TOLERANCE_PP", "VerifyOutcome", "verify_entry", "write_withheld"]

#: Allowed drift between two runs of the same entry, in percentage points.
DEFAULT_TOLERANCE_PP = 0.1


@dataclass(frozen=True)
class VerifyOutcome:
    system: str
    condition: str
    stored_wer: float
    rerun_wer: float
    tolerance_pp: float

    @property
    def drift_pp(self) -> float:
        return abs(self.rerun_wer - self.stored_wer) * 100

    @property
    def withheld(self) -> bool:
        """Withhold when the drift is *more than* the tolerance."""
        return self.drift_pp > self.tolerance_pp + 1e-9

    @property
    def reason(self) -> str:
        return (
            f"aggregate WER moved {self.drift_pp:.3f} percentage points between "
            f"runs ({self.stored_wer * 100:.2f}% then {self.rerun_wer * 100:.2f}%), "
            f"above the {self.tolerance_pp} point tolerance"
        )


def verify_entry(
    system: str,
    condition: str,
    stored_wer: float,
    rerun_wer: float,
    tolerance_pp: float = DEFAULT_TOLERANCE_PP,
) -> VerifyOutcome:
    return VerifyOutcome(
        system=system,
        condition=condition,
        stored_wer=stored_wer,
        rerun_wer=rerun_wer,
        tolerance_pp=tolerance_pp,
    )


def write_withheld(outcome: VerifyOutcome, root: Path) -> Path:
    """Record a withheld entry where it is visible but not part of the scorecard."""
    directory = root / "withheld"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{outcome.system}_{outcome.condition}.json"
    path.write_text(
        json.dumps(
            {
                "system": outcome.system,
                "condition": outcome.condition,
                "stored_wer": outcome.stored_wer,
                "rerun_wer": outcome.rerun_wer,
                "drift_pp": outcome.drift_pp,
                "tolerance_pp": outcome.tolerance_pp,
                "reason": outcome.reason,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return path
