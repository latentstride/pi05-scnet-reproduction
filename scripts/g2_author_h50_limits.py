"""Host-owned author H50 budget and receipt clock."""

from dataclasses import asdict, dataclass
from math import isfinite
from time import monotonic


MODE = "author_native_h50"
WINDOW = 50
MAX_ACTIONS = 800
EPISODE = "author-native-h50-seed100000"


@dataclass(frozen=True)
class AuthorBudget:
    physics_steps: int = 130_000
    soft_s: int = 2400
    hard_s: int = 2700
    reap_s: int = 10
    request_s: int = 390

    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for v in asdict(self).values()):
            raise ValueError("author budget requires positive integer fields")
        if not self.request_s <= self.soft_s < self.hard_s - self.reap_s:
            raise ValueError("author budget lacks a safe stop and reap interval")


BUDGET = AuthorBudget()


class ResourceCensored(ValueError):
    def __init__(self, reason, *, request_in_flight=False):
        super().__init__(reason)
        self.reason = reason
        self.request_in_flight = request_in_flight


@dataclass(frozen=True)
class TrialClock:
    budget: AuthorBudget
    started: float
    hard: float

    def __post_init__(self):
        if (
            not isfinite(self.started)
            or not isfinite(self.hard)
            or self.hard != self.started + self.budget.hard_s
        ):
            raise ValueError("author trial clock differs from host hard allowance")

    @property
    def soft(self):
        return self.started + self.budget.soft_s

    @property
    def reap_at(self):
        return self.hard - self.budget.reap_s

    def validate_entry(self):
        now = monotonic()
        if now < self.started:
            raise ValueError("author host clock starts in the future")
        self.remaining()

    def remaining(self):
        left = self.soft - monotonic()
        if left <= 0:
            raise ResourceCensored("soft_wall_clock")
        return left

    def receipt(self):
        return dict(
            started=self.started,
            soft=self.soft,
            hard=self.hard,
            reap_at=self.reap_at,
            budget=asdict(self.budget),
        )
