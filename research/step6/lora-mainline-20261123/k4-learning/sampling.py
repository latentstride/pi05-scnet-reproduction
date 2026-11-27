"""Balance grasp transitions within the four rows executed per policy query."""

import numpy as np


EXECUTED_ROWS = 4


def sample_aloha(step):
    """Alternate pairs of updates to keep uniform and grasp draws in both datasets."""
    return (step - 1) // 2 % 2 == 1


def grasp_events(targets):
    """Return the first left closure and first left arm move after it closes."""
    gripper = targets[:, 6]
    starts = np.flatnonzero(np.diff(gripper) < -1e-4) + 1
    if not starts.size:
        raise ValueError("Expert episode has no left gripper closure")
    close = int(starts[0])
    closed = np.flatnonzero(gripper[close:] <= 1e-4)
    if not closed.size:
        raise ValueError("Expert left gripper does not complete closure")
    closed = close + int(closed[0])
    moving = np.flatnonzero(np.max(np.abs(np.diff(targets[closed:, :6], axis=0)), axis=1) > 1e-4)
    if not moving.size:
        raise ValueError("Expert episode has no left arm movement after closure")
    return close, closed + 1 + int(moving[0])


class GraspSampler:
    """Keep half the draws uniform and cycle through the three grasp phases."""

    def __init__(self, dataset, split="train"):
        self.windows = tuple(dataset.indices(split))
        self.phases = {name: [] for name in ("approach", "close", "post_close")}
        events = {ep: grasp_events(group["drive_target"][:]) for ep, group in dataset.groups[split].items()}
        for ep, t in self.windows:
            close, move = events[ep]
            if close - 2 * EXECUTED_ROWS <= t < close - EXECUTED_ROWS:
                self.phases["approach"].append((ep, t))
            if t < close <= t + EXECUTED_ROWS:
                self.phases["close"].append((ep, t))
            if t < move <= t + EXECUTED_ROWS:
                self.phases["post_close"].append((ep, t))
        if any(len(pool) != len(events) * EXECUTED_ROWS for pool in self.phases.values()):
            raise ValueError(f"Every {split} episode must cover all three grasp windows")

    @staticmethod
    def draw(windows, seed, step):
        index = np.random.default_rng(np.random.SeedSequence([seed, step])).integers(len(windows))
        return windows[index]

    def uniform(self, seed, step):
        return self.draw(self.windows, seed, step)

    def window(self, seed, step):
        if step % 2:
            return self.uniform(seed, step)
        pools = tuple(self.phases.values())
        return self.draw(pools[(step // 2 - 1) % len(pools)], seed, step)

    def batch(self, seed, step, size, terminal_windows):
        windows = terminal_windows
        if not step % 2:
            pools = tuple(self.phases.values())
            windows = pools[(step // 2 - 1) % len(pools)]
        rng = np.random.default_rng(np.random.SeedSequence([seed, step]))
        return [windows[i] for i in rng.choice(len(windows), size=size, replace=False)]
