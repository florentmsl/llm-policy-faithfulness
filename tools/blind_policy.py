"""Generate game-blinded versions of SCoBots policies for the prior-knowledge ablation.

Renames game-specific object names (Ball, Player, Enemy, Chicken, Car) to neutral
identifiers (Obj_A/Agent/Obj_B/Hazard_*) and strips action-name comments. SCoBots
DSL operators (D, ED, LT, C, V, DV) are preserved — they are game-agnostic feature
ops, not game cues.

Run: `python tools/blind_policy.py`
"""

import re
from pathlib import Path

PONG_RENAMES = [
    (r"\bBall1\b", "Obj_A"),
    (r"\bPlayer1\b", "Agent"),
    (r"\bEnemy1\b", "Obj_B"),
]

FREEWAY_RENAMES = [
    (r"\bChicken1\b", "Agent"),
    (r"\bChicken2\b", "Other_Agent"),
] + [
    # Order matters: Car10 before Car1 to avoid Car1 matching Car10's prefix.
    (rf"\bCar{i}\b", f"Hazard_{i}")
    for i in range(10, 0, -1)
]

JOBS = [
    {
        "src": "01_policies/scobots/pong/aligned.py",
        "dst": "01_policies/scobots/_blinded/pong/aligned_blinded.py",
        "renames": PONG_RENAMES,
    },
    {
        "src": "01_policies/scobots/pong/ignore_ball.py",
        "dst": "01_policies/scobots/_blinded/pong/ignore_ball_blinded.py",
        "renames": PONG_RENAMES,
    },
    {
        "src": "01_policies/scobots/freeway/aligned.py",
        "dst": "01_policies/scobots/_blinded/freeway/aligned_blinded.py",
        "renames": FREEWAY_RENAMES,
    },
    {
        "src": "01_policies/scobots/freeway/stay_bottom.py",
        "dst": "01_policies/scobots/_blinded/freeway/stay_bottom_blinded.py",
        "renames": FREEWAY_RENAMES,
    },
]


def _strip_action_comments(text: str) -> str:
    # Strip trailing inline comments only on `return N  # NAME` lines, since the
    # action names (NOOP/UP/DOWN/FIRE/LEFT/RIGHT/LEFTFIRE/RIGHTFIRE) are game cues.
    return re.sub(r"(return\s+\d+)\s*#.*", r"\1", text)


def _apply_renames(text: str, renames: list[tuple[str, str]]) -> str:
    for pattern, replacement in renames:
        text = re.sub(pattern, replacement, text)
    return text


def main() -> None:
    for job in JOBS:
        src_text = Path(job["src"]).read_text(encoding="utf-8")
        out = _strip_action_comments(_apply_renames(src_text, job["renames"]))
        Path(job["dst"]).write_text(out, encoding="utf-8")
        print(f"wrote {job['dst']}")


if __name__ == "__main__":
    main()
