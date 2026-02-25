from __future__ import annotations

from dataclasses import dataclass


STATE_LOW = "LOW"
STATE_MEDIUM = "MEDIUM"
STATE_HIGH = "HIGH"

ACTION_NO_CHANGE = "NO_CHANGE"
ACTION_REDUCE_CLUTTER = "REDUCE_CLUTTER"
ACTION_SLOW_ANIMATIONS = "SLOW_ANIMATIONS"
ACTION_SUGGEST_BREAK = "SUGGEST_BREAK"


@dataclass
class AdaptationEngine:
    cooldown_sec: float = 10.0
    high_persist_sec: float = 60.0
    current_state: str = STATE_MEDIUM
    last_action: str = ACTION_REDUCE_CLUTTER
    last_action_time: float | None = None
    high_start_time: float | None = None

    def update(self, timestamp_sec: float, score: float) -> dict:
        next_state = self._next_state(score)
        if next_state != self.current_state:
            if next_state == STATE_HIGH:
                self.high_start_time = timestamp_sec
            else:
                self.high_start_time = None
            self.current_state = next_state

        if self.current_state == STATE_HIGH:
            if self.high_start_time is None:
                self.high_start_time = timestamp_sec
            high_duration = timestamp_sec - self.high_start_time
        else:
            high_duration = 0.0

        desired_action = self._state_action(self.current_state, high_duration)
        action, action_changed = self._apply_cooldown(
            timestamp_sec, desired_action
        )

        return {
            "timestamp_sec": float(timestamp_sec),
            "score": float(score),
            "state": self.current_state,
            "action": action,
            "action_changed": action_changed,
        }

    def _next_state(self, score: float) -> str:
        if self.current_state == STATE_HIGH:
            if score < 0.60:
                if score < 0.40:
                    return STATE_LOW
                return STATE_MEDIUM
            return STATE_HIGH

        if self.current_state == STATE_LOW:
            if score > 0.50:
                if score > 0.70:
                    return STATE_HIGH
                return STATE_MEDIUM
            return STATE_LOW

        if score > 0.70:
            return STATE_HIGH
        if score < 0.40:
            return STATE_LOW
        return STATE_MEDIUM

    def _state_action(self, state: str, high_duration: float) -> str:
        if state == STATE_HIGH and high_duration >= self.high_persist_sec:
            return ACTION_SUGGEST_BREAK
        if state == STATE_HIGH:
            return ACTION_SLOW_ANIMATIONS
        if state == STATE_MEDIUM:
            return ACTION_REDUCE_CLUTTER
        return ACTION_NO_CHANGE

    def _apply_cooldown(
        self, timestamp_sec: float, desired_action: str
    ) -> tuple[str, bool]:
        if self.last_action_time is None:
            self.last_action_time = timestamp_sec
            self.last_action = desired_action
            return desired_action, True

        if desired_action == self.last_action:
            return self.last_action, False

        if (timestamp_sec - self.last_action_time) < self.cooldown_sec:
            return self.last_action, False

        self.last_action_time = timestamp_sec
        self.last_action = desired_action
        return desired_action, True
