"""
MDP Battle Simulator v5.0 — See the AI Think

A Markov Decision Process-based turn-based battle simulator with a PySide6 GUI.
Demonstrates multiple design patterns: Dataclass, Enum, NamedTuple, Strategy,
Observer, Factory, Builder, State Machine, ABC/Protocol, and more.

Author: Enhanced with programming patterns
"""

from __future__ import annotations

__all__ = [
    "Action", "GamePhase", "BattleState", "RewardWeights", "UnitStats",
    "MDPConfig", "RewardStrategy", "DefaultRewardStrategy", "BossFactory",
    "MDPEngine", "StateObserver", "BattleContext",
    "MDPBattleGame", "main",
]

import sys
import random
import time
import json
import logging
import abc
import enum
import functools
from dataclasses import dataclass, field, asdict, replace
from typing import (
    NamedTuple, Optional, Dict, List, Tuple, Callable, Protocol, Iterator,
    Any, Sequence, TypeVar, Generic, Set, Final
)
from contextlib import contextmanager

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QSpinBox, QDoubleSpinBox, QGroupBox,
    QGridLayout, QTextEdit, QProgressBar, QFrame, QMessageBox,
    QTabWidget, QComboBox, QFileDialog, QSplitter,
)
from PySide6.QtCore import Qt, QTimer, QThread, Signal, QPropertyAnimation, QEasingCurve, QPoint
from PySide6.QtGui import QFont, QColor, QPalette


# =============================================================================
# 1. LOGGING SETUP (Pattern: Singleton Logger via module-level)
# =============================================================================
logger = logging.getLogger("mdp_battle")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[%(name)s] %(levelname)s: %(message)s"))
    logger.addHandler(_handler)


# =============================================================================
# 2. ENUM PATTERN — Replaces magic indices with self-documenting members
# =============================================================================
class Action(enum.IntEnum):
    """All possible actions in the battle system.

    Using IntEnum allows both human-readable names and integer indexing
    (e.g., ``Action.SLASH == 0`` is True).
    """
    SLASH = 0
    CRUSH = 1
    PARRY = 2
    DODGE = 3
    MEDITATE = 4
    POTION = 5
    IGNITE = 6
    POISON = 7

    @classmethod
    def stamina_cost(cls, action: Action) -> int:
        """Return the stamina cost for a given action.

        Raises:
            ValueError: If the action is unknown.
        """
        costs: Dict[Action, int] = {
            cls.SLASH: 1, cls.CRUSH: 2, cls.DODGE: 1,
            cls.IGNITE: 1, cls.POTION: 0, cls.PARRY: 0,
            cls.MEDITATE: 0, cls.POISON: 1,
        }
        if action not in costs:
            raise ValueError(f"Unknown action: {action}")
        return costs[action]

    @classmethod
    def requires_potion(cls, action: Action) -> bool:
        """Check if the action requires a potion to be available."""
        return action == cls.POTION


class GamePhase(enum.Enum):
    """States the game can be in — used by the State Machine pattern."""
    CONFIGURATION = "configuration"
    SOLVING = "solving"
    BATTLE = "battle"
    RESULTS = "results"


# =============================================================================
# 3. NAMEDTUPLE PATTERN — Immutable, self-documenting data containers
# =============================================================================
class BattleState(NamedTuple):
    """Represents the full state tuple used in the MDP.

    Fields:
        m_hp: Player (my) hit points remaining.
        e_hp: Enemy hit points remaining.
        m_st: Player stamina remaining.
        m_pt: Player potions remaining.
        m_br: Player burn stacks (0–2).
        e_br: Enemy burn stacks (0–2).
        m_ps: Player poison stacks (0–2).
        e_ps: Enemy poison stacks (0–2).
    """
    m_hp: int
    e_hp: int
    m_st: int
    m_pt: int
    m_br: int
    e_br: int
    m_ps: int
    e_ps: int

    @property
    def is_terminal(self) -> bool:
        """Return True if the battle is over (one side has 0 HP)."""
        return self.m_hp <= 0 or self.e_hp <= 0

    @property
    def is_player_dead(self) -> bool:
        return self.m_hp <= 0

    @property
    def is_enemy_dead(self) -> bool:
        return self.e_hp <= 0


class RewardWeights(NamedTuple):
    """Weights used to compute the reward signal for the MDP.

    Each dimension captures a different aspect the optimizer should
    consider when choosing actions.
    """
    attack: float
    damage_taken: float
    healing: float
    burn: float
    poison: float


# =============================================================================
# 4. DATACLASS PATTERN — Replaces manual __init__ boilerplate
# =============================================================================
@dataclass
class UnitStats:
    """Immutable (frozen) combat statistics for a battle unit.

    Attributes:
        max_hp: Maximum hit points.
        max_stam: Maximum stamina per battle.
        max_pots: Maximum potions available.
        slash_acc: Probability of a successful SLASH (0–1).
        crush_acc: Probability of a successful CRUSH (0–1).
        parry_rate: Probability of a successful PARRY (0–1).
        ignite_acc: Probability of a successful IGNITE (0–1).
        poison_acc: Probability of a successful POISON (0–1).
        dodge_rate: Probability of a successful DODGE (0–1).
        crit_chance: Probability of a critical hit (0–1).
    """
    max_hp: int = 4
    max_stam: int = 3
    max_pots: int = 2
    slash_acc: float = 0.85
    crush_acc: float = 0.65
    parry_rate: float = 0.75
    ignite_acc: float = 0.70
    poison_acc: float = 0.65
    dodge_rate: float = 0.70
    crit_chance: float = 0.15

    def __post_init__(self) -> None:
        """Validate that all probability values are in [0, 1] and stats are non-negative."""
        for name, value in asdict(self).items():
            if name.endswith("_acc") or name.endswith("_rate") or name.endswith("_chance"):
                if not 0.0 <= value <= 1.0:
                    raise ValueError(
                        f"UnitStats.{name}={value} is out of range [0.0, 1.0]"
                    )
            elif isinstance(value, (int, float)):
                if value < 0:
                    raise ValueError(f"UnitStats.{name}={value} must be non-negative")

    def get_probs(self) -> Tuple[float, ...]:
        """Return the combat probability tuple (s_acc, c_acc, p_rate, i_acc, ps_acc, d_rate, crit)."""
        return (
            self.slash_acc, self.crush_acc, self.parry_rate,
            self.ignite_acc, self.poison_acc, self.dodge_rate, self.crit_chance,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary (useful for JSON export)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> UnitStats:
        """Deserialize from a dictionary, ignoring unknown keys."""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


# =============================================================================
# 5. CONSTANTS (Pattern: Centralized Configuration)
# =============================================================================
@dataclass(frozen=True)
class MDPConfig:
    """Centralized configuration for the MDP solver.

    Using a frozen dataclass ensures these values cannot be accidentally
    mutated at runtime, and provides a single source of truth.
    """
    terminal_reward: float = 1000.0
    step_penalty: float = -0.5
    gamma_default: float = 0.9
    max_iterations: int = 2000
    convergence_threshold: float = 1e-4
    max_battle_turns: int = 50
    max_burn_stacks: int = 2
    max_poison_stacks: int = 2


CONFIG: Final[MDPConfig] = MDPConfig()


# =============================================================================
# 6. STRATEGY PATTERN — Pluggable reward computation
# =============================================================================
class RewardStrategy(Protocol):
    """Protocol (structural typing) for reward computation strategies.

    Any class that implements ``compute(state, action, next_state) -> float``
    satisfies this protocol — no inheritance required.
    """

    def compute(self, state: BattleState, action: Action, next_state: BattleState) -> float:
        ...


class DefaultRewardStrategy:
    """Standard reward function that rewards aggression and penalizes damage taken.

    Attributes:
        weights: A RewardWeights tuple controlling the relative importance
                 of each reward dimension.
    """

    def __init__(self, weights: RewardWeights) -> None:
        self.weights = weights

    def compute(self, state: BattleState, action: Action, next_state: BattleState) -> float:
        """Compute the scalar reward for a single transition.

        The reward combines:
        - A per-step time penalty to encourage quick victories.
        - A large terminal reward/penalty for winning/losing.
        - Weighted differences in HP, burn stacks, and poison stacks.

        Args:
            state: The state before the action.
            action: The action taken.
            next_state: The resulting state after the action.

        Returns:
            A float reward value (positive = good for the agent).
        """
        atk_w, dmg_w, heal_w, burn_w, poison_w = self.weights
        r = CONFIG.step_penalty

        # Terminal conditions
        if next_state.e_hp == 0:
            r += CONFIG.terminal_reward
        if next_state.m_hp == 0:
            r -= CONFIG.terminal_reward

        # Differential rewards
        r += (state.e_hp - next_state.e_hp) * atk_w      # damage dealt to enemy
        r += (state.m_hp - next_state.m_hp) * dmg_w      # damage taken by player
        r += (next_state.m_hp - state.m_hp) * heal_w     # healing received
        r += (next_state.e_br - state.e_br) * burn_w     # burn applied to enemy
        r += (next_state.m_br - state.m_br) * -burn_w    # burn received by player
        r += (next_state.e_ps - state.e_ps) * poison_w   # poison applied to enemy
        r += (next_state.m_ps - state.m_ps) * -poison_w  # poison received by player

        return r


# =============================================================================
# 7. MDP ENGINE (Pattern: Facade — unified interface to the MDP subsystem)
# =============================================================================
class MDPEngine:
    """Facade that encapsulates the full MDP pipeline: state space generation,
    transition dynamics, value iteration, and Q-value computation.

    This class owns all pure-computation logic and holds no UI state,
    making it independently testable.
    """

    def __init__(
        self,
        player_stats: UnitStats,
        enemy_stats: UnitStats,
        reward_strategy: RewardStrategy,
        config: MDPConfig = CONFIG,
    ) -> None:
        self.player_stats = player_stats
        self.enemy_stats = enemy_stats
        self.reward_strategy = reward_strategy
        self.config = config
        self._states: Optional[List[BattleState]] = None
        self._transition_cache: Dict[Tuple[BattleState, Action], List[Tuple[float, BattleState]]] = {}

    # ----- State Space -----

    def get_state_space(self) -> List[BattleState]:
        """Generate the full state space as a list of BattleState named tuples.

        Returns:
            A list of all possible (m_hp, e_hp, m_st, m_pt, m_br, e_br, m_ps, e_ps) tuples.
        """
        if self._states is not None:
            return self._states

        logger.info("Generating state space for player=%s, enemy=%s ...", self.player_stats, self.enemy_stats)
        ps, es = self.player_stats, self.enemy_stats
        self._states = [
            BattleState(m_hp, e_hp, m_st, m_pt, m_br, e_br, m_ps, e_ps)
            for m_hp in range(ps.max_hp + 1)
            for e_hp in range(es.max_hp + 1)
            for m_st in range(ps.max_stam + 1)
            for m_pt in range(ps.max_pots + 1)
            for m_br in range(self.config.max_burn_stacks + 1)
            for e_br in range(self.config.max_burn_stacks + 1)
            for m_ps in range(self.config.max_poison_stacks + 1)
            for e_ps in range(self.config.max_poison_stacks + 1)
        ]
        logger.info("State space size: %d", len(self._states))
        return self._states

    # ----- Legality -----

    @staticmethod
    def is_legal(state: BattleState, action: Action, max_stam: int, max_pots: int) -> bool:
        """Check whether an action is legal in the given state.

        Args:
            state: Current battle state.
            action: Proposed action.
            max_stam: Maximum stamina for the acting unit.
            max_pots: Maximum potions for the acting unit.

        Returns:
            True if the action can be performed.
        """
        try:
            if action in (Action.SLASH, Action.DODGE, Action.IGNITE, Action.POISON):
                return state.m_st >= Action.stamina_cost(action)
            if action == Action.CRUSH:
                return state.m_st >= Action.stamina_cost(action)
            if action == Action.POTION:
                return state.m_pt >= 1
        except ValueError:
            logger.warning("Unknown action in is_legal: %s", action)
        return True

    # ----- Transition Dynamics -----

    def apply_effects(self, state: BattleState) -> BattleState:
        """Apply burn and poison damage at the start of a turn.

        Burn deals 1 damage per stack and decrements by 1.
        Poison deals N damage (where N = stack count) and decrements by 1.

        Args:
            state: State before status effects.

        Returns:
            New state after status effects have been applied.
        """
        m_hp, e_hp, m_st, m_pt, m_br, e_br, m_ps, e_ps = state

        if m_br > 0:
            m_hp = max(0, m_hp - 1)
            m_br -= 1
        if e_br > 0:
            e_hp = max(0, e_hp - 1)
            e_br -= 1
        if m_ps > 0:
            m_hp = max(0, m_hp - m_ps)
            m_ps = max(0, m_ps - 1)
        if e_ps > 0:
            e_hp = max(0, e_hp - e_ps)
            e_ps = max(0, e_ps - 1)

        return BattleState(m_hp, e_hp, m_st, m_pt, m_br, e_br, m_ps, e_ps)

    def get_transitions(self, state: BattleState, action: Action, stats: UnitStats) -> List[Tuple[float, BattleState]]:
        """Compute the probabilistic transitions for a (state, action) pair.

        Each transition is a (probability, next_state) tuple.
        Terminal states and illegal actions produce a self-loop with probability 1.

        Args:
            state: Current battle state.
            action: Action to take.
            stats: Combat stats of the acting unit.

        Returns:
            List of (probability, next_state) pairs.
        """
        cache_key = (state, action)
        if cache_key in self._transition_cache:
            return self._transition_cache[cache_key]

        if state.is_terminal:
            result = [(1.0, state)]
            self._transition_cache[cache_key] = result
            return result

        if not self.is_legal(state, action, stats.max_stam, stats.max_pots):
            result = [(1.0, state)]
            self._transition_cache[cache_key] = result
            return result

        raw_transitions = self._compute_raw_transitions(state, action, stats)
        result = [(p, self.apply_effects(s)) for p, s in raw_transitions]
        self._transition_cache[cache_key] = result
        return result

    def _compute_raw_transitions(
        self, state: BattleState, action: Action, stats: UnitStats
    ) -> List[Tuple[float, BattleState]]:
        """Compute raw (before status effects) transition outcomes.

        This is the core combat logic. Each action has either a deterministic
        or probabilistic outcome based on the unit's accuracy stats.
        """
        m_hp, e_hp, m_st, m_pt, m_br, e_br, m_ps, e_ps = state
        s_acc, c_acc, p_rate, i_acc, ps_acc, d_rate, crit = stats.get_probs()

        def _new(m_hp_=None, e_hp_=None, m_st_=None, m_pt_=None,
                 m_br_=None, e_br_=None, m_ps_=None, e_ps_=None) -> BattleState:
            return BattleState(
                m_hp if m_hp_ is None else m_hp_,
                e_hp if e_hp_ is None else e_hp_,
                m_st if m_st_ is None else m_st_,
                m_pt if m_pt_ is None else m_pt_,
                m_br if m_br_ is None else m_br_,
                e_br if e_br_ is None else e_br_,
                m_ps if m_ps_ is None else m_ps_,
                e_ps if e_ps_ is None else e_ps_,
            )

        if action == Action.SLASH:
            ns = _new(e_hp_=max(0, e_hp - 1), m_st_=m_st - 1)
            ms = _new(m_hp_=max(0, m_hp - 1), m_st_=m_st - 1)
            return [(s_acc, ns), (1 - s_acc, ms)]

        elif action == Action.CRUSH:
            ns = _new(e_hp_=max(0, e_hp - 2), m_st_=m_st - 2)
            ms = _new(m_hp_=max(0, m_hp - 1), m_st_=m_st - 2)
            return [(c_acc, ns), (1 - c_acc, ms)]

        elif action == Action.PARRY:
            ns = _new(m_st_=min(stats.max_stam, m_st + 1))
            ms = _new(m_hp_=max(0, m_hp - 1))
            return [(p_rate, ns), (1 - p_rate, ms)]

        elif action == Action.DODGE:
            ns = _new(m_st_=m_st - 1)
            ms = _new(m_hp_=max(0, m_hp - 1), m_st_=m_st - 1)
            return [(d_rate, ns), (1 - d_rate, ms)]

        elif action == Action.MEDITATE:
            ns = _new(
                m_hp_=min(stats.max_hp, m_hp + 1),
                m_st_=min(stats.max_stam, m_st + 2),
            )
            return [(1.0, ns)]

        elif action == Action.POTION:
            ns = _new(
                m_hp_=min(stats.max_hp, m_hp + 2),
                m_pt_=m_pt - 1,
                m_br_=0, m_ps_=0,
            )
            return [(1.0, ns)]

        elif action == Action.IGNITE:
            ns = _new(m_st_=m_st - 1, e_br_=self.config.max_burn_stacks)
            ms = _new(m_st_=m_st - 1)
            return [(i_acc, ns), (1 - i_acc, ms)]

        elif action == Action.POISON:
            ns = _new(m_st_=m_st - 1, e_ps_=min(self.config.max_poison_stacks, e_ps + 1))
            ms = _new(m_st_=m_st - 1)
            return [(ps_acc, ns), (1 - ps_acc, ms)]

        logger.warning("Unhandled action in transitions: %s — returning self-loop", action)
        return [(1.0, state)]

    # ----- Value Iteration -----

    def value_iteration(
        self,
        gamma: float = CONFIG.gamma_default,
        theta: float = CONFIG.convergence_threshold,
        progress_callback: Optional[Callable[[int, float], None]] = None,
    ) -> Tuple[Dict[BattleState, int], Dict[BattleState, float], int]:
        """Solve the MDP using synchronous value iteration.

        Args:
            gamma: Discount factor (0 < gamma < 1).
            theta: Convergence threshold — stops when max |V_new - V_old| < theta.
            progress_callback: Optional callable(iteration, delta) for UI feedback.

        Returns:
            A tuple of (policy, value_function, iterations_completed).
            - policy: Maps each state to the optimal action index.
            - value_function: Maps each state to its expected return.
        """
        states = self.get_state_space()
        V: Dict[BattleState, float] = {s: 0.0 for s in states}
        policy: Dict[BattleState, int] = {s: 0 for s in states}
        num_actions = len(Action)

        # Pre-compute all transition-reward pairs (caching avoids redundant work)
        logger.info("Pre-computing transition-reward cache ...")
        T_R_cache: Dict[Tuple[BattleState, int], List[Tuple[float, BattleState, float]]] = {}
        for s in states:
            for a_idx in range(num_actions):
                action = Action(a_idx)
                trans = self.get_transitions(s, action, self.player_stats)
                rewards = [
                    (p, s_prime, self.reward_strategy.compute(s, action, s_prime))
                    for p, s_prime in trans
                ]
                T_R_cache[(s, a_idx)] = rewards

        logger.info("Running value iteration (theta=%.4f, gamma=%.2f) ...", theta, gamma)
        iterations = 0
        while True:
            delta = 0.0
            for s in states:
                if s.is_terminal:
                    continue

                old_v = V[s]
                action_values: List[float] = []

                for a_idx in range(num_actions):
                    if not self.is_legal(s, Action(a_idx), self.player_stats.max_stam, self.player_stats.max_pots):
                        action_values.append(-float("inf"))
                    else:
                        val = sum(
                            p * (r + gamma * V[s_prime])
                            for p, s_prime, r in T_R_cache[(s, a_idx)]
                        )
                        action_values.append(val)

                best_val = max(action_values)
                V[s] = best_val
                policy[s] = action_values.index(best_val)
                delta = max(delta, abs(old_v - V[s]))

            iterations += 1
            if progress_callback and iterations % 10 == 0:
                progress_callback(iterations, delta)
            if delta < theta or iterations > self.config.max_iterations:
                break

        logger.info("Value iteration converged in %d iterations (delta=%.6f)", iterations, delta)
        return policy, V, iterations

    # ----- Q-Values -----

    def compute_q_values(
        self,
        state: BattleState,
        policy: Dict[BattleState, int],
        V: Dict[BattleState, float],
        gamma: float,
    ) -> List[float]:
        """Compute Q-values for all actions in a given state.

        Args:
            state: Current battle state.
            policy: Optimal policy (used for context; not needed for Q itself).
            V: Value function.
            gamma: Discount factor.

        Returns:
            A list of Q-values, one per action.
        """
        q_vals: List[float] = []
        for a_idx in range(len(Action)):
            action = Action(a_idx)
            if not self.is_legal(state, action, self.player_stats.max_stam, self.player_stats.max_pots):
                q_vals.append(-float("inf"))
            else:
                trans = self.get_transitions(state, action, self.player_stats)
                val = sum(
                    p * (self.reward_strategy.compute(state, action, s_prime) + gamma * V[s_prime])
                    for p, s_prime in trans
                )
                q_vals.append(val)
        return q_vals


# =============================================================================
# 8. OBSERVER PATTERN — Decoupled state change notifications
# =============================================================================
class StateObserver(Protocol):
    """Protocol for objects that wish to observe battle state changes.

    Implementors receive a callback whenever the battle state transitions.
    This decouples the battle logic from the UI update logic.
    """

    def on_state_changed(self, old_state: BattleState, new_state: BattleState, action_taken: str) -> None:
        """Called after a battle state transition.

        Args:
            old_state: State before the transition.
            new_state: State after the transition.
            action_taken: Human-readable description of the action.
        """
        ...


class ObservableState:
    """Subject in the Observer pattern — tracks a BattleState and notifies listeners."""

    def __init__(self, initial_state: BattleState) -> None:
        self._state = initial_state
        self._observers: List[StateObserver] = []

    @property
    def state(self) -> BattleState:
        return self._state

    @state.setter
    def state(self, value: BattleState) -> None:
        old = self._state
        self._state = value
        self._notify(old, value)

    def attach(self, observer: StateObserver) -> None:
        """Register an observer."""
        if observer not in self._observers:
            self._observers.append(observer)

    def detach(self, observer: StateObserver) -> None:
        """Unregister an observer."""
        self._observers.remove(observer)

    def _notify(self, old_state: BattleState, new_state: BattleState, action_taken: str = "") -> None:
        for obs in self._observers:
            try:
                obs.on_state_changed(old_state, new_state, action_taken)
            except Exception:
                logger.exception("Observer %s failed during notification", obs)


# =============================================================================
# 9. FACTORY PATTERN — Boss preset creation
# =============================================================================
class BossPreset(NamedTuple):
    """A named boss configuration.

    Attributes:
        name: Display name.
        stats: Combat statistics for the boss.
        weights: Reward weights for the enemy MDP.
        description: Human-readable flavor text.
    """
    name: str
    stats: UnitStats
    weights: RewardWeights
    description: str


class BossFactory:
    """Factory class for creating boss configurations from presets or custom values.

    Usage:
        >>> boss = BossFactory.create("KNIGHT")
        >>> boss.name
        'KNIGHT'
    """

    _PRESETS: Dict[str, BossPreset] = {
        "GOBLIN": BossPreset(
            name="GOBLIN",
            stats=UnitStats(3, 2, 1, 0.70, 0.50, 0.60, 0.60, 0.50, 0.50, 0.05),
            weights=RewardWeights(10, -10, 5, 5, 3),
            description="Weak but scrappy. Low stats, aggressive.",
        ),
        "KNIGHT": BossPreset(
            name="KNIGHT",
            stats=UnitStats(4, 3, 2, 0.80, 0.70, 0.90, 0.50, 0.40, 0.70, 0.10),
            weights=RewardWeights(15, -20, 15, 2, 5),
            description="Armored defender. High parry, punishes mistakes.",
        ),
        "DRAGON": BossPreset(
            name="DRAGON",
            stats=UnitStats(5, 4, 3, 0.90, 0.80, 0.70, 0.90, 0.80, 0.60, 0.20),
            weights=RewardWeights(20, -15, 10, 15, 12),
            description="The final boss. High stats, loves status effects.",
        ),
        "VAMPIRE": BossPreset(
            name="VAMPIRE",
            stats=UnitStats(4, 3, 2, 0.85, 0.75, 0.60, 0.70, 0.80, 0.65, 0.25),
            weights=RewardWeights(25, -5, 20, 8, 10),
            description="Sustained fighter. Doesn't fear damage.",
        ),
        "BERSERKER": BossPreset(
            name="BERSERKER",
            stats=UnitStats(3, 4, 1, 0.95, 0.90, 0.30, 0.40, 0.30, 0.40, 0.35),
            weights=RewardWeights(30, -5, 5, 3, 2),
            description="Glass cannon. Massive damage, ignores defense.",
        ),
    }

    @classmethod
    def available_bosses(cls) -> Dict[str, BossPreset]:
        """Return a copy of all available boss presets."""
        return dict(cls._PRESETS)

    @classmethod
    def create(cls, name: str) -> BossPreset:
        """Create a boss preset by name.

        Args:
            name: Case-insensitive boss name.

        Returns:
            The matching BossPreset.

        Raises:
            KeyError: If the boss name is not found.
        """
        key = name.upper()
        if key not in cls._PRESETS:
            raise KeyError(f"Unknown boss: {name!r}. Available: {list(cls._PRESETS.keys())}")
        return cls._PRESETS[key]

    @classmethod
    def create_custom(
        cls,
        stats: UnitStats,
        attack_weight: float = 15.0,
        damage_weight: float = -15.0,
    ) -> BossPreset:
        """Create a custom boss from user-provided stats.

        Args:
            stats: The boss's combat statistics.
            attack_weight: How much the enemy values dealing damage.
            damage_weight: How much the enemy avoids taking damage.

        Returns:
            A new BossPreset with the custom configuration.
        """
        return BossPreset(
            name="CUSTOM",
            stats=stats,
            weights=RewardWeights(attack_weight, damage_weight, 10, 10, 8),
            description="Custom boss — designed by you!",
        )


# =============================================================================
# 10. BUILDER PATTERN — Incremental construction of game configuration
# =============================================================================
class GameConfigBuilder:
    """Builder for assembling the game configuration from UI inputs.

    Provides a fluent API for step-by-step construction and validates
    the final result before producing the output.
    """

    def __init__(self) -> None:
        self._player_stats: Optional[UnitStats] = None
        self._enemy_stats: Optional[UnitStats] = None
        self._player_weights: Optional[RewardWeights] = None
        self._enemy_weights: Optional[RewardWeights] = None
        self._gamma: float = CONFIG.gamma_default

    def set_player_stats(self, stats: UnitStats) -> GameConfigBuilder:
        """Set player combat statistics. Returns self for chaining."""
        self._player_stats = stats
        return self

    def set_enemy_stats(self, stats: UnitStats) -> GameConfigBuilder:
        """Set enemy combat statistics. Returns self for chaining."""
        self._enemy_stats = stats
        return self

    def set_player_weights(self, weights: RewardWeights) -> GameConfigBuilder:
        """Set player reward weights. Returns self for chaining."""
        self._player_weights = weights
        return self

    def set_enemy_weights(self, weights: RewardWeights) -> GameConfigBuilder:
        """Set enemy reward weights. Returns self for chaining."""
        self._enemy_weights = weights
        return self

    def set_gamma(self, gamma: float) -> GameConfigBuilder:
        """Set the discount factor. Returns self for chaining."""
        self._gamma = gamma
        return self

    def build(self) -> Tuple[UnitStats, UnitStats, RewardWeights, RewardWeights, float]:
        """Validate and return the complete configuration.

        Returns:
            Tuple of (player_stats, enemy_stats, player_weights, enemy_weights, gamma).

        Raises:
            ValueError: If any required field is missing or invalid.
        """
        errors: List[str] = []
        if self._player_stats is None:
            errors.append("Player stats not set")
        if self._enemy_stats is None:
            errors.append("Enemy stats not set")
        if self._player_weights is None:
            errors.append("Player weights not set")
        if self._enemy_weights is None:
            errors.append("Enemy weights not set")
        if not 0.0 < self._gamma < 1.0:
            errors.append(f"Gamma must be in (0, 1), got {self._gamma}")

        if errors:
            raise ValueError("Invalid game config: " + "; ".join(errors))

        return (
            self._player_stats,  # type: ignore[arg-type]
            self._enemy_stats,   # type: ignore[arg-type]
            self._player_weights,  # type: ignore[arg-type]
            self._enemy_weights,   # type: ignore[arg-type]
            self._gamma,
        )


# =============================================================================
# 11. CONTEXT MANAGER PATTERN — Battle state snapshot/restore
# =============================================================================
@contextmanager
def battle_snapshot(observable: ObservableState) -> Iterator[None]:
    """Context manager that saves and restores the battle state on exception.

    Usage:
        >>> with battle_snapshot(observable_state):
        ...     risky_operation()
        ... # If risky_operation() raises, state is rolled back.
    """
    original = observable.state
    try:
        yield
    except Exception:
        logger.warning("Battle operation failed — rolling back state")
        observable.state = original  # type: ignore[assignment]
        raise


# =============================================================================
# 12. CUSTOM EXCEPTIONS (Pattern: Domain-Specific Errors)
# =============================================================================
class BattleError(Exception):
    """Base exception for battle-related errors."""
    pass


class InvalidStateError(BattleError):
    """Raised when an operation is attempted on an invalid battle state."""
    pass


class SolverError(BattleError):
    """Raised when the MDP solver fails to converge or encounters an error."""
    pass


# =============================================================================
# 13. THREADS — Background workers for MDP solving and Monte Carlo
# =============================================================================
class MDPSolverThread(QThread):
    """Runs value iteration in a background thread and emits results via signals.

    Signals:
        finished(policy, V, iterations): Emitted when solving completes.
        progress(iteration, delta): Emitted periodically during solving.
        error(message): Emitted if an exception occurs.
    """
    finished = Signal(dict, dict, int)
    progress = Signal(int, float)
    error = Signal(str)

    def __init__(
        self,
        engine: MDPEngine,
        gamma: float = CONFIG.gamma_default,
    ) -> None:
        super().__init__()
        self.engine = engine
        self.gamma = gamma

    def run(self) -> None:
        """Execute value iteration and emit results."""
        try:
            policy, V, iters = self.engine.value_iteration(
                gamma=self.gamma,
                progress_callback=lambda i, d: self.progress.emit(i, d),
            )
            # Convert BattleState keys to plain tuples for Qt signal serialization
            policy_out = {tuple(s): a for s, a in policy.items()}
            v_out = {tuple(s): v for s, v in V.items()}
            self.finished.emit(policy_out, v_out, iters)
        except Exception as e:
            logger.exception("MDP solver failed")
            self.error.emit(str(e))


class MonteCarloThread(QThread):
    """Runs N simulated battles between two MDP policies in a background thread.

    Signals:
        progress(done, wins, losses, draws): Emitted every 50 battles.
        finished(wins, losses, draws): Emitted when all battles complete.
        error(message): Emitted if an exception occurs.
    """
    progress = Signal(int, int, int, int)
    finished = Signal(int, int, int)
    error = Signal(str)

    def __init__(
        self,
        p_policy: Dict[Any, int],
        p_stats: UnitStats,
        e_policy: Dict[Any, int],
        e_stats: UnitStats,
        n_battles: int,
    ) -> None:
        super().__init__()
        self.p_policy = p_policy
        self.p_stats = p_stats
        self.e_policy = e_policy
        self.e_stats = e_stats
        self.n_battles = n_battles

    def run(self) -> None:
        """Simulate N battles and emit aggregate results."""
        try:
            wins = losses = draws = 0
            for i in range(self.n_battles):
                result = self._run_one_battle()
                if result == "WIN":
                    wins += 1
                elif result == "LOSS":
                    losses += 1
                else:
                    draws += 1
                if i % 50 == 0:
                    self.progress.emit(i, wins, losses, draws)
            self.finished.emit(wins, losses, draws)
        except Exception as e:
            logger.exception("Monte Carlo simulation failed")
            self.error.emit(str(e))

    def _run_one_battle(self) -> str:
        """Simulate a single battle and return 'WIN', 'LOSS', or 'DRAW'."""
        m_hp, e_hp = self.p_stats.max_hp, self.e_stats.max_hp
        m_st, e_st = self.p_stats.max_stam, self.e_stats.max_stam
        m_pt, e_pt = self.p_stats.max_pots, self.e_stats.max_pots
        m_br = e_br = m_ps = e_ps = 0

        for _ in range(CONFIG.max_battle_turns):
            if m_hp <= 0 or e_hp <= 0:
                break

            # Player turn
            p_state = (m_hp, e_hp, m_st, m_pt, m_br, e_br, m_ps, e_ps)
            p_act = self.p_policy.get(p_state, 0)
            p_trans = get_transitions_compat(p_state, p_act, self.p_stats)
            probs, next_states = zip(*p_trans)
            m_hp, e_hp, m_st, m_pt, m_br, e_br, m_ps, e_ps = \
                random.choices(next_states, weights=probs, k=1)[0]

            if m_hp <= 0 or e_hp <= 0:
                break

            # Enemy turn
            e_state = (e_hp, m_hp, e_st, e_pt, e_br, m_br, e_ps, m_ps)
            e_act = self.e_policy.get(e_state, 0)
            e_trans = get_transitions_compat(e_state, e_act, self.e_stats)
            e_probs, e_next_states = zip(*e_trans)
            e_hp, m_hp, e_st, e_pt, e_br, m_br, e_ps, m_ps = \
                random.choices(e_next_states, weights=e_probs, k=1)[0]

        if m_hp > 0 and e_hp <= 0:
            return "WIN"
        if e_hp > 0 and m_hp <= 0:
            return "LOSS"
        return "DRAW"


# Compatibility function for Monte Carlo (works with plain tuples)
def get_transitions_compat(
    state: tuple, action_idx: int, stats: UnitStats
) -> List[Tuple[float, tuple]]:
    """Compute transitions using plain tuples (for backward-compatible policy dict keys).

    This bridges the gap between the old tuple-based policy system and the
    new BattleState-based engine, ensuring Monte Carlo simulations work
    with policies stored under plain tuple keys.
    """
    bs = BattleState(*state)
    action = Action(action_idx)
    engine = MDPEngine(stats, stats, DefaultRewardStrategy(RewardWeights(0, 0, 0, 0, 0)))
    transitions = engine.get_transitions(bs, action, stats)
    return [(p, tuple(s)) for p, s in transitions]


# =============================================================================
# 14. CUSTOM WIDGETS
# =============================================================================
class HealthBar(QProgressBar):
    """Animated health bar with customizable color and smooth transitions."""

    def __init__(self, color: str = "#55ff55") -> None:
        super().__init__()
        self.color = color
        self._update_style()
        self.setFixedHeight(28)
        self._anim: Optional[QPropertyAnimation] = None

    def _update_style(self) -> None:
        self.setStyleSheet(f"""
            QProgressBar {{
                border: 2px solid #555; border-radius: 5px;
                text-align: center; background-color: #2a2a2a;
                color: white; font-weight: bold;
            }}
            QProgressBar::chunk {{
                background-color: {self.color}; border-radius: 3px;
            }}
        """)

    def animate_value(self, new_value: int) -> None:
        """Smoothly animate the bar to a new value using QPropertyAnimation."""
        if self._anim is not None:
            self._anim.stop()
        self._anim = QPropertyAnimation(self, b"value")
        self._anim.setDuration(400)
        self._anim.setStartValue(self.value())
        self._anim.setEndValue(new_value)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        self._anim.start()


class QValuePanel(QWidget):
    """Displays Q-values for each action as animated horizontal bars.

    The best action is highlighted in green, illegal actions are grayed out,
    and a state value label shows V(s).
    """

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        title = QLabel("AI THOUGHT PROCESS (Q-Values)")
        title.setStyleSheet("font-weight: bold; color: #00ffff; font-size: 13px;")
        layout.addWidget(title)

        self.bars: Dict[str, QProgressBar] = {}
        self.value_labels: Dict[str, QLabel] = {}
        for action_name in [a.name for a in Action]:
            row = QHBoxLayout()
            name = QLabel(f"{action_name:9}")
            name.setFixedWidth(80)
            name.setStyleSheet("color: #e0e0e0; font-family: monospace;")
            row.addWidget(name)

            bar = QProgressBar()
            bar.setRange(-100, 100)
            bar.setTextVisible(False)
            bar.setFixedHeight(18)
            bar.setStyleSheet("""
                QProgressBar { border: 1px solid #444; background: #1a1a1a; border-radius: 3px; }
                QProgressBar::chunk { background: #0088ff; border-radius: 2px; }
            """)
            row.addWidget(bar)

            val_label = QLabel("0.00")
            val_label.setFixedWidth(60)
            val_label.setStyleSheet("color: #aaa; font-family: monospace; font-size: 11px;")
            row.addWidget(val_label)

            self.bars[action_name] = bar
            self.value_labels[action_name] = val_label
            layout.addLayout(row)

        self.state_value_label = QLabel("State Value V(s): 0.00")
        self.state_value_label.setStyleSheet("color: #ffff00; font-weight: bold; margin-top: 5px;")
        layout.addWidget(self.state_value_label)

    def update_values(self, q_values: List[float], state_value: float, best_action: int) -> None:
        """Update all Q-value bars and highlight the best action.

        Args:
            q_values: List of Q-values, one per action.
            state_value: The value of the current state V(s).
            best_action: Index of the action with the highest Q-value.
        """
        valid_q = [q for q in q_values if q != -float("inf")]
        if not valid_q:
            return

        max_q = max(valid_q)
        min_q = min(valid_q) if min(valid_q) != max_q else max_q - 1
        range_q = max(abs(max_q), abs(min_q), 1)

        action_names = [a.name for a in Action]
        for i, action_name in enumerate(action_names):
            q = q_values[i]
            bar = self.bars[action_name]
            label = self.value_labels[action_name]

            if q == -float("inf"):
                bar.setValue(0)
                label.setText("N/A")
                label.setStyleSheet("color: #555; font-family: monospace; font-size: 11px;")
                bar.setStyleSheet("""
                    QProgressBar { border: 1px solid #333; background: #1a1a1a; border-radius: 3px; }
                    QProgressBar::chunk { background: #333; }
                """)
            else:
                normalized = int((q / range_q) * 100)
                bar.setValue(normalized)
                label.setText(f"{q:+.2f}")

                if i == best_action:
                    label.setStyleSheet("color: #00ff00; font-family: monospace; font-size: 11px; font-weight: bold;")
                    bar.setStyleSheet("""
                        QProgressBar { border: 1px solid #00ff00; background: #1a1a1a; border-radius: 3px; }
                        QProgressBar::chunk { background: #00ff00; }
                    """)
                else:
                    label.setStyleSheet("color: #aaa; font-family: monospace; font-size: 11px;")
                    bar.setStyleSheet("""
                        QProgressBar { border: 1px solid #444; background: #1a1a1a; border-radius: 3px; }
                        QProgressBar::chunk { background: #0088ff; }
                    """)

        self.state_value_label.setText(f"State Value V(s): {state_value:+.2f}")


class CharacterWidget(QWidget):
    """Animated ASCII character that reacts to damage level and status effects."""

    def __init__(self, char_type: str = "player") -> None:
        super().__init__()
        self.char_type = char_type
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setFont(QFont("Courier", 14))
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.label)
        self._shake_anim: Optional[QPropertyAnimation] = None
        self.update_display(1.0, False, False)

    def update_display(self, hp_ratio: float, is_burning: bool, is_poisoned: bool) -> None:
        """Update the ASCII art and color based on HP ratio and status effects.

        Args:
            hp_ratio: Current HP / max HP (0.0 to 1.0+).
            is_burning: Whether the unit has burn stacks.
            is_poisoned: Whether the unit has poison stacks.
        """
        art, color = self._get_art_and_color(hp_ratio)
        status = ""
        if is_burning:
            status += "[BURN]"
        if is_poisoned:
            status += "[PSN]"

        self.label.setText(f'<pre style="color: {color};">{art}</pre>{status}')

    def _get_art_and_color(self, hp_ratio: float) -> Tuple[str, str]:
        """Return (ASCII art, color) based on character type and HP ratio."""
        if self.char_type == "player":
            if hp_ratio > 0.66:
                return "  O  \n /|\\ \n / \\", "#55ff55"
            elif hp_ratio > 0.33:
                return "  O  \n \\|/ \n / \\", "#ffff55"
            else:
                return "  x  \n \\|/ \n / \\", "#ff5555"
        else:
            if hp_ratio > 0.66:
                return "  @  \n /|\\ \n / \\", "#ff5555"
            elif hp_ratio > 0.33:
                return "  @  \n \\|/ \n / \\", "#ff8855"
            else:
                return "  x  \n \\|/ \n / \\", "#ff3333"

    def shake(self) -> None:
        """Play a shake animation to indicate the unit was hit."""
        if self._shake_anim is not None:
            self._shake_anim.stop()
        self._shake_anim = QPropertyAnimation(self, b"pos")
        self._shake_anim.setDuration(300)
        orig = self.pos()
        self._shake_anim.setKeyValueAt(0.0, orig)
        self._shake_anim.setKeyValueAt(0.2, orig + QPoint(10, 0))
        self._shake_anim.setKeyValueAt(0.4, orig + QPoint(-10, 0))
        self._shake_anim.setKeyValueAt(0.6, orig + QPoint(8, 0))
        self._shake_anim.setKeyValueAt(0.8, orig + QPoint(-5, 0))
        self._shake_anim.setKeyValueAt(1.0, orig)
        self._shake_anim.start()


class FloatingDamage(QWidget):
    """Floating damage/heal number that rises and fades out.

    Automatically deletes itself after the animation completes via
    the ``deleteLater`` slot connected to the animation's finished signal.
    """

    def __init__(self, parent: QWidget, text: str, color: str, start_pos: QPoint) -> None:
        super().__init__(parent)
        self.label = QLabel(text, self)
        self.label.setStyleSheet(f"color: {color}; font-size: 24px; font-weight: bold;")
        self.label.adjustSize()
        self.label.move(start_pos)
        self.show()

        self._anim = QPropertyAnimation(self.label, b"pos")
        self._anim.setDuration(1200)
        self._anim.setStartValue(start_pos)
        self._anim.setEndValue(start_pos + QPoint(0, -80))
        self._anim.setEasingCurve(QEasingCurve.OutQuad)
        self._anim.start()
        self._anim.finished.connect(self.deleteLater)


class StatusIcon(QLabel):
    """Pulsating status effect icon with stack counter."""

    def __init__(self, icon: str, color: str) -> None:
        super().__init__()
        self.icon = icon
        self.color = color
        self.setStyleSheet(f"color: {color}; font-size: 16px; font-weight: bold;")
        self._pulse: Optional[QPropertyAnimation] = None

    def set_count(self, count: int) -> None:
        """Update the displayed stack count and start/stop the pulse animation.

        Args:
            count: Number of active stacks (0 hides the icon).
        """
        if count > 0:
            self.setText(f"{self.icon} x{count}")
            if self._pulse is None:
                self._pulse = QPropertyAnimation(self, b"styleSheet")
                self._pulse.setDuration(800)
                self._pulse.setLoopCount(-1)
                self._pulse.setKeyValueAt(0.0, f"color: {self.color}; font-size: 16px; font-weight: bold;")
                self._pulse.setKeyValueAt(0.5, f"color: #ffffff; font-size: 18px; font-weight: bold;")
                self._pulse.setKeyValueAt(1.0, f"color: {self.color}; font-size: 16px; font-weight: bold;")
                self._pulse.start()
        else:
            if self._pulse is not None:
                self._pulse.stop()
                self._pulse = None
            self.setText("")


# =============================================================================
# 15. STATE MACHINE PATTERN — Game phase management
# =============================================================================
class GameStateMachine:
    """Manages game phase transitions using the State Machine pattern.

    Ensures that the game progresses through valid phases:
        CONFIGURATION -> SOLVING -> BATTLE -> RESULTS

    Transitions can only occur between adjacent phases, preventing
    illegal state jumps (e.g., jumping directly from CONFIGURATION to RESULTS).
    """

    _VALID_TRANSITIONS: Dict[GamePhase, Set[GamePhase]] = {
        GamePhase.CONFIGURATION: {GamePhase.SOLVING},
        GamePhase.SOLVING: {GamePhase.BATTLE, GamePhase.CONFIGURATION},
        GamePhase.BATTLE: {GamePhase.RESULTS, GamePhase.SOLVING},
        GamePhase.RESULTS: {GamePhase.CONFIGURATION},
    }

    def __init__(self) -> None:
        self._phase = GamePhase.CONFIGURATION
        self._history: List[GamePhase] = [GamePhase.CONFIGURATION]

    @property
    def phase(self) -> GamePhase:
        """Current game phase."""
        return self._phase

    def transition_to(self, new_phase: GamePhase) -> None:
        """Attempt to transition to a new game phase.

        Args:
            new_phase: The desired next phase.

        Raises:
            InvalidStateError: If the transition is not allowed.
        """
        allowed = self._VALID_TRANSITIONS.get(self._phase, set())
        if new_phase not in allowed:
            raise InvalidStateError(
                f"Cannot transition from {self._phase.value!r} to {new_phase.value!r}. "
                f"Allowed: {[p.value for p in allowed]}"
            )
        logger.info("Game phase: %s -> %s", self._phase.value, new_phase.value)
        self._phase = new_phase
        self._history.append(new_phase)

    def reset(self) -> None:
        """Reset the state machine back to CONFIGURATION."""
        self._phase = GamePhase.CONFIGURATION
        self._history = [GamePhase.CONFIGURATION]


# =============================================================================
# 16. BATTLE CONTEXT (Pattern: Context Object — carries all shared state)
# =============================================================================
@dataclass
class BattleContext:
    """Encapsulates all mutable battle state in a single object.

    This Context Object pattern replaces the scattered dict-based
    battle_state with a typed, validated container that can be passed
    around cleanly.

    Attributes:
        player_hp, enemy_hp: Current hit points.
        player_st, enemy_st: Current stamina.
        player_pt, enemy_pt: Current potions remaining.
        player_br, enemy_br: Current burn stacks.
        player_ps, enemy_ps: Current poison stacks.
        turn: Current turn number.
        enemy_name: Display name of the enemy.
        stats: Aggregate battle statistics (damage dealt, etc.).
    """
    player_hp: int = 4
    enemy_hp: int = 4
    player_st: int = 3
    enemy_st: int = 3
    player_pt: int = 2
    enemy_pt: int = 2
    player_br: int = 0
    enemy_br: int = 0
    player_ps: int = 0
    enemy_ps: int = 0
    turn: int = 0
    enemy_name: str = "ENEMY"
    stats: Dict[str, int] = field(default_factory=lambda: {
        "dmg_dealt": 0, "dmg_taken": 0, "crits": 0, "pots_used": 0,
    })

    def to_battle_state(self) -> BattleState:
        """Convert to an MDP BattleState (for policy lookups)."""
        return BattleState(
            self.player_hp, self.enemy_hp, self.player_st, self.player_pt,
            self.player_br, self.enemy_br, self.player_ps, self.enemy_ps,
        )

    def enemy_perspective_state(self) -> tuple:
        """Return the state from the enemy's perspective (swap roles)."""
        return (
            self.enemy_hp, self.player_hp, self.enemy_st, self.enemy_pt,
            self.enemy_br, self.player_br, self.enemy_ps, self.player_ps,
        )


# =============================================================================
# 17. MAIN WINDOW (Pattern: MVC — Model-View-Controller separation)
# =============================================================================
class MDPBattleGame(QMainWindow):
    """Main application window — serves as the View and Controller.

    The Model is the MDPEngine + BattleContext + GameStateMachine.
    This class handles only UI construction and event wiring.

    Improved from v4.0:
    - Type hints on all methods
    - State machine for phase management
    - Builder pattern for configuration assembly
    - Observer pattern for state change notifications
    - Proper error handling with user-visible messages
    - Context Object (BattleContext) instead of raw dicts
    - Separation of concerns: engine logic in MDPEngine, not here
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("MDP Battle Simulator v5.0 — See the AI Think")
        self.setMinimumSize(1200, 800)

        # Model
        self._state_machine = GameStateMachine()
        self._battle_context = BattleContext()
        self.p_stats: Optional[UnitStats] = None
        self.e_stats: Optional[UnitStats] = None
        self.p_policy: Optional[Dict] = None
        self.p_V: Optional[Dict] = None
        self.e_policy: Optional[Dict] = None
        self.e_V: Optional[Dict] = None
        self.weights: Optional[RewardWeights] = None
        self.gamma: float = CONFIG.gamma_default

        self._solver_thread: Optional[MDPSolverThread] = None
        self._enemy_solver: Optional[MDPSolverThread] = None
        self._mc_thread: Optional[MonteCarloThread] = None

        self.init_ui()

    def init_ui(self) -> None:
        """Construct the entire UI hierarchy."""
        central = QWidget()
        self.setCentralWidget(central)
        main = QVBoxLayout(central)

        title = QLabel("MDP BATTLE SIMULATOR v5.0")
        title.setStyleSheet("font-size: 26px; font-weight: bold; color: #00ffff; padding: 10px;")
        title.setAlignment(Qt.AlignCenter)
        main.addWidget(title)

        self.tabs = QTabWidget()
        main.addWidget(self.tabs)

        self.config_tab = QWidget()
        self.setup_config_tab()
        self.tabs.addTab(self.config_tab, "Configure")

        self.battle_tab = QWidget()
        self.setup_battle_tab()
        self.tabs.addTab(self.battle_tab, "Battle")
        self.tabs.setTabEnabled(1, False)

        self.results_tab = QWidget()
        self.setup_results_tab()
        self.tabs.addTab(self.results_tab, "Results")
        self.tabs.setTabEnabled(2, False)

        self.apply_theme()

    def apply_theme(self) -> None:
        """Apply the global dark theme stylesheet."""
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; }
            QWidget { background-color: #2a2a2a; color: #e0e0e0; }
            QGroupBox {
                border: 2px solid #555; border-radius: 5px;
                margin-top: 10px; font-weight: bold; color: #00ffff;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QPushButton {
                background-color: #0066cc; color: white; border: none;
                padding: 10px 20px; border-radius: 5px; font-weight: bold;
            }
            QPushButton:hover { background-color: #0088ff; }
            QPushButton:disabled { background-color: #444; color: #888; }
            QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox {
                background-color: #3a3a3a; border: 1px solid #555;
                border-radius: 3px; padding: 5px; color: white;
            }
            QTextEdit {
                background-color: #1a1a1a; border: 2px solid #555;
                border-radius: 5px; color: #00ff00;
                font-family: 'Courier New', monospace;
            }
            QTabWidget::pane { border: 2px solid #555; background-color: #2a2a2a; }
            QTabBar::tab {
                background-color: #3a3a3a; color: #e0e0e0;
                padding: 10px 20px; border-top-left-radius: 5px; border-top-right-radius: 5px;
            }
            QTabBar::tab:selected { background-color: #0066cc; color: white; }
        """)

    # ----- Configuration Tab -----

    def setup_config_tab(self) -> None:
        """Build the configuration tab with player stats, boss selection, and MC controls."""
        layout = QHBoxLayout(self.config_tab)

        # LEFT: Player config
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        stats_group = QGroupBox("Your Combat Stats")
        stats_grid = QGridLayout()
        self.stat_spins: Dict[str, QDoubleSpinBox] = {}
        stat_defs = [
            ("Slash Accuracy", "slash_acc", 0.85),
            ("Crush Accuracy", "crush_acc", 0.65),
            ("Parry Rate", "parry_rate", 0.75),
            ("Ignite Accuracy", "ignite_acc", 0.70),
            ("Poison Accuracy", "poison_acc", 0.65),
            ("Dodge Rate", "dodge_rate", 0.70),
            ("Crit Chance", "crit_chance", 0.15),
        ]
        for i, (label, key, default) in enumerate(stat_defs):
            stats_grid.addWidget(QLabel(label + ":"), i, 0)
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 1.0)
            spin.setSingleStep(0.05)
            spin.setValue(default)
            stats_grid.addWidget(spin, i, 1)
            self.stat_spins[key] = spin
        stats_group.setLayout(stats_grid)
        left_layout.addWidget(stats_group)

        rewards_group = QGroupBox("Reward Weights (Psychology)")
        rew_grid = QGridLayout()
        self.reward_spins: Dict[str, QDoubleSpinBox] = {}
        rew_defs = [
            ("Damage Dealt", "atk_w", 15.0),
            ("Damage Taken (negative)", "dmg_w", -20.0),
            ("Heal Value", "heal_w", 10.0),
            ("Burn Weight", "burn_w", 12.0),
            ("Poison Weight", "poison_w", 10.0),
        ]
        for i, (label, key, default) in enumerate(rew_defs):
            rew_grid.addWidget(QLabel(label + ":"), i, 0)
            spin = QDoubleSpinBox()
            spin.setRange(-100, 100)
            spin.setValue(default)
            rew_grid.addWidget(spin, i, 1)
            self.reward_spins[key] = spin
        rewards_group.setLayout(rew_grid)
        left_layout.addWidget(rewards_group)

        gamma_row = QHBoxLayout()
        gamma_row.addWidget(QLabel("Discount Factor (gamma):"))
        self.gamma_spin = QDoubleSpinBox()
        self.gamma_spin.setRange(0.0, 0.99)
        self.gamma_spin.setSingleStep(0.05)
        self.gamma_spin.setValue(0.90)
        gamma_row.addWidget(self.gamma_spin)
        left_layout.addLayout(gamma_row)

        btn_row = QHBoxLayout()
        save_btn = QPushButton("Save Config")
        save_btn.clicked.connect(self.save_config)
        btn_row.addWidget(save_btn)
        load_btn = QPushButton("Load Config")
        load_btn.clicked.connect(self.load_config)
        btn_row.addWidget(load_btn)
        left_layout.addLayout(btn_row)

        layout.addWidget(left)

        # RIGHT: Enemy + Monte Carlo
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        enemy_group = QGroupBox("Select Boss")
        enemy_layout = QVBoxLayout()
        self.boss_combo = QComboBox()
        presets = BossFactory.available_bosses()
        for name, data in presets.items():
            self.boss_combo.addItem(f"{name} — {data.description}", name)
        self.boss_combo.addItem("CUSTOM — Design your own boss!", "CUSTOM")
        self.boss_combo.currentIndexChanged.connect(self.on_boss_changed)
        enemy_layout.addWidget(self.boss_combo)

        self.custom_group = QGroupBox("Custom Enemy Stats")
        custom_grid = QGridLayout()
        self.custom_spins: Dict[str, Any] = {}
        custom_defs = [
            ("Max HP", "max_hp", 4, 1, 10, False),
            ("Max Stamina", "max_stam", 3, 1, 10, False),
            ("Max Potions", "max_pots", 2, 0, 10, False),
            ("Slash Acc", "slash_acc", 0.8, 0, 1, True),
            ("Crush Acc", "crush_acc", 0.7, 0, 1, True),
            ("Parry Rate", "parry_rate", 0.7, 0, 1, True),
            ("Ignite Acc", "ignite_acc", 0.7, 0, 1, True),
            ("Poison Acc", "poison_acc", 0.6, 0, 1, True),
            ("Dodge Rate", "dodge_rate", 0.6, 0, 1, True),
            ("Crit Chance", "crit_chance", 0.15, 0, 1, True),
        ]
        for i, (label, key, default, min_v, max_v, is_float) in enumerate(custom_defs):
            custom_grid.addWidget(QLabel(label + ":"), i, 0)
            if is_float:
                spin = QDoubleSpinBox()
                spin.setSingleStep(0.05)
            else:
                spin = QSpinBox()
            spin.setRange(min_v, max_v)
            spin.setValue(default)
            custom_grid.addWidget(spin, i, 1)
            self.custom_spins[key] = spin

        custom_grid.addWidget(QLabel("Enemy Atk Weight:"), 10, 0)
        self.e_atk_w = QDoubleSpinBox()
        self.e_atk_w.setRange(-100, 100)
        self.e_atk_w.setValue(15)
        custom_grid.addWidget(self.e_atk_w, 10, 1)

        custom_grid.addWidget(QLabel("Enemy Dmg Weight:"), 11, 0)
        self.e_dmg_w = QDoubleSpinBox()
        self.e_dmg_w.setRange(-100, 100)
        self.e_dmg_w.setValue(-15)
        custom_grid.addWidget(self.e_dmg_w, 11, 1)

        self.custom_group.setLayout(custom_grid)
        self.custom_group.setVisible(False)
        enemy_layout.addWidget(self.custom_group)
        enemy_group.setLayout(enemy_layout)
        right_layout.addWidget(enemy_group)

        # Monte Carlo section
        mc_group = QGroupBox("Monte Carlo Simulator")
        mc_layout = QVBoxLayout()
        mc_layout.addWidget(QLabel("Test your MDP against simulated battles:"))

        mc_btn_row = QHBoxLayout()
        self.mc_btn = QPushButton("Run 1000 Battles")
        self.mc_btn.clicked.connect(self.run_monte_carlo)
        mc_btn_row.addWidget(self.mc_btn)
        self.mc_n = QSpinBox()
        self.mc_n.setRange(100, 10000)
        self.mc_n.setSingleStep(100)
        self.mc_n.setValue(1000)
        mc_btn_row.addWidget(QLabel("N="))
        mc_btn_row.addWidget(self.mc_n)
        mc_layout.addLayout(mc_btn_row)

        self.mc_progress = QProgressBar()
        self.mc_progress.setVisible(False)
        mc_layout.addWidget(self.mc_progress)

        self.mc_result = QLabel("")
        self.mc_result.setStyleSheet("font-size: 14px; font-weight: bold; color: #ffff00;")
        self.mc_result.setWordWrap(True)
        mc_layout.addWidget(self.mc_result)

        mc_group.setLayout(mc_layout)
        right_layout.addWidget(mc_group)

        self.start_btn = QPushButton("SOLVE MDP & START BATTLE")
        self.start_btn.setStyleSheet("""
            QPushButton { background-color: #cc0066; font-size: 16px; padding: 15px; }
            QPushButton:hover { background-color: #ff0088; }
        """)
        self.start_btn.clicked.connect(self.start_game)
        right_layout.addWidget(self.start_btn)

        self.config_log = QTextEdit()
        self.config_log.setReadOnly(True)
        self.config_log.setMaximumHeight(120)
        right_layout.addWidget(self.config_log)

        layout.addWidget(right)

    # ----- Battle Tab -----

    def setup_battle_tab(self) -> None:
        """Build the battle arena with character widgets, health bars, and Q-value panel."""
        layout = QHBoxLayout(self.battle_tab)

        # LEFT: Battle arena
        arena = QWidget()
        arena_layout = QVBoxLayout(arena)
        arena_layout.setContentsMargins(0, 0, 0, 0)

        # Enemy panel
        enemy_panel = QFrame()
        enemy_panel.setStyleSheet("background-color: #3a1a1a; border-radius: 10px; padding: 10px;")
        enemy_layout = QVBoxLayout(enemy_panel)

        self.enemy_name_label = QLabel("ENEMY")
        self.enemy_name_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #ff5555;")
        self.enemy_name_label.setAlignment(Qt.AlignCenter)
        enemy_layout.addWidget(self.enemy_name_label)

        enemy_mid = QHBoxLayout()
        self.enemy_char = CharacterWidget("enemy")
        enemy_mid.addWidget(self.enemy_char)

        enemy_stats = QVBoxLayout()
        self.enemy_hp_bar = HealthBar("#ff5555")
        enemy_stats.addWidget(self.enemy_hp_bar)

        status_row = QHBoxLayout()
        self.enemy_burn_icon = StatusIcon("[BURN]", "#ff8800")
        status_row.addWidget(self.enemy_burn_icon)
        self.enemy_poison_icon = StatusIcon("[PSN]", "#88ff00")
        status_row.addWidget(self.enemy_poison_icon)
        enemy_stats.addLayout(status_row)

        info_row = QHBoxLayout()
        self.enemy_stam_label = QLabel("ST 0/0")
        self.enemy_stam_label.setStyleSheet("color: #ffff00;")
        info_row.addWidget(self.enemy_stam_label)
        self.enemy_pots_label = QLabel("POT 0")
        self.enemy_pots_label.setStyleSheet("color: #00ffff;")
        info_row.addWidget(self.enemy_pots_label)
        enemy_stats.addLayout(info_row)

        enemy_mid.addLayout(enemy_stats)
        enemy_layout.addLayout(enemy_mid)
        arena_layout.addWidget(enemy_panel)

        # Action display
        self.action_label = QLabel("Waiting...")
        self.action_label.setStyleSheet("font-size: 22px; font-weight: bold; color: #00ffff; padding: 15px;")
        self.action_label.setAlignment(Qt.AlignCenter)
        arena_layout.addWidget(self.action_label)

        # Player panel
        player_panel = QFrame()
        player_panel.setStyleSheet("background-color: #1a3a1a; border-radius: 10px; padding: 10px;")
        player_layout = QVBoxLayout(player_panel)

        player_mid = QHBoxLayout()
        self.player_char = CharacterWidget("player")
        player_mid.addWidget(self.player_char)

        player_stats = QVBoxLayout()
        self.player_hp_bar = HealthBar("#55ff55")
        player_stats.addWidget(self.player_hp_bar)

        p_status_row = QHBoxLayout()
        self.player_burn_icon = StatusIcon("[BURN]", "#ff8800")
        p_status_row.addWidget(self.player_burn_icon)
        self.player_poison_icon = StatusIcon("[PSN]", "#88ff00")
        p_status_row.addWidget(self.player_poison_icon)
        player_stats.addLayout(p_status_row)

        p_info_row = QHBoxLayout()
        self.player_stam_label = QLabel("ST 0/0")
        self.player_stam_label.setStyleSheet("color: #ffff00;")
        p_info_row.addWidget(self.player_stam_label)
        self.player_pots_label = QLabel("POT 0")
        self.player_pots_label.setStyleSheet("color: #00ffff;")
        p_info_row.addWidget(self.player_pots_label)
        player_stats.addLayout(p_info_row)

        player_mid.addLayout(player_stats)
        player_layout.addLayout(player_mid)
        arena_layout.addWidget(player_panel)

        self.continue_btn = QPushButton("NEXT TURN")
        self.continue_btn.clicked.connect(self.continue_battle)
        self.continue_btn.setEnabled(False)
        self.continue_btn.setStyleSheet("font-size: 16px; padding: 12px;")
        arena_layout.addWidget(self.continue_btn)

        layout.addWidget(arena, stretch=2)

        # RIGHT: Q-Value panel + log
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.q_panel = QValuePanel()
        right_layout.addWidget(self.q_panel)

        right_layout.addWidget(QLabel("Battle Log"))
        self.battle_log = QTextEdit()
        self.battle_log.setReadOnly(True)
        right_layout.addWidget(self.battle_log)

        layout.addWidget(right, stretch=1)

    # ----- Results Tab -----

    def setup_results_tab(self) -> None:
        """Build the results summary tab."""
        layout = QVBoxLayout(self.results_tab)

        self.results_label = QLabel("BATTLE RESULTS")
        self.results_label.setStyleSheet("font-size: 28px; font-weight: bold; color: #00ffff;")
        self.results_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.results_label)

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        layout.addWidget(self.results_text)

        self.restart_btn = QPushButton("PLAY AGAIN")
        self.restart_btn.clicked.connect(self.restart_game)
        layout.addWidget(self.restart_btn)

    # ----- Logging Helpers -----

    def _log_config(self, msg: str, color: str = "#00ff00") -> None:
        """Append an HTML-formatted message to the configuration log."""
        self.config_log.append(f'<span style="color: {color};">{msg}</span>')

    def _log_battle(self, msg: str, color: str = "#00ff00") -> None:
        """Append an HTML-formatted message to the battle log and auto-scroll."""
        self.battle_log.append(f'<span style="color: {color};">{msg}</span>')
        sb = self.battle_log.verticalScrollBar()
        sb.setValue(sb.maximum())

    # ----- Config Handlers -----

    def on_boss_changed(self, idx: int) -> None:
        """Show/hide the custom enemy stats panel based on selection."""
        boss_name = self.boss_combo.currentData()
        self.custom_group.setVisible(boss_name == "CUSTOM")

    def save_config(self) -> None:
        """Export the current configuration to a JSON file.

        Uses the UnitStats.to_dict() serialization method.
        """
        try:
            config = {
                "stats": {k: v.value() for k, v in self.stat_spins.items()},
                "rewards": {k: v.value() for k, v in self.reward_spins.items()},
                "gamma": self.gamma_spin.value(),
                "boss": self.boss_combo.currentData(),
            }
            path, _ = QFileDialog.getSaveFileName(
                self, "Save Config", "mdp_config.json", "JSON (*.json)"
            )
            if path:
                with open(path, "w") as f:
                    json.dump(config, f, indent=2)
                self._log_config(f"Config saved to {path}", "#00ff00")
                logger.info("Configuration saved to %s", path)
        except Exception as e:
            logger.exception("Failed to save configuration")
            QMessageBox.warning(self, "Save Error", f"Failed to save config: {e}")

    def load_config(self) -> None:
        """Import configuration from a JSON file.

        Uses the UnitStats.from_dict() deserialization method.
        """
        try:
            path, _ = QFileDialog.getOpenFileName(
                self, "Load Config", "", "JSON (*.json)"
            )
            if not path:
                return
            with open(path, "r") as f:
                config = json.load(f)
            for k, v in config.get("stats", {}).items():
                if k in self.stat_spins:
                    self.stat_spins[k].setValue(v)
            for k, v in config.get("rewards", {}).items():
                if k in self.reward_spins:
                    self.reward_spins[k].setValue(v)
            if "gamma" in config:
                self.gamma_spin.setValue(config["gamma"])
            if "boss" in config:
                idx = self.boss_combo.findData(config["boss"])
                if idx >= 0:
                    self.boss_combo.setCurrentIndex(idx)
            self._log_config(f"Config loaded from {path}", "#00ff00")
            logger.info("Configuration loaded from %s", path)
        except (json.JSONDecodeError, KeyError) as e:
            logger.exception("Invalid configuration file")
            QMessageBox.warning(self, "Load Error", f"Invalid config file: {e}")
        except Exception as e:
            logger.exception("Failed to load configuration")
            QMessageBox.warning(self, "Load Error", f"Failed to load config: {e}")

    # ----- Validation & Building (uses Builder Pattern) -----

    def _validate_config(self) -> bool:
        """Validate the current UI configuration.

        Checks that all probability values are in [0, 1], HP/stamina/potions
        are positive, and gamma is in (0, 1).

        Returns:
            True if the configuration is valid.

        Raises:
            Shows a QMessageBox if validation fails.
        """
        errors: List[str] = []

        for name, spin in self.stat_spins.items():
            v = spin.value()
            if not 0.0 <= v <= 1.0:
                errors.append(f"Player {name}={v:.2f} must be in [0, 1]")

        gamma = self.gamma_spin.value()
        if not 0.0 < gamma < 1.0:
            errors.append(f"Gamma={gamma:.2f} must be in (0, 1)")

        if self.boss_combo.currentData() == "CUSTOM":
            for name, spin in self.custom_spins.items():
                v = spin.value()
                if name in ("max_hp", "max_stam", "max_pots") and v < 1:
                    errors.append(f"Enemy {name}={v} must be >= 1")
                elif name not in ("max_hp", "max_stam", "max_pots") and not 0.0 <= v <= 1.0:
                    errors.append(f"Enemy {name}={v:.2f} must be in [0, 1]")

        if errors:
            msg = "Configuration errors:\n\n" + "\n".join(f"- {e}" for e in errors)
            QMessageBox.warning(self, "Invalid Configuration", msg)
            return False
        return True

    def _build_config(self) -> Tuple[UnitStats, UnitStats, RewardWeights, RewardWeights, float]:
        """Assemble game configuration using the Builder pattern.

        Returns:
            Tuple of (player_stats, enemy_stats, player_weights, enemy_weights, gamma).
        """
        builder = GameConfigBuilder()

        p_stats = UnitStats(
            4, 3, 2,
            self.stat_spins["slash_acc"].value(),
            self.stat_spins["crush_acc"].value(),
            self.stat_spins["parry_rate"].value(),
            self.stat_spins["ignite_acc"].value(),
            self.stat_spins["poison_acc"].value(),
            self.stat_spins["dodge_rate"].value(),
            self.stat_spins["crit_chance"].value(),
        )
        builder.set_player_stats(p_stats)

        boss_name = self.boss_combo.currentData()
        if boss_name == "CUSTOM":
            e_stats = UnitStats(
                self.custom_spins["max_hp"].value(),
                self.custom_spins["max_stam"].value(),
                self.custom_spins["max_pots"].value(),
                self.custom_spins["slash_acc"].value(),
                self.custom_spins["crush_acc"].value(),
                self.custom_spins["parry_rate"].value(),
                self.custom_spins["ignite_acc"].value(),
                self.custom_spins["poison_acc"].value(),
                self.custom_spins["dodge_rate"].value(),
                self.custom_spins["crit_chance"].value(),
            )
            e_weights = RewardWeights(self.e_atk_w.value(), self.e_dmg_w.value(), 10, 10, 8)
        else:
            preset = BossFactory.create(boss_name)
            e_stats = preset.stats
            e_weights = preset.weights
        builder.set_enemy_stats(e_stats)

        p_weights = RewardWeights(
            self.reward_spins["atk_w"].value(),
            self.reward_spins["dmg_w"].value(),
            self.reward_spins["heal_w"].value(),
            self.reward_spins["burn_w"].value(),
            self.reward_spins["poison_w"].value(),
        )
        builder.set_player_weights(p_weights)
        builder.set_enemy_weights(e_weights)
        builder.set_gamma(self.gamma_spin.value())

        return builder.build()

    # ----- Monte Carlo -----

    def run_monte_carlo(self) -> None:
        """Launch the Monte Carlo simulation in a background thread."""
        if not self._validate_config():
            return

        try:
            p_stats, e_stats, p_weights, e_weights, gamma = self._build_config()
        except ValueError as e:
            QMessageBox.warning(self, "Config Error", str(e))
            return

        self._log_config("Solving MDPs for Monte Carlo...", "#ffff00")
        self.mc_btn.setEnabled(False)
        self.mc_progress.setVisible(True)
        self.mc_progress.setRange(0, self.mc_n.value())
        self.mc_progress.setValue(0)
        self.mc_result.setText("Solving MDPs...")
        QApplication.processEvents()

        try:
            # Solve both MDPs synchronously (small state spaces)
            p_engine = MDPEngine(p_stats, e_stats, DefaultRewardStrategy(p_weights))
            p_policy, _, _ = p_engine.value_iteration(gamma)

            e_engine = MDPEngine(e_stats, p_stats, DefaultRewardStrategy(e_weights))
            e_policy, _, _ = e_engine.value_iteration(gamma)

            self._mc_thread = MonteCarloThread(
                p_policy, p_stats, e_policy, e_stats, self.mc_n.value()
            )
            self._mc_thread.progress.connect(self._on_mc_progress)
            self._mc_thread.finished.connect(self._on_mc_finished)
            self._mc_thread.error.connect(lambda msg: QMessageBox.critical(self, "MC Error", msg))
            self._mc_thread.start()
        except Exception as e:
            logger.exception("Monte Carlo setup failed")
            self.mc_btn.setEnabled(True)
            QMessageBox.critical(self, "Error", f"Failed to start Monte Carlo: {e}")

    def _on_mc_progress(self, done: int, wins: int, losses: int, draws: int) -> None:
        """Update Monte Carlo progress bar and live statistics."""
        self.mc_progress.setValue(done)
        total = done or 1
        self.mc_result.setText(
            f"Progress: {done}/{self.mc_n.value()} | "
            f"W:{wins} L:{losses} D:{draws} | "
            f"Win rate: {wins / total * 100:.1f}%"
        )

    def _on_mc_finished(self, wins: int, losses: int, draws: int) -> None:
        """Display final Monte Carlo results with color-coded win rate."""
        self.mc_progress.setValue(self.mc_n.value())
        self.mc_btn.setEnabled(True)
        total = wins + losses + draws
        wr = wins / total * 100 if total > 0 else 0
        color = "#00ff00" if wr > 60 else "#ffff00" if wr > 40 else "#ff5555"
        self.mc_result.setText(
            f'<span style="color: {color}; font-size: 16px;">'
            f"Final: {wins}W / {losses}L / {draws}D  —  Win Rate: {wr:.1f}%</span>"
        )

    # ----- Game Lifecycle -----

    def start_game(self) -> None:
        """Validate config, transition to SOLVING, and launch the MDP solver thread."""
        if not self._validate_config():
            return

        try:
            p_stats, e_stats, p_weights, e_weights, gamma = self._build_config()
        except ValueError as e:
            QMessageBox.warning(self, "Config Error", str(e))
            return

        self.p_stats = p_stats
        self.e_stats = e_stats
        self.weights = p_weights
        self.gamma = gamma
        self._battle_context = BattleContext(enemy_name=self.boss_combo.currentData() or "ENEMY")

        self.start_btn.setEnabled(False)
        self._state_machine.transition_to(GamePhase.SOLVING)
        self._log_config("Solving your MDP...", "#ffff00")
        QApplication.processEvents()

        p_engine = MDPEngine(p_stats, e_stats, DefaultRewardStrategy(p_weights))
        self._solver_thread = MDPSolverThread(p_engine, gamma)
        self._solver_thread.progress.connect(self._on_solver_progress)
        self._solver_thread.finished.connect(self._on_player_solved)
        self._solver_thread.error.connect(lambda msg: self._handle_solver_error(msg, is_player=True))
        self._solver_thread.start()

    def _on_solver_progress(self, iters: int, delta: float) -> None:
        """Log solver iteration progress."""
        self._log_config(f"  Iteration {iters}, delta={delta:.6f}", "#888888")

    def _on_player_solved(self, p_policy: dict, p_V: dict, iters: int) -> None:
        """Handle completion of the player's MDP, then solve the enemy's MDP."""
        self.p_policy = p_policy
        self.p_V = p_V
        self._log_config(f"Your MDP solved in {iters} iterations", "#00ff00")

        boss_name = self.boss_combo.currentData()
        if boss_name == "CUSTOM":
            e_weights = RewardWeights(self.e_atk_w.value(), self.e_dmg_w.value(), 10, 10, 8)
        else:
            e_weights = BossFactory.create(boss_name).weights

        self._log_config("Solving enemy MDP...", "#ffff00")
        e_engine = MDPEngine(self.e_stats, self.p_stats, DefaultRewardStrategy(e_weights))
        self._enemy_solver = MDPSolverThread(e_engine, self.gamma)
        self._enemy_solver.finished.connect(self._on_enemy_solved)
        self._enemy_solver.error.connect(lambda msg: self._handle_solver_error(msg, is_player=False))
        self._enemy_solver.start()

    def _on_enemy_solved(self, e_policy: dict, e_V: dict, iters: int) -> None:
        """Handle completion of the enemy's MDP and transition to BATTLE phase."""
        self.e_policy = e_policy
        self.e_V = e_V
        self._log_config(f"Enemy MDP solved in {iters} iterations", "#00ff00")

        self._state_machine.transition_to(GamePhase.BATTLE)
        self.tabs.setTabEnabled(1, True)
        self.tabs.setCurrentIndex(1)
        self._init_battle_display()

    def _handle_solver_error(self, message: str, is_player: bool) -> None:
        """Handle MDP solver errors with user feedback and state rollback."""
        who = "player" if is_player else "enemy"
        logger.error("%s MDP solver error: %s", who, message)
        QMessageBox.critical(self, "Solver Error", f"Failed to solve {who} MDP:\n{message}")
        self.start_btn.setEnabled(True)
        try:
            self._state_machine.transition_to(GamePhase.CONFIGURATION)
        except InvalidStateError:
            self._state_machine.reset()

    def _init_battle_display(self) -> None:
        """Initialize the battle display with full HP/stamina/potions."""
        ctx = self._battle_context
        ctx.player_hp = self.p_stats.max_hp if self.p_stats else 4
        ctx.enemy_hp = self.e_stats.max_hp if self.e_stats else 4
        ctx.player_st = self.p_stats.max_stam if self.p_stats else 3
        ctx.enemy_st = self.e_stats.max_stam if self.e_stats else 3
        ctx.player_pt = self.p_stats.max_pots if self.p_stats else 2
        ctx.enemy_pt = self.e_stats.max_pots if self.e_stats else 2
        ctx.turn = 0

        self.enemy_name_label.setText(ctx.enemy_name)
        self._refresh_battle_display()
        self.battle_log.clear()
        self._log_battle(f"=== FIGHTING {ctx.enemy_name} ===", "#00ffff")
        self._log_battle("Click NEXT TURN to begin...", "#ffff00")
        self.continue_btn.setEnabled(True)
        self._update_q_panel()

    def _update_q_panel(self) -> None:
        """Recompute and display Q-values for the current battle state."""
        if self.p_policy is None or self.p_V is None or self.p_stats is None:
            return
        state_tuple = tuple(self._battle_context.to_battle_state())
        action = Action(self.p_policy.get(state_tuple, 0))
        state_value = self.p_V.get(state_tuple, 0.0)

        # Compute Q-values using the engine
        engine = MDPEngine(
            self.p_stats, self.e_stats,
            DefaultRewardStrategy(self.weights) if self.weights else DefaultRewardStrategy(
                RewardWeights(15, -20, 10, 12, 10)
            ),
        )
        q_vals = engine.compute_q_values(
            self._battle_context.to_battle_state(),
            {BattleState(*k): v for k, v in self.p_policy.items()},
            {BattleState(*k): v for k, v in self.p_V.items()},
            self.gamma,
        )
        self.q_panel.update_values(q_vals, state_value, int(action))

    def _refresh_battle_display(self) -> None:
        """Synchronize all UI elements with the current BattleContext."""
        ctx = self._battle_context

        if self.e_stats:
            self.enemy_hp_bar.setMaximum(self.e_stats.max_hp)
            self.enemy_hp_bar.animate_value(ctx.enemy_hp)
            self.enemy_hp_bar.setFormat(f"{ctx.enemy_hp}/{self.e_stats.max_hp}")

        if self.p_stats:
            self.player_hp_bar.setMaximum(self.p_stats.max_hp)
            self.player_hp_bar.animate_value(ctx.player_hp)
            self.player_hp_bar.setFormat(f"{ctx.player_hp}/{self.p_stats.max_hp}")

        self.enemy_burn_icon.set_count(ctx.enemy_br)
        self.enemy_poison_icon.set_count(ctx.enemy_ps)
        self.enemy_stam_label.setText(f"ST {ctx.enemy_st}/{self.e_stats.max_stam if self.e_stats else 0}")
        self.enemy_pots_label.setText(f"POT {ctx.enemy_pt}")

        self.player_burn_icon.set_count(ctx.player_br)
        self.player_poison_icon.set_count(ctx.player_ps)
        self.player_stam_label.setText(f"ST {ctx.player_st}/{self.p_stats.max_stam if self.p_stats else 0}")
        self.player_pots_label.setText(f"POT {ctx.player_pt}")

        if self.e_stats:
            self.enemy_char.update_display(
                ctx.enemy_hp / self.e_stats.max_hp if self.e_stats.max_hp > 0 else 0,
                ctx.enemy_br > 0, ctx.enemy_ps > 0,
            )
        if self.p_stats:
            self.player_char.update_display(
                ctx.player_hp / self.p_stats.max_hp if self.p_stats.max_hp > 0 else 0,
                ctx.player_br > 0, ctx.player_ps > 0,
            )

    def _spawn_damage_number(self, text: str, color: str, target_is_enemy: bool) -> None:
        """Create a floating damage/heal number above the target character widget.

        Args:
            text: The text to display (e.g., "-1", "+2").
            color: CSS color string.
            target_is_enemy: If True, float above the enemy; otherwise the player.
        """
        if target_is_enemy:
            pos = self.enemy_char.pos() + QPoint(30, 20)
            parent = self.enemy_char.parentWidget()
        else:
            pos = self.player_char.pos() + QPoint(30, 20)
            parent = self.player_char.parentWidget()
        FloatingDamage(parent, text, color, pos)

    # ----- Battle Logic -----

    def continue_battle(self) -> None:
        """Execute one full turn (player + enemy) of the battle."""
        ctx = self._battle_context

        if ctx.player_hp <= 0 or ctx.enemy_hp <= 0 or ctx.turn >= CONFIG.max_battle_turns:
            self._end_battle()
            return

        ctx.turn += 1
        self._log_battle(f"\n--- Turn {ctx.turn} ---", "#00ffff")

        # --- Player turn ---
        p_state_tuple = tuple(ctx.to_battle_state())
        if self.p_policy is None:
            return

        p_act_idx = self.p_policy.get(p_state_tuple, 0)
        p_action = Action(p_act_idx)
        self.action_label.setText(f"YOU: {p_action.name}")
        self._log_battle(f"You use {p_action.name}", "#55ff55")

        trans = get_transitions_compat(p_state_tuple, p_act_idx, self.p_stats) if self.p_stats else []
        if trans:
            probs, next_states = zip(*trans)
            old_hp_e, old_hp_m, old_pt = ctx.enemy_hp, ctx.player_hp, ctx.player_pt
            new_state = random.choices(next_states, weights=probs, k=1)[0]
            ctx.player_hp, ctx.enemy_hp, ctx.player_st, ctx.player_pt, \
                ctx.player_br, ctx.enemy_br, ctx.player_ps, ctx.enemy_ps = new_state

            if ctx.enemy_hp < old_hp_e:
                dmg = old_hp_e - ctx.enemy_hp
                self._log_battle(f"  -> You dealt {dmg} damage!", "#55ff55")
                self._spawn_damage_number(f"-{dmg}", "#ff5555", True)
                self.enemy_char.shake()
                ctx.stats["dmg_dealt"] += dmg
            if ctx.player_hp < old_hp_m:
                dmg = old_hp_m - ctx.player_hp
                self._log_battle(f"  -> You took {dmg} damage!", "#ff5555")
                self._spawn_damage_number(f"-{dmg}", "#ff0000", False)
                self.player_char.shake()
                ctx.stats["dmg_taken"] += dmg
            if ctx.player_hp > old_hp_m:
                heal = ctx.player_hp - old_hp_m
                self._log_battle(f"  -> You healed {heal} HP!", "#00ffff")
                self._spawn_damage_number(f"+{heal}", "#00ffff", False)
            if ctx.player_pt < old_pt:
                ctx.stats["pots_used"] += 1

        self._refresh_battle_display()
        QApplication.processEvents()

        if ctx.enemy_hp <= 0 or ctx.player_hp <= 0:
            QTimer.singleShot(800, self._end_battle)
            return

        # --- Enemy turn (delayed) ---
        QTimer.singleShot(600, self._enemy_turn)

    def _enemy_turn(self) -> None:
        """Execute the enemy's turn."""
        ctx = self._battle_context
        if self.e_policy is None or self.e_stats is None:
            return

        e_state_tuple = ctx.enemy_perspective_state()
        e_act_idx = self.e_policy.get(e_state_tuple, 0)
        e_action = Action(e_act_idx)
        self.action_label.setText(f"{ctx.enemy_name}: {e_action.name}")
        self._log_battle(f"{ctx.enemy_name} uses {e_action.name}", "#ff8855")

        trans = get_transitions_compat(e_state_tuple, e_act_idx, self.e_stats)
        if trans:
            e_probs, e_next_states = zip(*trans)
            old_e_hp, old_m_hp = ctx.enemy_hp, ctx.player_hp
            new_e_state = random.choices(e_next_states, weights=e_probs, k=1)[0]
            ctx.enemy_hp, ctx.player_hp, ctx.enemy_st, ctx.enemy_pt, \
                ctx.enemy_br, ctx.player_br, ctx.enemy_ps, ctx.player_ps = new_e_state

            if ctx.player_hp < old_m_hp:
                dmg = old_m_hp - ctx.player_hp
                self._log_battle(f"  -> {ctx.enemy_name} dealt {dmg} damage!", "#ff5555")
                self._spawn_damage_number(f"-{dmg}", "#ff0000", False)
                self.player_char.shake()
                ctx.stats["dmg_taken"] += dmg
            if ctx.enemy_hp < old_e_hp:
                dmg = old_e_hp - ctx.enemy_hp
                self._log_battle(f"  -> {ctx.enemy_name} took {dmg} damage!", "#55ff55")
                self._spawn_damage_number(f"-{dmg}", "#ff5555", True)
                self.enemy_char.shake()

        self._refresh_battle_display()
        self._update_q_panel()

        if ctx.enemy_hp <= 0 or ctx.player_hp <= 0:
            QTimer.singleShot(800, self._end_battle)

    def _end_battle(self) -> None:
        """Transition to RESULTS phase and display the battle summary."""
        ctx = self._battle_context
        self.continue_btn.setEnabled(False)

        self._state_machine.transition_to(GamePhase.RESULTS)
        self.tabs.setTabEnabled(2, True)
        self.tabs.setCurrentIndex(2)

        if ctx.player_hp > 0:
            self.results_label.setText("VICTORY!")
            self.results_label.setStyleSheet("font-size: 28px; font-weight: bold; color: #00ff00;")
            self.results_text.append(f"You defeated {ctx.enemy_name} in {ctx.turn} turns!")
        else:
            self.results_label.setText("DEFEAT")
            self.results_label.setStyleSheet("font-size: 28px; font-weight: bold; color: #ff0000;")
            self.results_text.append(f"You were defeated by {ctx.enemy_name} on turn {ctx.turn}.")

        self.results_text.append("\nBattle Statistics:")
        self.results_text.append(f"  Damage Dealt: {ctx.stats['dmg_dealt']}")
        self.results_text.append(f"  Damage Taken: {ctx.stats['dmg_taken']}")
        self.results_text.append(f"  Potions Used: {ctx.stats['pots_used']}")
        self.results_text.append(f"  Turns: {ctx.turn}")

        self.results_text.append("\nMDP Info:")
        state_count = len(self.p_V) if self.p_V else 0
        self.results_text.append(f"  States in MDP: {state_count:,}")
        final_state = (
            ctx.player_hp, ctx.enemy_hp, 0, 0, 0, 0, 0, 0
        )
        final_v = self.p_V.get(final_state, 0) if self.p_V else 0
        self.results_text.append(f"  Final State Value: {final_v:.2f}")

    def restart_game(self) -> None:
        """Reset all state and return to CONFIGURATION phase."""
        self._state_machine.reset()
        self._battle_context = BattleContext()
        self.p_policy = None
        self.p_V = None
        self.e_policy = None
        self.e_V = None

        self.tabs.setCurrentIndex(0)
        self.tabs.setTabEnabled(1, False)
        self.tabs.setTabEnabled(2, False)
        self.config_log.clear()
        self.battle_log.clear()
        self.results_text.clear()
        self.start_btn.setEnabled(True)
        self.action_label.setText("Waiting...")


# =============================================================================
# 18. ENTRY POINT (Pattern: Guarded Main)
# =============================================================================
def main() -> None:
    """Application entry point with guarded execution."""
    app = QApplication(sys.argv)
    window = MDPBattleGame()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()