"""Pure-Python, checkpointable shared D-learning-rate controller.

Inputs must already be global, row-weighted raw GAN losses. D is pre-update
window loss; G is actual post-update active loss. Each series is compared only
with its own history. g_proxy is optional audit evidence, never a G substitute.
No torch, optimizers, collectives, generator learning rate or MMD are touched.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
import copy
import hashlib
import json
import math
import numbers


HEADS = ('dinov3_l', 'qwen3vl_4b', 'aimv2_huge')
HEAD_KEYS = tuple(f's{step}/{head}' for step in (1, 2) for head in HEADS)


def _number(value):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError('Expected a finite real number')
    try:
        value = float(value)
    except (OverflowError, ValueError):
        raise ValueError('Expected a finite real number') from None
    if not math.isfinite(value):
        raise ValueError('Expected a finite real number')
    return value


def _integer(value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError('Invalid integer counter/configuration')
    return value


def _median(values):
    values = sorted(values)
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    # Avoid overflow of(a+b)/2 for otherwise finite nonnegative loss values.
    return values[middle-1] + (values[middle] - values[middle-1]) / 2


def _ge(value, threshold):
    return value >= threshold or math.isclose(value, threshold, rel_tol=1e-12, abs_tol=1e-15)


@dataclass(frozen=True)
class AdaptiveDLRConfig:
    enabled: bool = False
    window_size: int = 10
    min_abs_change: float = 0.05
    min_relative_change: float = 0.20
    cooldown_steps: int = 20
    min_lr: float = 1e-7
    max_lr: float = 2e-5
    decrease_factor: float = 0.5
    increase_factor: float = 1.25
    loss_floor: float = 1e-6
    volatility_ratio: float = 4.0
    volatility_abs_range: float = 0.5
    volatility_adjacent_ratio: float = 2.0
    volatility_min_jumps: int = 3

    @classmethod
    def from_mapping(cls, mapping=None):
        if mapping is None:
            mapping = {}
        if not isinstance(mapping, Mapping):
            raise ValueError('adaptive_d_lr must be a mapping')
        unknown = set(mapping) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f'Unknown adaptive_d_lr configuration keys:{sorted(unknown)}')
        config = cls(**dict(mapping))
        config.validate()
        return config

    def validate(self):
        if type(self.enabled) is not bool:
            raise ValueError('adaptive_d_lr.enabled must be a boolean')
        _integer(self.window_size, 2)
        _integer(self.cooldown_steps)
        _integer(self.volatility_min_jumps, 1)
        if self.volatility_min_jumps >= self.window_size:
            raise ValueError('volatility_min_jumps must fit the window adjacent pairs')
        for name in ('min_abs_change', 'min_lr', 'max_lr', 'loss_floor', 'volatility_abs_range'):
            if _number(getattr(self, name)) <= 0:
                raise ValueError(f'{name} must be positive')
        if not 0 < _number(self.min_relative_change) < 1:
            raise ValueError('min_relative_change must be between0 and1')
        if not 0 < _number(self.decrease_factor) < 1 or _number(self.increase_factor) <= 1:
            raise ValueError('D learning-rate factors must decrease/increase respectively')
        if self.max_lr <= self.min_lr:
            raise ValueError('max_lr must exceed min_lr')
        if _number(self.volatility_ratio) <= 1 or _number(self.volatility_adjacent_ratio) <= 1:
            raise ValueError('Volatility ratios must exceed1')


def _percentile(values, q):
    values = sorted(values)
    index = (len(values) - 1) * q
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


class AdaptiveDLRController:
    VERSION = 1

    def __init__(self, base_lr=1e-5, cfg=None, *, heads=HEADS):
        self.cfg = AdaptiveDLRConfig() if cfg is None else cfg
        if not isinstance(self.cfg, AdaptiveDLRConfig):
            raise ValueError('cfg must be AdaptiveDLRConfig')
        self.cfg.validate()
        self.heads = tuple(heads)
        if self.heads not in (HEADS, HEADS[:2]):
            raise ValueError('Adaptive D LR requires DINOv3/Qwen-VL, optionally legacy AIM')
        self.head_keys = tuple(f's{step}/{head}' for step in (1, 2) for head in self.heads)
        self.base_lr = _number(base_lr)
        if not self.cfg.min_lr <= self.base_lr <= self.cfg.max_lr:
            raise ValueError('Base D learning rate is outside the configured finite bounds')
        self.current_lr = self.base_lr
        self.policy_hash = hashlib.sha256(json.dumps(asdict(self.cfg), sort_keys=True,
                                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        self.last_completed_step = 0
        self.valid_observations = 0
        self.invalid_observations = 0
        self.cooldown_remaining = 0
        self.action_sequence = 0
        self.last_action = None
        self._mode = None
        self._history = []

    def _metrics(self, values, active):
        if not isinstance(values, Mapping) or set(values) != set(self.head_keys):
            raise ValueError('Expected exactly the configured GAN step/head metric keys')
        result = {}
        for name in self.head_keys:
            row = values[name]
            if not isinstance(row, Mapping) or 'd_loss' not in row \
                    or set(row) - {'d_loss', 'g_loss', 'g_proxy'}:
                raise ValueError(f'Invalid metric fields:{name}')
            clean = {}
            for key in ('d_loss', 'g_loss', 'g_proxy'):
                value = row.get(key)
                if value is None:
                    if key == 'd_loss' or key == 'g_loss' and active:
                        raise ValueError(f'Missing required {key}:{name}')
                    clean[key] = None
                else:
                    value = _number(value)
                    if value < 0:
                        raise ValueError(f'Raw GAN loss must be nonnegative:{name}/{key}')
                    clean[key] = value
            if not active:
                clean['g_loss'] = None  # Warmup0 is never a true G observation.
            result[name] = clean
        return result

    def _trend(self, values):
        width = self.cfg.window_size
        if len(values) < 2 * width or any(value is None for value in values):
            return None
        old, new = values[-2*width:-width], values[-width:]
        old_log = _median(math.log(max(value, self.cfg.loss_floor)) for value in old)
        new_log = _median(math.log(max(value, self.cfg.loss_floor)) for value in new)
        # Compare logs to avoid overflow when finite raw losses have huge ratios.
        difference = new_log - old_log
        raw_delta = _median(new) - _median(old)
        up = _ge(raw_delta, self.cfg.min_abs_change) and _ge(difference, math.log1p(self.cfg.min_relative_change))
        down = _ge(-raw_delta, self.cfg.min_abs_change) and _ge(-difference, -math.log1p(-self.cfg.min_relative_change))
        return {'old_median': _median(old), 'new_median': _median(new),
                'log_ratio': difference, 'absolute_delta': raw_delta,
                'direction': 'up' if up else 'down' if down else 'flat'}

    def _volatility(self, values):
        width = self.cfg.window_size
        values = values[-width:]
        if len(values) != width or any(value is None for value in values):
            return None
        low, high = _percentile(values, 0.1), _percentile(values, 0.9)
        jumps = sum(_ge(math.log(max(max(a, b), self.cfg.loss_floor)) -
                    math.log(max(min(a, b), self.cfg.loss_floor)), math.log(self.cfg.volatility_adjacent_ratio))
                    for a, b in zip(values, values[1:]))
        log_ratio = math.log(max(high, self.cfg.loss_floor)) - math.log(max(low, self.cfg.loss_floor))
        return {'p10': low, 'p90': high, 'absolute_range': high-low, 'log_ratio': log_ratio,
                'large_adjacent_changes': jumps,
                'large': _ge(log_ratio, math.log(self.cfg.volatility_ratio))
                         and _ge(high-low, self.cfg.volatility_abs_range)
                         and jumps >= self.cfg.volatility_min_jumps}

    def _statistics(self, active):
        heads, votes = {}, {f's{step}': {'decrease': [], 'increase': [], 'volatile': []} for step in (1, 2)}
        conflicts = []
        for name in self.head_keys:
            series = {key: [point['metrics'][name][key] for point in self._history]
                      for key in ('d_loss', 'g_loss', 'g_proxy')}
            trends = {key: self._trend(value) for key, value in series.items()}
            volatility = {key: self._volatility(value) for key, value in series.items()}
            step = name.split('/')[0]
            if any(volatility[key] and volatility[key]['large'] for key in
                   (('d_loss', 'g_loss') if active else ('d_loss',))):
                votes[step]['volatile'].append(name)
            d, g, proxy = (trends[key] for key in ('d_loss', 'g_loss', 'g_proxy'))
            direction = None
            if active and d and g:
                if d['direction'] == 'down' and g['direction'] == 'up': direction = 'decrease'
                if d['direction'] == 'up' and g['direction'] == 'down': direction = 'increase'
                if direction: votes[step][direction].append(name)
                if proxy and g['direction'] in ('up', 'down') and proxy['direction'] in ('up', 'down') \
                        and g['direction'] != proxy['direction']:
                    conflicts.append(name)
            heads[name] = {'trends': trends, 'volatility': volatility, 'vote': direction}
        width = self.cfg.window_size
        steps = [point['completed_step'] for point in self._history]
        return {'valid_window_points': len(steps), 'window_steps': steps,
                'old_window_steps': steps[-2*width:-width] if len(steps) >= 2*width else [],
                'new_window_steps': steps[-width:], 'heads': heads, 'votes': votes,
                'proxy_conflicts': conflicts, 'actual_G_is_control_signal': active}

    def _decision(self, step, reason, statistics=None, **extra):
        return {'completed_step': step, 'effective_from_step': step + 1,
                'old_lr': self.current_lr, 'new_lr': self.current_lr, 'action': 'hold',
                'reason': reason, 'changed': False, 'action_sequence': self.action_sequence,
                'cooldown_remaining': self.cooldown_remaining,
                'valid_observations': self.valid_observations,
                'invalid_observations': self.invalid_observations,
                'statistics': {} if statistics is None else statistics, **extra}

    def observe(self, completed_step, head_metrics, *, generator_active, valid):
        _integer(completed_step, 1)
        if type(generator_active) is not bool or type(valid) is not bool:
            raise ValueError('generator_active and valid must be booleans')
        if completed_step <= self.last_completed_step:
            return self._decision(completed_step, 'nonmonotonic_step')
        gap = self.last_completed_step > 0 and completed_step != self.last_completed_step + 1
        self.last_completed_step = completed_step
        try:
            if not valid: raise ValueError('caller_marked_invalid')
            metrics = self._metrics(head_metrics, generator_active)
        except ValueError as error:
            self.invalid_observations += 1
            self._history.clear()
            return self._decision(completed_step, 'invalid_observation', invalid_detail=str(error))
        self.valid_observations += 1
        if gap or self._mode != generator_active:
            self._history.clear()
        self._mode = generator_active
        if not self.cfg.enabled:
            self._history.clear()
            return self._decision(completed_step, 'disabled')
        self._history.append({'completed_step': completed_step, 'generator_active': generator_active,
                              'metrics': metrics})
        self._history = self._history[-2*self.cfg.window_size:]
        statistics = self._statistics(generator_active)
        if self.cooldown_remaining:
            self.cooldown_remaining -= 1
            return self._decision(completed_step, 'cooldown', statistics)
        votes = statistics['votes']
        volatile = any(len(votes[step]['volatile']) >= 2 for step in ('s1', 's2'))
        if volatile:
            direction, reason = 'decrease', 'large_volatility'
        elif not generator_active:
            return self._decision(completed_step, 'generator_warmup', statistics)
        elif len(self._history) < 2 * self.cfg.window_size:
            return self._decision(completed_step, 'insufficient_actual_G_history', statistics)
        elif statistics['proxy_conflicts']:
            return self._decision(completed_step, 'actual_G_proxy_conflict', statistics)
        else:
            down = sum(len(votes[step]['decrease']) for step in ('s1', 's2'))
            up = sum(len(votes[step]['increase']) for step in ('s1', 's2'))
            if down and up:
                return self._decision(completed_step, 'opposing_head_trends', statistics)
            if all(len(votes[step]['decrease']) >= 2 for step in ('s1', 's2')):
                direction, reason = 'decrease', 'D_falling_actual_G_rising'
            elif all(len(votes[step]['increase']) >= 2 for step in ('s1', 's2')):
                direction, reason = 'increase', 'D_rising_actual_G_falling'
            else:
                return self._decision(completed_step, 'no_consistent_two_step_trend', statistics)
        factor = self.cfg.decrease_factor if direction == 'decrease' else self.cfg.increase_factor
        proposed = self.current_lr * factor
        overflow = not math.isfinite(proposed)
        if overflow:
            proposed = self.cfg.max_lr  # Keep diagnostic JSON finite as well as applied LR.
        new_lr = min(self.cfg.max_lr, max(self.cfg.min_lr, proposed))
        if new_lr == self.current_lr:
            return self._decision(completed_step, 'lower_bound' if direction == 'decrease' else 'upper_bound',
                                  statistics, proposed_lr=proposed, clipped=True)
        decision = self._decision(completed_step, reason, statistics)
        self.current_lr = new_lr
        self.action_sequence += 1
        self.cooldown_remaining = self.cfg.cooldown_steps
        self.last_action = {'completed_step': completed_step, 'old_lr': decision['old_lr'],
                            'new_lr': new_lr, 'action': direction, 'reason': reason,
                            'action_sequence': self.action_sequence}
        self._history.clear()  # Never compare samples produced across an LR change.
        decision.update(new_lr=new_lr, action=direction, changed=True, proposed_lr=proposed,
                        clipped=overflow or new_lr != proposed, action_sequence=self.action_sequence,
                        cooldown_remaining=self.cooldown_remaining)
        return decision

    def state_dict(self):
        return copy.deepcopy({'version': self.VERSION, 'config': asdict(self.cfg),
            'policy_hash': self.policy_hash, 'base_lr': self.base_lr, 'current_lr': self.current_lr,
            'head_keys': list(self.head_keys), 'last_completed_step': self.last_completed_step,
            'valid_observations': self.valid_observations, 'invalid_observations': self.invalid_observations,
            'cooldown_remaining': self.cooldown_remaining, 'action_sequence': self.action_sequence,
            'last_action': self.last_action, 'generator_active_mode': self._mode, 'history': self._history})

    def load_state_dict(self, state):
        if not isinstance(state, Mapping) or set(state) != set(self.state_dict()):
            raise ValueError('Adaptive D LR state schema mismatch')
        saved_config = AdaptiveDLRConfig.from_mapping(state['config'])
        if _integer(state['version'], 1) != self.VERSION or asdict(saved_config) != asdict(self.cfg) \
                or state['policy_hash'] != self.policy_hash or state['head_keys'] != list(self.head_keys) \
                or _number(state['base_lr']) != self.base_lr:
            raise ValueError('Adaptive D LR state policy/base/head contract mismatch')
        lr = _number(state['current_lr'])
        if not self.cfg.min_lr <= lr <= self.cfg.max_lr:
            raise ValueError('Saved effective D LR is outside finite bounds')
        step = _integer(state['last_completed_step'])
        count = _integer(state['valid_observations'])
        invalid = _integer(state['invalid_observations'])
        sequence = _integer(state['action_sequence'])
        cooldown = _integer(state['cooldown_remaining'])
        mode, history, action = state['generator_active_mode'], state['history'], state['last_action']
        if count + invalid > step or sequence > count or cooldown > self.cfg.cooldown_steps \
                or mode is not None and type(mode) is not bool \
                or (count == 0) != (mode is None) \
                or not isinstance(history, list) or len(history) > min(count, 2*self.cfg.window_size):
            raise ValueError('Invalid adaptive D LR counts/phase/history')
        previous = None
        clean_history = []
        for row in history:
            if not isinstance(row, Mapping) or set(row) != {'completed_step', 'generator_active', 'metrics'}:
                raise ValueError('Invalid adaptive D LR history schema')
            index = _integer(row['completed_step'], 1)
            if index > step or previous is not None and index != previous + 1 \
                    or type(row['generator_active']) is not bool or row['generator_active'] != mode:
                raise ValueError('Adaptive D LR history has mismatched step/phase')
            metrics = self._metrics(row['metrics'], mode)
            if metrics != row['metrics']:
                raise ValueError('Adaptive D LR history contains warmup G values')
            clean_history.append({'completed_step': index, 'generator_active': mode, 'metrics': metrics})
            previous = index
        if history and previous != step:
            raise ValueError('Adaptive D LR history does not end at its latest completed step')
        if cooldown > max(0, self.cfg.cooldown_steps - len(history)):
            raise ValueError('Adaptive D LR cooldown is inconsistent with post-action valid history')
        if not sequence:
            if action is not None or cooldown or lr != self.base_lr:
                raise ValueError('Adaptive D LR state changed without a recorded action')
        else:
            required = {'completed_step', 'old_lr', 'new_lr', 'action', 'reason', 'action_sequence'}
            if not isinstance(action, Mapping) or set(action) != required \
                    or _integer(action['completed_step'], 1) > step \
                    or _integer(action['action_sequence'], 1) != sequence \
                    or action['action'] not in ('increase', 'decrease') \
                    or action['reason'] not in ('large_volatility', 'D_falling_actual_G_rising', 'D_rising_actual_G_falling'):
                raise ValueError('Invalid adaptive D LR last-action record')
            old, new = _number(action['old_lr']), _number(action['new_lr'])
            expected_direction = 'increase' if action['reason'] == 'D_rising_actual_G_falling' else 'decrease'
            if action['action'] != expected_direction:
                raise ValueError('Adaptive D LR action direction disagrees with its reason')
            factor = self.cfg.increase_factor if action['action'] == 'increase' else self.cfg.decrease_factor
            expected = min(self.cfg.max_lr, max(self.cfg.min_lr, old * factor))
            if not self.cfg.min_lr <= old <= self.cfg.max_lr or new != lr or new != expected or new == old \
                    or history and history[0]['completed_step'] <= action['completed_step']:
                raise ValueError('Adaptive D LR action/rate/history mismatch')
            if cooldown < max(0, self.cfg.cooldown_steps - (step - action['completed_step'])):
                raise ValueError('Adaptive D LR cooldown expires before enough completed steps')
        if not self.cfg.enabled and (sequence or history):
            raise ValueError('Disabled adaptive D LR cannot contain actions/history')
        # Publish only after every check passes; invalid checkpoint cannot partially mutate state.
        self.current_lr, self.last_completed_step = lr, step
        self.valid_observations, self.invalid_observations = count, invalid
        self.action_sequence, self.cooldown_remaining = sequence, cooldown
        self.last_action, self._mode, self._history = copy.deepcopy(action), mode, clean_history
