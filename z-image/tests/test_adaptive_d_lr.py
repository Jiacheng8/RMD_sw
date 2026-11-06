"""Pure Python controller tests: no torch, models, optimizers or remote calls."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import sys
import unittest

SOURCE = Path(__file__).resolve().parents[2] / 'rdm/sw_lmmd/adaptive_d_lr.py'
spec = importlib.util.spec_from_file_location('adaptive_d_lr_under_test', SOURCE)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
Config, Controller, KEYS = module.AdaptiveDLRConfig, module.AdaptiveDLRController, module.HEAD_KEYS


def metrics(d=2., g=1., proxy=None):
    return {name: {'d_loss': d, 'g_loss': g, 'g_proxy': proxy} for name in KEYS}


def controller(**overrides):
    return Controller(base_lr=1e-5, cfg=Config.from_mapping({'enabled': True, **overrides}))


def sequence(owner, old=(2., 1.), new=(1., 2.), *, start=51, modify=None, proxy=True):
    decisions = []
    for index in range(20):
        d, g = old if index < 10 else new
        values = metrics(d, g, g if proxy else None)
        if modify is not None: modify(index, values)
        decisions.append(owner.observe(start + index, values, generator_active=True, valid=True))
    return decisions


class AdaptiveDLRTests(unittest.TestCase):
    def test_two_gan_heads_control_and_resume_without_aim_metrics(self):
        config = Config.from_mapping({'enabled': True})
        for old, new, action in [((2., 1.), (1., 2.), 'decrease'),
                                  ((1., 2.), (2., 1.), 'increase')]:
            owner = Controller(cfg=config, heads=module.HEADS[:2])
            for index in range(20):
                d, g = old if index < 10 else new
                values = {name: {'d_loss': d, 'g_loss': g, 'g_proxy': g}
                          for name in owner.head_keys}
                decision = owner.observe(51 + index, values, generator_active=True, valid=True)
                if index == 14:
                    resumed = Controller(cfg=config, heads=module.HEADS[:2])
                    resumed.load_state_dict(json.loads(json.dumps(owner.state_dict())))
                elif index > 14:
                    self.assertEqual(resumed.observe(51 + index, values, generator_active=True, valid=True),
                                     decision)
            self.assertEqual(decision['action'], action)
            self.assertEqual(len(decision['statistics']['heads']), 4)
            with self.assertRaisesRegex(ValueError, 'head contract'):
                Controller(cfg=config).load_state_dict(owner.state_dict())
        for wrong in [('dinov3_l',), ('qwen3vl_4b', 'dinov3_l'), ('dinov3_l', 'siglip2')]:
            with self.assertRaises(ValueError): Controller(cfg=config, heads=wrong)

    def test_two_heads_require_both_votes_and_never_treat_missing_head_as_zero(self):
        for mode in ['one_head_flat', 'opposite_head', 'extra_aim', 'missing_qwen']:
            owner = Controller(cfg=Config.from_mapping({'enabled': True}), heads=module.HEADS[:2])
            for index in range(20):
                values = {name: {'d_loss': 2. if index < 10 else 1.,
                                 'g_loss': 1. if index < 10 else 2., 'g_proxy': None}
                          for name in owner.head_keys}
                if mode == 'one_head_flat':
                    for step in [1, 2]:values[f's{step}/qwen3vl_4b'] = {'d_loss': 1., 'g_loss': 1.}
                elif mode == 'opposite_head':
                    values['s1/qwen3vl_4b'] = {'d_loss': 1. if index < 10 else 2.,
                                              'g_loss': 2. if index < 10 else 1.}
                elif mode == 'extra_aim':values['s1/aimv2_huge'] = {'d_loss': 1., 'g_loss': 1.}
                else:del values['s2/qwen3vl_4b']
                decision = owner.observe(index + 51, values, generator_active=True, valid=True)
            self.assertFalse(decision['changed'], mode)
            self.assertEqual(owner.current_lr, 1e-5)
            if mode in ['extra_aim', 'missing_qwen']:
                self.assertEqual(decision['reason'], 'invalid_observation')

    def test_disabled_default_and_explicit_configuration_validation(self):
        owner = Controller()
        decisions = sequence(owner)
        self.assertTrue(all(d['action'] == 'hold' and d['reason'] == 'disabled' for d in decisions))
        self.assertEqual(owner.current_lr, 1e-5)
        restored = Controller(); restored.load_state_dict(owner.state_dict())
        self.assertEqual(restored.state_dict(), owner.state_dict())
        invalid = [{'typo': 1}, {'enabled': 1}, {'window_size': 1}, {'cooldown_steps': -1},
                   {'min_lr': 0}, {'max_lr': float('inf')}, {'min_relative_change': 1},
                   {'min_abs_change': float('nan')}, {'decrease_factor': 1}, {'increase_factor': 1},
                   {'volatility_min_jumps': 10}, {'volatility_ratio': 1}]
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(ValueError): Config.from_mapping(changes)
        with self.assertRaises(ValueError): Config.from_mapping([])

    def test_D_down_actual_G_up_decreases_shared_rate_only_after20_G_points(self):
        owner = controller(); decisions = sequence(owner)
        self.assertTrue(all(not d['changed'] for d in decisions[:-1]))
        last = decisions[-1]
        self.assertEqual(last['action'], 'decrease'); self.assertEqual(last['old_lr'], 1e-5)
        self.assertEqual(last['new_lr'], 5e-6); self.assertEqual(last['effective_from_step'], 71)
        self.assertEqual(last['statistics']['old_window_steps'], list(range(51, 61)))
        self.assertEqual(last['statistics']['new_window_steps'], list(range(61, 71)))
        self.assertEqual(owner.state_dict()['history'], [])

    def test_D_up_actual_G_down_increases_without_proxy_substitution(self):
        owner = controller()
        last = sequence(owner, old=(1., 2.), new=(2., 1.), proxy=False)[-1]
        self.assertEqual(last['action'], 'increase'); self.assertEqual(last['new_lr'], 1.25e-5)
        self.assertEqual(last['reason'], 'D_rising_actual_G_falling')

    def test_warmup_zero_G_not_used_and_actual_history_starts_at51(self):
        owner = controller()
        for step in range(1, 51):
            decision = owner.observe(step, metrics(2. if step <= 10 else 1., 0., 3. if step > 10 else 1.),
                                     generator_active=False, valid=True)
            self.assertFalse(decision['changed'])
        self.assertTrue(all(row['metrics'][name]['g_loss'] is None
                            for row in owner.state_dict()['history'] for name in KEYS))
        decisions = sequence(owner, old=(1., 1.), new=(.5, 2.), start=51)
        self.assertTrue(all(not d['changed'] for d in decisions[:-1]))
        self.assertEqual(decisions[-1]['action'], 'decrease')

    def test_sustained_D_volatility_alone_can_decrease_during_warmup(self):
        owner = controller()
        for step in range(1, 11):
            values = metrics()
            for name in ('s1/dinov3_l', 's1/qwen3vl_4b'):
                values[name]['d_loss'] = .5 if step % 2 else 3.
            decision = owner.observe(step, values, generator_active=False, valid=True)
        self.assertEqual(decision['action'], 'decrease')
        self.assertEqual(decision['reason'], 'large_volatility')

    def test_actual_G_volatility_can_decrease_but_proxy_only_volatility_cannot(self):
        for signal in ('g_loss', 'g_proxy'):
            owner = controller()
            for index in range(10):
                values = metrics()
                for name in ('s2/dinov3_l', 's2/qwen3vl_4b'):
                    values[name][signal] = .5 if index % 2 else 3.
                decision = owner.observe(51+index, values, generator_active=True, valid=True)
            self.assertEqual(decision['action'], 'decrease' if signal == 'g_loss' else 'hold')

    def test_single_outlier_does_not_trigger_large_volatility(self):
        owner = controller()
        for step in range(1, 11):
            decision = owner.observe(step, metrics(100. if step == 10 else 1., 0.),
                                     generator_active=False, valid=True)
        self.assertEqual(decision['action'], 'hold')

    def test_majorities_per_step_and_single_head_never_dominates(self):
        for support in (1, 2):
            owner = controller()
            def change(index, values):
                for step in (1, 2):
                    for head in module.HEADS[support:]: values[f's{step}/{head}'] = metrics()[KEYS[0]].copy()
            last = sequence(owner, modify=change)[-1]
            self.assertEqual(last['action'], 'hold' if support == 1 else 'decrease')

    def test_reverse_head_or_generation_step_prevents_trend_increase(self):
        for conflicting_names in (('s1/aimv2_huge',), tuple(n for n in KEYS if n.startswith('s2/'))):
            owner = controller()
            def opposite(index, values):
                for name in conflicting_names:
                    values[name] = {'d_loss': 2. if index < 10 else 1.,
                                    'g_loss': 1. if index < 10 else 2., 'g_proxy': None}
            last = sequence(owner, old=(1., 2.), new=(2., 1.), modify=opposite)[-1]
            self.assertEqual(last['action'], 'hold'); self.assertEqual(last['reason'], 'opposing_head_trends')

    def test_proxy_opposite_audits_hold_and_proxy_trend_never_replaces_flat_G(self):
        owner = controller()
        def opposite(index, values):
            for value in values.values(): value['g_proxy'] = 2. if index < 10 else 1.
        last = sequence(owner, modify=opposite)[-1]
        self.assertEqual(last['reason'], 'actual_G_proxy_conflict'); self.assertEqual(last['action'], 'hold')
        owner = controller()
        def proxy_only(index, values):
            for value in values.values(): value['g_proxy'] = 1. if index < 10 else 2.
        self.assertEqual(sequence(owner, old=(2., 1.), new=(1., 1.), modify=proxy_only)[-1]['action'], 'hold')

    def test_absolute_and_relative_thresholds_both_required_and_exact20percent_boundary(self):
        examples = [((.10, .2), (.075, .4), 'hold'), ((2., 1.), (1.7, 2.), 'hold'),
                    ((1., 1.), (.81, 1.2), 'hold'), ((1., 1.), (.8, 1.2), 'decrease')]
        for old, new, action in examples:
            with self.subTest(old=old, new=new):
                self.assertEqual(sequence(controller(), old=old, new=new)[-1]['action'], action)

    def test_large_volatility_has_priority_over_other_heads_upward_trend(self):
        owner = controller()
        decisions = []
        for index in range(20):
            values = metrics(1. if index < 10 else 2., 2. if index < 10 else 1.)
            if index >= 16:
                for name in ('s1/dinov3_l', 's1/qwen3vl_4b'):
                    values[name]['d_loss'] = (2.1, .5, 3., .5)[index-16]
            decision = owner.observe(51+index, values, generator_active=True, valid=True)
            decisions.append(decision)
        self.assertTrue(all(not d['changed'] for d in decisions[:-1]))
        self.assertEqual(decision['action'], 'decrease'); self.assertEqual(decision['reason'], 'large_volatility')

    def test_cooldown_blocks20_valid_observations_even_with_volatility(self):
        owner = controller(); sequence(owner)
        for offset in range(1, 21):
            values = metrics(.5 if offset % 2 else 3., 1.)
            decision = owner.observe(70+offset, values, generator_active=True, valid=True)
            self.assertEqual(decision['reason'], 'cooldown'); self.assertEqual(decision['new_lr'], 5e-6)
        self.assertEqual(owner.cooldown_remaining, 0)
        decision = owner.observe(91, metrics(.5, 1.), generator_active=True, valid=True)
        self.assertEqual(decision['reason'], 'large_volatility'); self.assertEqual(decision['new_lr'], 2.5e-6)

    def test_invalid_skip_or_NaN_clears_contiguous_history_and_does_not_use_cooldown(self):
        for field in ('d_loss', 'g_loss', 'g_proxy', 'caller_skip', 'missing_head'):
            owner = controller()
            for step in range(51, 61): owner.observe(step, metrics(), generator_active=True, valid=True)
            values = metrics(1., 2.)
            if field in ('d_loss', 'g_loss', 'g_proxy'): values[KEYS[0]][field] = float('nan')
            if field == 'missing_head': del values[KEYS[0]]
            decision = owner.observe(61, values, generator_active=True, valid=field != 'caller_skip')
            self.assertEqual(decision['reason'], 'invalid_observation'); self.assertEqual(owner.state_dict()['history'], [])
            json.dumps(decision, allow_nan=False)
            for step in range(62, 72): decision = owner.observe(step, metrics(1., 2.), generator_active=True, valid=True)
            self.assertEqual(decision['action'], 'hold')
        owner = controller(); sequence(owner)
        owner.observe(71, None, generator_active=True, valid=False)
        self.assertEqual(owner.cooldown_remaining, 20)

    def test_missing_active_G_and_bool_or_negative_losses_are_invalid(self):
        for value in (None, True, -1., float('inf')):
            values = metrics(); values[KEYS[0]]['g_loss'] = value
            self.assertEqual(controller().observe(51, values, generator_active=True, valid=True)['reason'], 'invalid_observation')

    def test_duplicate_or_backward_step_never_repeats_action_or_mutates_state(self):
        owner = controller(); sequence(owner); before = owner.state_dict()
        for step in (70, 69):
            self.assertEqual(owner.observe(step, metrics(), generator_active=True, valid=True)['reason'], 'nonmonotonic_step')
            self.assertEqual(owner.state_dict(), before)
        with self.assertRaises(ValueError): owner.observe(0, metrics(), generator_active=True, valid=True)

    def test_gap_discards_old_half_window(self):
        owner = controller()
        for step in range(51, 61): owner.observe(step, metrics(), generator_active=True, valid=True)
        for step in range(62, 72): decision = owner.observe(step, metrics(1., 2.), generator_active=True, valid=True)
        self.assertEqual(decision['action'], 'hold')
        self.assertEqual(owner.state_dict()['history'][0]['completed_step'], 62)

    def test_rate_clips_and_bound_hold_does_not_count_as_change(self):
        for overrides, old, new, expected, bound in [
            ({'max_lr':1.1e-5}, (1.,2.), (2.,1.), 1.1e-5, 'upper_bound'),
            ({'min_lr':7.5e-6}, (2.,1.), (1.,2.), 7.5e-6, 'lower_bound')]:
            owner = controller(cooldown_steps=0, **overrides)
            first = sequence(owner, old=old, new=new)[-1]
            self.assertTrue(first['clipped']); self.assertEqual(first['new_lr'], expected)
            second = sequence(owner, old=old, new=new, start=71)[-1]
            self.assertEqual(second['reason'], bound); self.assertFalse(second['changed'])
            self.assertEqual(owner.action_sequence, 1)

    def test_finite_huge_losses_never_make_infinite_statistics(self):
        owner = controller()
        for step in range(51, 71): decision = owner.observe(step, metrics(1e308, 1e308), generator_active=True, valid=True)
        self.assertEqual(decision['action'], 'hold'); json.dumps(decision, allow_nan=False)

    def test_input_and_exported_state_have_no_aliases(self):
        owner = controller(); values = metrics(); before = copy.deepcopy(values)
        owner.observe(51, values, generator_active=True, valid=True)
        self.assertEqual(values, before)
        values[KEYS[0]]['d_loss'] = 999
        saved = owner.state_dict(); saved['history'][0]['metrics'][KEYS[0]]['d_loss'] = 777
        self.assertEqual(owner.state_dict()['history'][0]['metrics'][KEYS[0]]['d_loss'], 2.)

    def test_checkpoint_restores_same_future_decisions_and_action_timing(self):
        original = controller(); sequence(original)
        for step in range(71, 76): original.observe(step, metrics(), generator_active=True, valid=True)
        restored = controller(); restored.load_state_dict(json.loads(json.dumps(original.state_dict())))
        for step in range(76, 126):
            values = metrics(.5 if step % 2 else 3., 1.)
            left = original.observe(step, values, generator_active=True, valid=True)
            right = restored.observe(step, values, generator_active=True, valid=True)
            self.assertEqual(left, right); self.assertEqual(original.state_dict(), restored.state_dict())

    def test_state_load_strict_validation_and_atomic_failure(self):
        owner = controller()
        for step in range(51, 63): owner.observe(step, metrics(), generator_active=True, valid=True)
        good = owner.state_dict()
        cases = []
        for key, value in [('version',True), ('current_lr',float('nan')), ('current_lr',1.),
                           ('valid_observations',1000), ('invalid_observations',-1),
                           ('cooldown_remaining',21), ('action_sequence',1), ('generator_active_mode',None),
                           ('policy_hash','wrong'), ('head_keys',list(reversed(KEYS)))]:
            bad = copy.deepcopy(good); bad[key] = value; cases.append(bad)
        bad = copy.deepcopy(good); bad['config']['enabled'] = 1; cases.append(bad)
        bad = copy.deepcopy(good); bad['history'][0]['metrics'][KEYS[0]]['d_loss'] = float('inf'); cases.append(bad)
        bad = copy.deepcopy(good); bad['history'][-1]['completed_step'] -= 1; cases.append(bad)
        bad = copy.deepcopy(good); bad['history'][0]['generator_active'] = False; cases.append(bad)
        for bad in cases:
            target = controller(); before = target.state_dict()
            with self.assertRaises(ValueError): target.load_state_dict(bad)
            self.assertEqual(target.state_dict(), before)
        with self.assertRaises(ValueError): controller(min_abs_change=.1).load_state_dict(good)
        sequence(owner, start=63)
        changed = owner.state_dict()
        if changed['last_action'] is not None:
            bad = copy.deepcopy(changed); bad['last_action']['reason'] = 'D_rising_actual_G_falling'
            with self.assertRaises(ValueError): controller().load_state_dict(bad)


if __name__ == '__main__':
    unittest.main()
