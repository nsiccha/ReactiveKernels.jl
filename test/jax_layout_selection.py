"""Deterministic callback, ranking, and lifetime checks for an emitted selector."""
import gc
import weakref
from unittest.mock import patch

import numpy as np


def check_selection(module, data):
    candidates = [(1, False), (3, False), (3, True)]

    class Timer:
        def __init__(self, fail=None, failure=MemoryError):
            self.now = 0
            self.events = []
            self.refs = []
            self.builds = []
            self.fail = fail
            self.failure = failure

        def build(self, chosen, count=None):
            layout = chosen['_rk_plate_layout']['totals_cell']
            k = 0 if layout is None else (2 if layout.sequential else 1)
            # A new width must not retain any timing callbacks from older widths.
            gc.collect()
            assert all(ref() is None for width, ref in self.refs if width != count)
            self.builds.append((k, count))
            self.now += 11
            timer = self

            class Step:
                calls = 0

                def __call__(self):
                    self.calls += 1
                    timer.events.append((k, count, self.calls))
                    return self

                def block_until_ready(self):
                    if timer.fail is not None and timer.fail(k, count, self.calls):
                        raise timer.failure('timed step failed')
                    # Disjoint pilot ranges deliberately favor candidate 0;
                    # the confirmation medians favor candidate 1. Only waiting
                    # for the result advances this clock.
                    pilot = ([1, 2, 3], [5, 6, 7], [8, 9, 10])
                    latency = pilot[k][self.calls - 2] if 2 <= self.calls <= 4 else (9, 4, 6)[k]
                    timer.now += latency * (count or 1)
                    return self

            step = Step()
            self.refs.append((count, weakref.ref(step)))
            return step

        def run(self, **kwargs):
            with patch('time.perf_counter', lambda: self.now):
                return module.select_plate_layout(data, self.build, candidates=iter(candidates), **kwargs)

        def released(self):
            gc.collect()
            assert all(ref() is None for _, ref in self.refs)

    timer = Timer()
    best, report = timer.run(chains=iter((4, 2)))
    assert report[0]['groups'] == 3 and not report[0]['sequential']
    assert not best['_rk_plate_layout']['totals_cell'].sequential
    assert len(report) == 6 and timer.builds == [(k, n) for n in (2, 4) for k in range(3)]
    for row in report:
        k = 0 if row['groups'] == 1 else (2 if row['sequential'] else 1)
        n = row['chains']
        assert row['samples_seconds'] == [v * n for v in ([1, 2, 3], [5, 6, 7], [8, 9, 10])[k]]
        assert row['median_seconds'] == (2, 6, 9)[k] * n
        assert row['confirmation_samples_seconds'] == [(9, 4, 6)[k] * n] * 12
        assert row['seconds'] == row['confirmation_median_seconds'] == (9, 4, 6)[k] * n
        assert row['score_method'] == 'confirmation_median_per_chain'
        assert row['calibration_seconds'] == row['setup_seconds'] + sum(row['samples_seconds']) + sum(row['confirmation_samples_seconds'])
    for n in (2, 4):
        confirmation = [(k, c) for k, width, c in timer.events if width == n and c > 4]
        assert confirmation == [(k, i + 5) for i in range(12)
                                for k in list(range(3))[i % 3:] + list(range(3))[:i % 3]]
    timer.released()

    timer = Timer()
    _, report = timer.run(chains=(2, 4), calls=lambda n: 10 / n ** 2)
    assert report[0]['chains'] == 4 and report[0]['groups'] == 3
    assert [r['fit_seconds'] for r in report] == sorted(r['fit_seconds'] for r in report)
    assert all(r['fit_seconds'] == r['seconds'] * 10 / r['chains'] ** 2 for r in report)
    assert all(r['score_method'] == 'confirmation_median_projected_calls' for r in report)
    timer.released()

    # Explicit opt-out keeps pilot-minimum ranking and original callback order.
    timer = Timer()
    _, report = timer.run(chains=(2, 4), confirmation_rounds=0)
    assert report[0]['groups'] == 1 and report[0]['seconds'] == 2
    assert timer.builds == [(k, n) for k in range(3) for n in (2, 4)]
    assert all(r['confirmation_samples_seconds'] == [] for r in report)
    assert all(r['score_method'] == 'pilot_minimum_per_chain' for r in report)
    timer.released()

    # Single effective layout: no extra calls, including the fit objective.
    timer = Timer()
    with patch('time.perf_counter', lambda: timer.now):
        _, report = module.select_plate_layout(data, timer.build, chains=(2, 4),
                                               candidates=[(1, False)], calls=lambda n: 5)
    assert len(timer.events) == 8
    assert report[0]['seconds'] == 4 and report[0]['fit_seconds'] == 20
    assert all(r['score_method'] == 'pilot_median_projected_calls' for r in report)
    timer.released()

    # No chain axis still confirms competing layouts.
    timer = Timer()
    _, report = timer.run(confirmation_rounds=3)
    assert report[0]['chains'] is None and report[0]['seconds'] == 4
    assert all(len(r['confirmation_samples_seconds']) == 3 for r in report)
    timer.released()

    # Confirmation OOM keeps partial evidence, excludes the row and skips its
    # later widths. The remaining width still receives a full confirmation.
    timer = Timer(fail=lambda k, n, c: k != 1 and c == 6)
    _, report = timer.run(chains=(2, 4))
    failed = [r for r in report if r['error']]
    assert len(failed) == 2 and all(r['chains'] == 2 for r in failed)
    assert all(len(r['confirmation_samples_seconds']) == 1 for r in failed)
    assert all(r['confirmation_median_seconds'] is None and np.isinf(r['seconds']) for r in failed)
    assert all(len(r['confirmation_samples_seconds']) == 12 for r in report if not r['error'])
    assert timer.builds[-1] == (1, 4) and len(timer.builds) == 4
    timer.released()

    for fail in (lambda k, n, c: c == 2, lambda k, n, c: c == 5):
        timer = Timer(fail=fail)
        with np.testing.assert_raises_regex(RuntimeError, 'no candidate ran'):
            timer.run(chains=(2, 4))
        timer.released()
        assert len(timer.builds) == 3

    timer = Timer(fail=lambda k, n, c: c == 5, failure=ValueError)
    with np.testing.assert_raises_regex(ValueError, 'timed step failed'):
        timer.run()
    timer.released()

    with np.testing.assert_raises_regex(ValueError, 'confirmation_rounds'):
        Timer().run(confirmation_rounds=-1)
    with np.testing.assert_raises(TypeError):
        Timer().run(confirmation_rounds=1.5)
    print('Public selector deterministic checks passed', flush=True)
