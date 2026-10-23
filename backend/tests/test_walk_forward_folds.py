from __future__ import annotations

from datetime import date, timedelta

import pytest

from backend.app.research.walk_forward import (
    FoldMode,
    folds_have_overlapping_evaluation_windows,
    generate_walk_forward_folds,
)

_SESSIONS = [date(2024, 1, 1) + timedelta(days=i) for i in range(40)]


class TestInputValidation:
    def test_rejects_zero_train_window(self):
        with pytest.raises(ValueError, match="train_window_sessions"):
            generate_walk_forward_folds(_SESSIONS, 0, 5, 5, 5, FoldMode.ROLLING)

    def test_rejects_zero_validation_window(self):
        with pytest.raises(ValueError, match="train_window_sessions"):
            generate_walk_forward_folds(_SESSIONS, 5, 0, 5, 5, FoldMode.ROLLING)

    def test_rejects_zero_test_window(self):
        with pytest.raises(ValueError, match="train_window_sessions"):
            generate_walk_forward_folds(_SESSIONS, 5, 5, 0, 5, FoldMode.ROLLING)

    def test_rejects_zero_step(self):
        with pytest.raises(ValueError, match="step_sessions"):
            generate_walk_forward_folds(_SESSIONS, 5, 5, 5, 0, FoldMode.ROLLING)

    def test_rejects_empty_session_dates(self):
        with pytest.raises(ValueError, match="must not be empty"):
            generate_walk_forward_folds([], 5, 5, 5, 5, FoldMode.ROLLING)

    def test_rejects_non_ascending_session_dates(self):
        bad = [_SESSIONS[0], _SESSIONS[2], _SESSIONS[1]]
        with pytest.raises(ValueError, match="ascending"):
            generate_walk_forward_folds(bad, 1, 1, 1, 1, FoldMode.ROLLING)

    def test_rejects_duplicate_session_dates(self):
        bad = [_SESSIONS[0], _SESSIONS[0], _SESSIONS[1]]
        with pytest.raises(ValueError, match="ascending"):
            generate_walk_forward_folds(bad, 1, 1, 1, 1, FoldMode.ROLLING)


class TestRollingWindows:
    def test_first_fold_boundaries_are_exact(self):
        folds = generate_walk_forward_folds(
            _SESSIONS, train_window_sessions=10, validation_window_sessions=5,
            test_window_sessions=5, step_sessions=5, mode=FoldMode.ROLLING,
        )
        first = folds[0]
        assert first.fold_index == 0
        assert first.train_start == _SESSIONS[0]
        assert first.train_end == _SESSIONS[9]
        assert first.validation_start == _SESSIONS[10]
        assert first.validation_end == _SESSIONS[14]
        assert first.test_start == _SESSIONS[15]
        assert first.test_end == _SESSIONS[19]

    def test_train_window_width_stays_fixed_across_folds(self):
        folds = generate_walk_forward_folds(
            _SESSIONS, train_window_sessions=10, validation_window_sessions=5,
            test_window_sessions=5, step_sessions=5, mode=FoldMode.ROLLING,
        )
        for fold in folds:
            width = (fold.train_end - fold.train_start).days + 1
            assert width == 10

    def test_second_fold_train_start_advances_by_step(self):
        folds = generate_walk_forward_folds(
            _SESSIONS, train_window_sessions=10, validation_window_sessions=5,
            test_window_sessions=5, step_sessions=5, mode=FoldMode.ROLLING,
        )
        assert folds[1].train_start == _SESSIONS[5]
        assert folds[1].train_end == _SESSIONS[14]


class TestExpandingWindows:
    def test_train_start_is_always_the_first_session(self):
        folds = generate_walk_forward_folds(
            _SESSIONS, train_window_sessions=10, validation_window_sessions=5,
            test_window_sessions=5, step_sessions=5, mode=FoldMode.EXPANDING,
        )
        for fold in folds:
            assert fold.train_start == _SESSIONS[0]

    def test_train_end_grows_by_step_each_fold(self):
        folds = generate_walk_forward_folds(
            _SESSIONS, train_window_sessions=10, validation_window_sessions=5,
            test_window_sessions=5, step_sessions=5, mode=FoldMode.EXPANDING,
        )
        assert folds[0].train_end == _SESSIONS[9]
        assert folds[1].train_end == _SESSIONS[14]
        if len(folds) > 2:
            assert folds[2].train_end == _SESSIONS[19]

    def test_expanding_produces_wider_train_windows_over_time(self):
        folds = generate_walk_forward_folds(
            _SESSIONS, train_window_sessions=10, validation_window_sessions=5,
            test_window_sessions=5, step_sessions=5, mode=FoldMode.EXPANDING,
        )
        widths = [(f.train_end - f.train_start).days for f in folds]
        assert widths == sorted(widths)
        assert len(set(widths)) > 1


class TestDeterministicRegeneration:
    def test_calling_twice_with_identical_inputs_gives_identical_output(self):
        kwargs = dict(
            session_dates=_SESSIONS, train_window_sessions=8, validation_window_sessions=4,
            test_window_sessions=4, step_sessions=4, mode=FoldMode.ROLLING,
        )
        first = generate_walk_forward_folds(**kwargs)
        second = generate_walk_forward_folds(**kwargs)
        assert first == second

    def test_calling_twice_with_identical_inputs_gives_identical_output_expanding(self):
        kwargs = dict(
            session_dates=_SESSIONS, train_window_sessions=8, validation_window_sessions=4,
            test_window_sessions=4, step_sessions=4, mode=FoldMode.EXPANDING,
        )
        first = generate_walk_forward_folds(**kwargs)
        second = generate_walk_forward_folds(**kwargs)
        assert first == second


class TestMonotonicChronologyAndNoIntraFoldOverlap:
    @pytest.mark.parametrize("mode", [FoldMode.ROLLING, FoldMode.EXPANDING])
    def test_every_fold_is_internally_chronological(self, mode):
        folds = generate_walk_forward_folds(
            _SESSIONS, train_window_sessions=8, validation_window_sessions=4,
            test_window_sessions=4, step_sessions=4, mode=mode,
        )
        assert folds
        for fold in folds:
            assert fold.train_start <= fold.train_end
            assert fold.train_end < fold.validation_start
            assert fold.validation_start <= fold.validation_end
            assert fold.validation_end < fold.test_start
            assert fold.test_start <= fold.test_end


class TestCorrectStepAdvancement:
    def test_step_of_one_produces_the_maximum_number_of_folds(self):
        wide_step = generate_walk_forward_folds(
            _SESSIONS, train_window_sessions=8, validation_window_sessions=4,
            test_window_sessions=4, step_sessions=4, mode=FoldMode.ROLLING,
        )
        narrow_step = generate_walk_forward_folds(
            _SESSIONS, train_window_sessions=8, validation_window_sessions=4,
            test_window_sessions=4, step_sessions=1, mode=FoldMode.ROLLING,
        )
        assert len(narrow_step) > len(wide_step)


class TestBoundaryConditions:
    def test_exact_fit_produces_exactly_one_fold(self):
        exact = _SESSIONS[:16]
        folds = generate_walk_forward_folds(
            exact, train_window_sessions=8, validation_window_sessions=4,
            test_window_sessions=4, step_sessions=4, mode=FoldMode.ROLLING,
        )
        assert len(folds) == 1
        assert folds[0].test_end == exact[-1]

    def test_one_session_short_produces_zero_folds(self):
        one_short = _SESSIONS[:15]
        folds = generate_walk_forward_folds(
            one_short, train_window_sessions=8, validation_window_sessions=4,
            test_window_sessions=4, step_sessions=4, mode=FoldMode.ROLLING,
        )
        assert folds == []


class TestInsufficientHistoryBehavior:
    def test_far_too_few_sessions_returns_empty_list_not_an_error(self):
        tiny = _SESSIONS[:3]
        folds = generate_walk_forward_folds(
            tiny, train_window_sessions=10, validation_window_sessions=5,
            test_window_sessions=5, step_sessions=5, mode=FoldMode.ROLLING,
        )
        assert folds == []

    def test_single_session_returns_empty_list(self):
        folds = generate_walk_forward_folds(
            _SESSIONS[:1], train_window_sessions=1, validation_window_sessions=1,
            test_window_sessions=1, step_sessions=1, mode=FoldMode.ROLLING,
        )
        assert folds == []


class TestFoldOverlapDetection:
    def test_wide_step_produces_no_overlap(self):
        folds = generate_walk_forward_folds(
            _SESSIONS, train_window_sessions=8, validation_window_sessions=2,
            test_window_sessions=2, step_sessions=12, mode=FoldMode.ROLLING,
        )
        assert len(folds) >= 2
        assert folds_have_overlapping_evaluation_windows(folds) is False

    def test_narrow_step_produces_overlap(self):
        folds = generate_walk_forward_folds(
            _SESSIONS, train_window_sessions=8, validation_window_sessions=4,
            test_window_sessions=4, step_sessions=1, mode=FoldMode.ROLLING,
        )
        assert len(folds) >= 2
        assert folds_have_overlapping_evaluation_windows(folds) is True

    def test_single_fold_never_overlaps(self):
        folds = generate_walk_forward_folds(
            _SESSIONS[:16], train_window_sessions=8, validation_window_sessions=4,
            test_window_sessions=4, step_sessions=4, mode=FoldMode.ROLLING,
        )
        assert len(folds) == 1
        assert folds_have_overlapping_evaluation_windows(folds) is False

    def test_overlap_is_permitted_not_rejected(self):
        folds = generate_walk_forward_folds(
            _SESSIONS, train_window_sessions=8, validation_window_sessions=4,
            test_window_sessions=4, step_sessions=1, mode=FoldMode.ROLLING,
        )
        assert len(folds) > 0
