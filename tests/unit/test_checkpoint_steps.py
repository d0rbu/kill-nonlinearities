"""Unit tests for ``checkpoint_steps`` (spec §4.5)."""

from kill_nonlinearities.training.schedule import checkpoint_steps


def test_includes_zero_and_final_step() -> None:
    result = checkpoint_steps(total_steps=10, steps_per_epoch=5, every_epochs=1)
    assert result[0] == 0
    assert result[-1] == 9  # total_steps - 1


def test_is_sorted_and_deduplicated() -> None:
    result = checkpoint_steps(total_steps=10, steps_per_epoch=5, every_epochs=1)
    assert result == sorted(set(result))


def test_hand_enumerated_every_epoch() -> None:
    # 2 epochs * 5 steps/epoch = 10 total steps (steps 0..9).
    # epoch multiples of 5: {0, 5}; plus final step 9; plus 0.
    result = checkpoint_steps(total_steps=10, steps_per_epoch=5, every_epochs=1)
    assert result == [0, 5, 9]


def test_hand_enumerated_every_two_epochs() -> None:
    # 4 epochs * 3 steps/epoch = 12 total steps (steps 0..11).
    # every_epochs=2 -> stride 6: multiples of 6 within range {0, 6};
    # plus final step 11; plus 0.
    result = checkpoint_steps(total_steps=12, steps_per_epoch=3, every_epochs=2)
    assert result == [0, 6, 11]


def test_final_step_deduplicated_when_it_lands_on_a_multiple() -> None:
    # 3 epochs * 4 steps/epoch = 12 (steps 0..11). multiples of 4 in range:
    # {0, 4, 8}; final step is 11; 8 is not 11 so all distinct.
    result = checkpoint_steps(total_steps=12, steps_per_epoch=4, every_epochs=1)
    assert result == [0, 4, 8, 11]


def test_single_step_run_collapses_to_zero() -> None:
    result = checkpoint_steps(total_steps=1, steps_per_epoch=1, every_epochs=1)
    assert result == [0]
