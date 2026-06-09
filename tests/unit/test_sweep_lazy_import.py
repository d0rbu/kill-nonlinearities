"""Guard: importing the sweep module must not import wandb (spec §7 [R19], §8).

`wandb` is heavy and has a network side; the pure helpers (`build_sweep_config`,
`config_from_wandb`) must be usable without it. We assert in a FRESH subprocess so the result
is independent of whatever the parent pytest process has already imported.
"""

import subprocess
import sys


def test_importing_sweep_does_not_import_wandb() -> None:
    """`import ...experiments.sweep` leaves wandb absent from sys.modules ([R19])."""
    code = (
        "import sys\n"
        "import kill_nonlinearities.experiments.sweep\n"
        "assert 'wandb' not in sys.modules, sorted(m for m in sys.modules if 'wandb' in m)\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
