# sample_app

A tiny project used to demo and test DevPilot. It deliberately contains:

- an authentication flow (`auth/`, `web/views.py`)
- a module-level import cycle (`utils/cycle_a.py` <-> `utils/cycle_b.py`)
- a cycle that only exists through a function-level import (`utils/lazy_c.py` <-> `utils/lazy_d.py`)
- direct and mutual recursion (`utils/mathx.py`)
- one intentionally failing test (`tests/test_known_bug.py`)
