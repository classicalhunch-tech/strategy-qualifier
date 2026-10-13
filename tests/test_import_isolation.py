"""Importing research modules must never shadow the standard library."""
import subprocess
import sys


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


def test_session_module_does_not_shadow_stdlib_statistics():
    result = _run(
        "import sys\n"
        "import strategy_qualifier.research.session_vs_outside\n"
        "import strategy_qualifier.research.exit_study\n"
        "import statistics\n"
        "assert hasattr(statistics, 'NormalDist')\n"
        "assert 'strategy_qualifier' not in statistics.__file__\n"
    )
    assert result.returncode == 0, result.stderr


def test_importing_research_modules_does_not_modify_sys_path():
    result = _run(
        "import sys\n"
        "before = list(sys.path)\n"
        "import strategy_qualifier.automatic_evidence\n"
        "import strategy_qualifier.research.z_vs_room\n"
        "import strategy_qualifier.research.exit_study\n"
        "assert sys.path == before, [p for p in sys.path if p not in before]\n"
    )
    assert result.returncode == 0, result.stderr
