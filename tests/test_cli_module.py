import os
import subprocess
import sys


def test_python_module_help_runs() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"src{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)
    result = subprocess.run(
        [sys.executable, "-m", "lob_market_making_agents.experiments.run", "--help"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.returncode == 0
    assert "Run LOB market-making experiments." in result.stdout
