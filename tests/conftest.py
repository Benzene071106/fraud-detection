"""Shared fixtures. Tests use a throw-away SQLite database and the real models in models/."""
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_DB = Path(tempfile.mkdtemp()) / "test_soc.db"
os.environ["DATABASE_URL"] = f"sqlite:///{_DB.as_posix()}"     # must be set before api.main is imported
for var in ("SLACK_WEBHOOK_URL", "SMTP_HOST", "API_KEY"):          # no real notifications / auth in tests
    os.environ.pop(var, None)

MODEL = ROOT / "models" / "xgb_v1_nofp_nohdr_clean_r1500.json"
needs_models = pytest.mark.skipif(not MODEL.exists(), reason="model files not present (not in git)")


@pytest.fixture(scope="session")
def client():
    if not MODEL.exists():
        pytest.skip("model files not present")
    from fastapi.testclient import TestClient
    import api.main as main
    return TestClient(main.app)


@pytest.fixture(scope="session")
def detector(client):
    import api.main as main
    return main.detector


@pytest.fixture(scope="session")
def sample_flows():
    return pd.read_csv(ROOT / "dashboard" / "sample_flows.csv")
