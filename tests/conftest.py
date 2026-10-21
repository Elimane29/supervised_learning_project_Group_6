import pandas as pd
import pytest

from api.app import Encounter


@pytest.fixture
def api_like_rows() -> pd.DataFrame:
    """65 default encounters as the API would build them (all raw columns present)."""
    return pd.DataFrame([Encounter().model_dump() for _ in range(65)])
