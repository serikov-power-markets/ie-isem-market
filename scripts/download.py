"""Download and cache all open data (ENTSO-E needs a token in ENTSOE_TOKEN or ~/.entsoe_token)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from isem import eirgrid, entsoe  # noqa: E402

raw = Path(__file__).resolve().parents[1] / "data" / "raw"
entsoe.download_all(raw)
eirgrid.fetch(raw)
