from __future__ import annotations

import json
import logging
from pathlib import Path

from backend.app.data.sources.nse_corporate_actions import fetch_corporate_actions

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RAW_CORPORATE_ACTIONS_DIR = Path("data/raw/corporate_actions")
SAMPLE_SYMBOLS = ["RELIANCE", "KOTAKBANK", "BAJFINANCE", "HDFCBANK"]


def main() -> None:
    for symbol in SAMPLE_SYMBOLS:
        path = fetch_corporate_actions(symbol, RAW_CORPORATE_ACTIONS_DIR)
        if path is None:
            logger.info("%s: byte-identical snapshot already stored - no-op", symbol)
            continue
        meta = json.loads(path.with_suffix(path.suffix + ".meta.json").read_text())
        logger.info(
            "%s: %d records, announcement_date_observed=%s -> %s",
            symbol, meta["record_count"], meta["announcement_date_observed"], path,
        )


if __name__ == "__main__":
    main()
