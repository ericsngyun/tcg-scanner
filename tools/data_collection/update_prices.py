#!/usr/bin/env python3
"""
Daily Price Update Script

Fetches latest prices from tcgcsv.com and updates local data.
Designed to be run via Windows Task Scheduler daily at 2 PM.

Usage:
    python update_prices.py
    python update_prices.py --config ml/data/tcgcsv_config.yaml
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from fetch_tcgcsv import TCGCSVFetcher, Product

# Configure logging
LOG_DIR = Path(__file__).parent.parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "price_updates.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def update_prices(config_path: str, output_dir: str) -> dict:
    """
    Fetch and save latest prices.

    Returns:
        dict with summary of the update
    """
    start_time = datetime.now()
    logger.info(f"Starting price update at {start_time.isoformat()}")

    output_base = Path(output_dir)

    try:
        fetcher = TCGCSVFetcher(config_path, output_base)

        # Fetch all prices
        prices = fetcher.fetch_all_prices()
        logger.info(f"Fetched prices for {len(prices)} products")

        # Save to history
        history_file = fetcher.save_price_history(prices)
        logger.info(f"Saved price history to {history_file}")

        # Update cached products if they exist
        cache_file = fetcher.raw_dir / "all_products.json"
        products_updated = 0

        if cache_file.exists():
            import json
            with open(cache_file) as f:
                products = [Product(**p) for p in json.load(f)]

            products = fetcher.update_products_with_prices(products, prices)

            with open(cache_file, "w") as f:
                json.dump([p.to_dict() for p in products], f, indent=2)

            products_updated = len(products)
            logger.info(f"Updated {products_updated} products with new prices")

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        summary = {
            "status": "success",
            "started_at": start_time.isoformat(),
            "ended_at": end_time.isoformat(),
            "duration_seconds": duration,
            "prices_fetched": len(prices),
            "products_updated": products_updated,
            "history_file": str(history_file),
        }

        logger.info(f"Price update completed in {duration:.1f} seconds")
        return summary

    except Exception as e:
        logger.error(f"Price update failed: {e}", exc_info=True)
        return {
            "status": "error",
            "started_at": start_time.isoformat(),
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="Update TCG prices from tcgcsv.com")
    parser.add_argument(
        "--config",
        type=str,
        default="ml/data/tcgcsv_config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ml/data",
        help="Output base directory",
    )
    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = Path(__file__).parent.parent.parent
    config_path = project_root / args.config
    output_dir = project_root / args.output

    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    result = update_prices(str(config_path), str(output_dir))

    if result["status"] == "error":
        sys.exit(1)

    print(f"\nPrice update summary:")
    print(f"  Prices fetched: {result['prices_fetched']}")
    print(f"  Products updated: {result['products_updated']}")
    print(f"  Duration: {result['duration_seconds']:.1f}s")


if __name__ == "__main__":
    main()
