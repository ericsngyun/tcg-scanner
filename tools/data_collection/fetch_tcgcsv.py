#!/usr/bin/env python3
"""
TCGCSV Data Collection Script

Downloads card data, images, and prices from tcgcsv.com for ML training.
Designed to minimize API calls and respect rate limits.

Usage:
    python fetch_tcgcsv.py --config ml/data/tcgcsv_config.yaml
    python fetch_tcgcsv.py --config ml/data/tcgcsv_config.yaml --images-only
    python fetch_tcgcsv.py --config ml/data/tcgcsv_config.yaml --metadata-only
    python fetch_tcgcsv.py --config ml/data/tcgcsv_config.yaml --prices-only
"""

import argparse
import csv
import io
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import hashlib

import requests
import yaml
from tqdm import tqdm


@dataclass
class PriceData:
    """Represents price information for a product."""
    product_id: int
    low_price: Optional[float] = None
    mid_price: Optional[float] = None
    high_price: Optional[float] = None
    market_price: Optional[float] = None
    direct_low_price: Optional[float] = None
    fetched_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "product_id": self.product_id,
            "low_price": self.low_price,
            "mid_price": self.mid_price,
            "high_price": self.high_price,
            "market_price": self.market_price,
            "direct_low_price": self.direct_low_price,
            "fetched_at": self.fetched_at,
        }


@dataclass
class Product:
    """Represents a TCG product/card."""
    product_id: int
    name: str
    clean_name: str
    image_url: str
    category_id: int
    group_id: int
    group_name: str
    rarity: Optional[str] = None
    number: Optional[str] = None
    description: Optional[str] = None
    card_type: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    extended_data: Dict[str, str] = field(default_factory=dict)
    # Price data
    low_price: Optional[float] = None
    mid_price: Optional[float] = None
    high_price: Optional[float] = None
    market_price: Optional[float] = None
    direct_low_price: Optional[float] = None

    @classmethod
    def from_api_response(cls, data: dict, group_name: str) -> "Product":
        """Create Product from API response."""
        # Extract extended data
        extended = {}
        rarity = None
        number = None
        description = None
        card_type = None
        tags = []

        for item in data.get("extendedData", []):
            name = item.get("name", "")
            value = item.get("value", "")
            extended[name] = value

            if name == "Rarity":
                rarity = value
            elif name == "Number":
                number = value
            elif name == "Description":
                description = value
            elif name == "Card Type":
                card_type = value
            elif name == "Tag":
                tags = [t.strip() for t in value.split(";") if t.strip()]

        return cls(
            product_id=data["productId"],
            name=data["name"],
            clean_name=data.get("cleanName", data["name"]),
            image_url=data.get("imageUrl", ""),
            category_id=data["categoryId"],
            group_id=data["groupId"],
            group_name=group_name,
            rarity=rarity,
            number=number,
            description=description,
            card_type=card_type,
            tags=tags,
            extended_data=extended,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "product_id": self.product_id,
            "name": self.name,
            "clean_name": self.clean_name,
            "image_url": self.image_url,
            "category_id": self.category_id,
            "group_id": self.group_id,
            "group_name": self.group_name,
            "rarity": self.rarity,
            "number": self.number,
            "description": self.description,
            "card_type": self.card_type,
            "tags": self.tags,
            "extended_data": self.extended_data,
            "low_price": self.low_price,
            "mid_price": self.mid_price,
            "high_price": self.high_price,
            "market_price": self.market_price,
            "direct_low_price": self.direct_low_price,
        }


class TCGCSVFetcher:
    """Fetches and caches data from tcgcsv.com."""

    def __init__(self, config_path: str, output_base: Path):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.base_url = self.config["api"]["base_url"]
        self.cdn_url = self.config["api"]["cdn_url"]
        self.request_delay = self.config["api"]["request_delay_seconds"]
        self.image_delay = self.config["api"]["image_delay_seconds"]
        self.max_retries = self.config["api"]["max_retries"]
        self.retry_delay = self.config["api"]["retry_delay_seconds"]

        self.output_base = output_base
        self.raw_dir = output_base / self.config["output"]["raw_data_dir"]
        self.processed_dir = output_base / self.config["output"]["processed_dir"]
        self.images_dir = output_base / self.config["output"]["images_dir"]

        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "TCG-Scanner-DataCollector/1.0 (ML Training; Respectful)",
            "Accept": "application/json",
        })

    def _request_with_retry(self, url: str) -> Optional[requests.Response]:
        """Make request with retry logic."""
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                return response
            except requests.RequestException as e:
                print(f"  Attempt {attempt + 1}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
        return None

    def fetch_products(self, category_id: int, group_id: int) -> List[dict]:
        """Fetch products for a group from API."""
        url = f"{self.base_url}/{category_id}/{group_id}/products"
        print(f"  Fetching: {url}")

        response = self._request_with_retry(url)
        if response is None:
            return []

        data = response.json()
        if isinstance(data, dict) and "results" in data:
            return data["results"]
        elif isinstance(data, list):
            return data
        return []

    def fetch_products_and_prices_csv(self, category_id: int, group_id: int) -> List[dict]:
        """Fetch products with prices from CSV endpoint."""
        url = f"{self.base_url}/{category_id}/{group_id}/ProductsAndPrices.csv"
        print(f"  Fetching CSV: {url}")

        response = self._request_with_retry(url)
        if response is None:
            return []

        # Parse CSV
        reader = csv.DictReader(io.StringIO(response.text))
        return list(reader)

    def _parse_price(self, value: str) -> Optional[float]:
        """Parse price string to float, handling empty values."""
        if not value or value.strip() == "":
            return None
        try:
            return float(value)
        except ValueError:
            return None

    def fetch_prices(self, category_id: int, group_id: int) -> Dict[int, PriceData]:
        """Fetch current prices for all products in a group."""
        rows = self.fetch_products_and_prices_csv(category_id, group_id)
        prices = {}
        fetched_at = datetime.now().isoformat()

        for row in rows:
            try:
                product_id = int(row.get("productId", 0))
                if product_id:
                    prices[product_id] = PriceData(
                        product_id=product_id,
                        low_price=self._parse_price(row.get("lowPrice", "")),
                        mid_price=self._parse_price(row.get("midPrice", "")),
                        high_price=self._parse_price(row.get("highPrice", "")),
                        market_price=self._parse_price(row.get("marketPrice", "")),
                        direct_low_price=self._parse_price(row.get("directLowPrice", "")),
                        fetched_at=fetched_at,
                    )
            except Exception as e:
                print(f"  Error parsing price row: {e}")

        return prices

    def fetch_all_prices(self) -> Dict[int, PriceData]:
        """Fetch prices for all Riftbound products."""
        print("Fetching prices from tcgcsv.com...")
        all_prices = {}
        riftbound = self.config["riftbound"]
        category_id = riftbound["category_id"]

        for group in tqdm(riftbound["groups"], desc="Fetching prices"):
            group_id = group["id"]
            group_name = group["name"]

            prices = self.fetch_prices(category_id, group_id)
            print(f"  {group_name}: {len(prices)} prices")
            all_prices.update(prices)

            # Rate limiting
            time.sleep(self.request_delay)

        return all_prices

    def update_products_with_prices(
        self, products: List[Product], prices: Dict[int, PriceData]
    ) -> List[Product]:
        """Update product list with current prices."""
        for product in products:
            if product.product_id in prices:
                price_data = prices[product.product_id]
                product.low_price = price_data.low_price
                product.mid_price = price_data.mid_price
                product.high_price = price_data.high_price
                product.market_price = price_data.market_price
                product.direct_low_price = price_data.direct_low_price
        return products

    def save_price_history(self, prices: Dict[int, PriceData]) -> Path:
        """Save prices to history file with timestamp."""
        history_dir = self.processed_dir / "price_history"
        history_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_file = history_dir / f"prices_{timestamp}.json"

        price_data = {pid: p.to_dict() for pid, p in prices.items()}
        with open(history_file, "w") as f:
            json.dump(price_data, f, indent=2)

        # Also save as latest
        latest_file = self.processed_dir / "latest_prices.json"
        with open(latest_file, "w") as f:
            json.dump(price_data, f, indent=2)

        print(f"Prices saved to {history_file}")
        print(f"Latest prices updated at {latest_file}")
        return history_file

    def load_latest_prices(self) -> Optional[Dict[int, PriceData]]:
        """Load the most recent price data."""
        latest_file = self.processed_dir / "latest_prices.json"
        if not latest_file.exists():
            return None

        with open(latest_file) as f:
            data = json.load(f)

        return {
            int(pid): PriceData(**pdata)
            for pid, pdata in data.items()
        }

    def fetch_all_riftbound_products(self, force_refresh: bool = False) -> List[Product]:
        """Fetch all Riftbound products, using cache if available."""
        cache_file = self.raw_dir / "all_products.json"
        cache_meta = self.raw_dir / "fetch_metadata.json"

        # Check cache
        if not force_refresh and cache_file.exists():
            print(f"Loading cached data from {cache_file}")
            with open(cache_file) as f:
                data = json.load(f)
            return [Product(**p) for p in data]

        # Fetch fresh data
        print("Fetching fresh data from tcgcsv.com...")
        all_products = []
        riftbound = self.config["riftbound"]
        category_id = riftbound["category_id"]

        for group in tqdm(riftbound["groups"], desc="Fetching groups"):
            group_id = group["id"]
            group_name = group["name"]

            raw_products = self.fetch_products(category_id, group_id)
            print(f"  {group_name}: {len(raw_products)} products")

            # Save raw response
            raw_file = self.raw_dir / f"group_{group_id}_raw.json"
            with open(raw_file, "w") as f:
                json.dump(raw_products, f, indent=2)

            # Parse products
            for raw in raw_products:
                try:
                    product = Product.from_api_response(raw, group_name)
                    all_products.append(product)
                except Exception as e:
                    print(f"  Error parsing product: {e}")

            # Rate limiting
            time.sleep(self.request_delay)

        # Save cache
        with open(cache_file, "w") as f:
            json.dump([p.to_dict() for p in all_products], f, indent=2)

        # Save metadata
        with open(cache_meta, "w") as f:
            json.dump({
                "fetched_at": datetime.now().isoformat(),
                "total_products": len(all_products),
                "groups_fetched": len(riftbound["groups"]),
            }, f, indent=2)

        print(f"\nTotal products fetched: {len(all_products)}")
        return all_products

    def filter_cards_only(self, products: List[Product]) -> List[Product]:
        """Filter to only include actual cards (not sealed products)."""
        cards = []
        for p in products:
            # Cards have a number and aren't sealed products
            if p.number and p.card_type:
                cards.append(p)
            elif "booster" not in p.clean_name.lower() and "box" not in p.clean_name.lower():
                # Might still be a card without full metadata
                if p.rarity:
                    cards.append(p)
        return cards

    def get_image_url(self, product: Product, size: str = "400w") -> str:
        """Get image URL for a product at specified size."""
        return f"{self.cdn_url}/{product.product_id}_{size}.jpg"

    def download_image(self, product: Product, size: str = "400w") -> Optional[Path]:
        """Download image for a product."""
        # Create group subdirectory
        group_dir = self.images_dir / f"group_{product.group_id}"
        group_dir.mkdir(exist_ok=True)

        # Output filename
        safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in product.clean_name)
        filename = f"{product.product_id}_{safe_name}.jpg"
        output_path = group_dir / filename

        # Skip if already downloaded
        if output_path.exists():
            return output_path

        # Try preferred size, fallback to smaller
        sizes_to_try = [size, self.config["images"]["fallback_size"]]

        for img_size in sizes_to_try:
            url = self.get_image_url(product, img_size)
            response = self._request_with_retry(url)

            if response and response.status_code == 200:
                with open(output_path, "wb") as f:
                    f.write(response.content)
                return output_path

        return None

    def download_all_images(
        self,
        products: List[Product],
        size: str = "400w",
    ) -> Dict[int, Path]:
        """Download images for all products with progress bar."""
        downloaded = {}
        failed = []

        print(f"\nDownloading {len(products)} images...")

        for product in tqdm(products, desc="Downloading images"):
            path = self.download_image(product, size)
            if path:
                downloaded[product.product_id] = path
            else:
                failed.append(product.product_id)

            # Rate limiting
            time.sleep(self.image_delay)

        print(f"\nDownloaded: {len(downloaded)}, Failed: {len(failed)}")

        if failed:
            # Save failed IDs for retry
            failed_file = self.raw_dir / "failed_images.json"
            with open(failed_file, "w") as f:
                json.dump(failed, f)
            print(f"Failed IDs saved to {failed_file}")

        return downloaded

    def create_training_manifest(
        self,
        products: List[Product],
        image_paths: Dict[int, Path],
    ) -> Path:
        """Create a manifest file for ML training."""
        manifest = []

        for product in products:
            if product.product_id in image_paths:
                manifest.append({
                    "product_id": product.product_id,
                    "name": product.name,
                    "clean_name": product.clean_name,
                    "image_path": str(image_paths[product.product_id].relative_to(self.output_base)),
                    "group_id": product.group_id,
                    "group_name": product.group_name,
                    "rarity": product.rarity,
                    "number": product.number,
                    "card_type": product.card_type,
                    "tags": product.tags,
                })

        manifest_file = self.processed_dir / "training_manifest.json"
        with open(manifest_file, "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"\nTraining manifest saved to {manifest_file}")
        print(f"Total cards with images: {len(manifest)}")

        # Also create a simple label file for classification
        labels_file = self.processed_dir / "labels.json"
        labels = {p["product_id"]: p["clean_name"] for p in manifest}
        with open(labels_file, "w") as f:
            json.dump(labels, f, indent=2)

        return manifest_file


def main():
    parser = argparse.ArgumentParser(description="Fetch TCG data from tcgcsv.com")
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
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Only fetch metadata, skip images",
    )
    parser.add_argument(
        "--images-only",
        action="store_true",
        help="Only download images (requires prior metadata fetch)",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force refresh metadata from API",
    )
    parser.add_argument(
        "--cards-only",
        action="store_true",
        default=True,
        help="Filter to only actual cards (not sealed products)",
    )
    parser.add_argument(
        "--prices-only",
        action="store_true",
        help="Only fetch and update prices (for daily price updates)",
    )
    parser.add_argument(
        "--with-prices",
        action="store_true",
        default=True,
        help="Include price data when fetching products",
    )
    args = parser.parse_args()

    output_base = Path(args.output)
    fetcher = TCGCSVFetcher(args.config, output_base)

    # Handle prices-only mode (for daily updates)
    if args.prices_only:
        print("=" * 60)
        print("PRICE UPDATE MODE")
        print("=" * 60)
        prices = fetcher.fetch_all_prices()
        fetcher.save_price_history(prices)

        # Update cached products with new prices if they exist
        cache_file = fetcher.raw_dir / "all_products.json"
        if cache_file.exists():
            print("\nUpdating cached products with new prices...")
            with open(cache_file) as f:
                products = [Product(**p) for p in json.load(f)]
            products = fetcher.update_products_with_prices(products, prices)
            with open(cache_file, "w") as f:
                json.dump([p.to_dict() for p in products], f, indent=2)
            print(f"Updated {len(products)} products with current prices")

        print("\n" + "=" * 60)
        print(f"DONE! Fetched prices for {len(prices)} products")
        print("=" * 60)
        return

    # Fetch metadata
    if not args.images_only:
        print("=" * 60)
        print("STEP 1: Fetching product metadata")
        print("=" * 60)
        products = fetcher.fetch_all_riftbound_products(force_refresh=args.force_refresh)
    else:
        # Load from cache
        cache_file = fetcher.raw_dir / "all_products.json"
        if not cache_file.exists():
            print("Error: No cached metadata found. Run without --images-only first.")
            return
        with open(cache_file) as f:
            products = [Product(**p) for p in json.load(f)]

    # Fetch and attach prices
    if args.with_prices:
        print("\n" + "=" * 60)
        print("STEP 1.5: Fetching price data")
        print("=" * 60)
        prices = fetcher.fetch_all_prices()
        products = fetcher.update_products_with_prices(products, prices)
        fetcher.save_price_history(prices)

        # Re-save products with prices
        cache_file = fetcher.raw_dir / "all_products.json"
        with open(cache_file, "w") as f:
            json.dump([p.to_dict() for p in products], f, indent=2)

    # Filter to cards only
    if args.cards_only:
        cards = fetcher.filter_cards_only(products)
        print(f"\nFiltered to {len(cards)} cards (from {len(products)} products)")
    else:
        cards = products

    # Download images
    if not args.metadata_only:
        print("\n" + "=" * 60)
        print("STEP 2: Downloading card images")
        print("=" * 60)
        image_paths = fetcher.download_all_images(
            cards,
            size=fetcher.config["images"]["preferred_size"],
        )

        # Create training manifest
        print("\n" + "=" * 60)
        print("STEP 3: Creating training manifest")
        print("=" * 60)
        fetcher.create_training_manifest(cards, image_paths)

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
