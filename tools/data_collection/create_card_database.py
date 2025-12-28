#!/usr/bin/env python3
"""
Create card database JSON for Flutter app from collected TCG data.

This script transforms the raw product data into a format optimized for
the mobile app's ML pipeline and UI.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any


def is_actual_card(product: Dict[str, Any]) -> bool:
    """
    Filter out non-card products like box sets, bundles, etc.

    Args:
        product: Product data dictionary

    Returns:
        True if this is an actual playable card
    """
    name = product.get('name', '').lower()

    # Exclude box sets, bundles, and other non-cards
    exclude_keywords = [
        'box set',
        'bundle',
        'booster',
        'pack',
        'playmat',
        'sleeves',
        'deck box',
    ]

    for keyword in exclude_keywords:
        if keyword in name:
            return False

    # Must have a card type to be considered a card
    card_type = product.get('card_type')
    if not card_type:
        return False

    # Must have an image
    if not product.get('image_url'):
        return False

    return True


def transform_product_to_card(product: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform raw product data to card database format.

    Args:
        product: Raw product data

    Returns:
        Transformed card data for Flutter app (matches Card.fromJson format)
    """
    # Extract relevant fields matching Card model structure
    card = {
        'product_id': product['product_id'],
        'name': product['name'],
        'clean_name': product.get('clean_name', product['name']),
        'image_path': product.get('image_url', ''),  # Will be URL for now
        'group_id': product.get('group_id', 0),
        'group_name': product.get('group_name', 'Unknown'),
        'rarity': product.get('rarity', 'Unknown'),
        'number': product.get('number', ''),
        'card_type': product.get('card_type', 'Unknown'),
        'tags': product.get('tags', []),
        'pricing': {
            'low_price': product.get('low_price'),
            'mid_price': product.get('mid_price'),
            'high_price': product.get('high_price'),
            'market_price': product.get('market_price'),
            'last_updated': None,  # Will be updated by price fetcher
        }
    }

    return card


def create_card_database(
    input_file: Path,
    output_file: Path,
    verbose: bool = False
) -> None:
    """
    Create card database JSON from raw product data.

    Args:
        input_file: Path to all_products.json
        output_file: Path for output cards.json
        verbose: Print progress information
    """
    # Load raw product data
    if verbose:
        print(f"Loading products from: {input_file}")

    with open(input_file, 'r') as f:
        products = json.load(f)

    if verbose:
        print(f"Total products loaded: {len(products)}")

    # Filter and transform
    cards = []
    for product in products:
        if is_actual_card(product):
            card = transform_product_to_card(product)
            cards.append(card)

    if verbose:
        print(f"Actual cards found: {len(cards)}")
        print(f"\nBreakdown by type:")
        type_counts = {}
        for card in cards:
            card_type = card['card_type']
            type_counts[card_type] = type_counts.get(card_type, 0) + 1

        for card_type, count in sorted(type_counts.items()):
            print(f"  {card_type}: {count}")

        print(f"\nBreakdown by set:")
        set_counts = {}
        for card in cards:
            card_set = card['group_name']
            set_counts[card_set] = set_counts.get(card_set, 0) + 1

        for card_set, count in sorted(set_counts.items()):
            print(f"  {card_set}: {count}")

    # Sort by name for easier browsing
    cards.sort(key=lambda c: c['name'])

    # Save to output file
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(cards, f, indent=2)

    if verbose:
        print(f"\n✅ Card database saved to: {output_file}")
        print(f"   Total cards: {len(cards)}")
        print(f"   File size: {output_file.stat().st_size / 1024:.1f} KB")


def main():
    parser = argparse.ArgumentParser(
        description='Create card database JSON for Flutter app'
    )
    parser.add_argument(
        '--input',
        type=Path,
        default=Path('ml/data/raw/riftbound/all_products.json'),
        help='Input all_products.json file'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('mobile/flutter/assets/data/cards.json'),
        help='Output cards.json file for Flutter app'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Print progress information'
    )

    args = parser.parse_args()

    create_card_database(
        input_file=args.input,
        output_file=args.output,
        verbose=args.verbose
    )


if __name__ == '__main__':
    main()
