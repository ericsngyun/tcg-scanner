import 'package:equatable/equatable.dart';

/// Represents a TCG card with its metadata and pricing.
class Card extends Equatable {
  final int productId;
  final String name;
  final String cleanName;
  final String imagePath;
  final int groupId;
  final String groupName;
  final String rarity;
  final String number;
  final String cardType;
  final List<String> tags;
  final CardPricing? pricing;

  const Card({
    required this.productId,
    required this.name,
    required this.cleanName,
    required this.imagePath,
    required this.groupId,
    required this.groupName,
    required this.rarity,
    required this.number,
    required this.cardType,
    required this.tags,
    this.pricing,
  });

  factory Card.fromJson(Map<String, dynamic> json) {
    return Card(
      productId: json['product_id'] as int,
      name: json['name'] as String,
      cleanName: json['clean_name'] as String,
      imagePath: json['image_path'] as String,
      groupId: json['group_id'] as int,
      groupName: json['group_name'] as String,
      rarity: json['rarity'] as String? ?? 'Unknown',
      number: json['number'] as String? ?? '',
      cardType: json['card_type'] as String? ?? 'Unknown',
      tags: (json['tags'] as List<dynamic>?)?.cast<String>() ?? [],
      pricing: json['pricing'] != null
          ? CardPricing.fromJson(json['pricing'] as Map<String, dynamic>)
          : null,
    );
  }

  Map<String, dynamic> toJson() => {
        'product_id': productId,
        'name': name,
        'clean_name': cleanName,
        'image_path': imagePath,
        'group_id': groupId,
        'group_name': groupName,
        'rarity': rarity,
        'number': number,
        'card_type': cardType,
        'tags': tags,
        'pricing': pricing?.toJson(),
      };

  @override
  List<Object?> get props => [productId];
}

/// Pricing information for a card.
class CardPricing extends Equatable {
  final double? lowPrice;
  final double? midPrice;
  final double? highPrice;
  final double? marketPrice;
  final DateTime? lastUpdated;

  const CardPricing({
    this.lowPrice,
    this.midPrice,
    this.highPrice,
    this.marketPrice,
    this.lastUpdated,
  });

  factory CardPricing.fromJson(Map<String, dynamic> json) {
    return CardPricing(
      lowPrice: (json['low_price'] as num?)?.toDouble(),
      midPrice: (json['mid_price'] as num?)?.toDouble(),
      highPrice: (json['high_price'] as num?)?.toDouble(),
      marketPrice: (json['market_price'] as num?)?.toDouble(),
      lastUpdated: json['last_updated'] != null
          ? DateTime.parse(json['last_updated'] as String)
          : null,
    );
  }

  Map<String, dynamic> toJson() => {
        'low_price': lowPrice,
        'mid_price': midPrice,
        'high_price': highPrice,
        'market_price': marketPrice,
        'last_updated': lastUpdated?.toIso8601String(),
      };

  /// Returns the best available price for display.
  double? get displayPrice => marketPrice ?? midPrice ?? lowPrice;

  @override
  List<Object?> get props =>
      [lowPrice, midPrice, highPrice, marketPrice, lastUpdated];
}
