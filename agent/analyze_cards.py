"""
カードデータ解析スクリプト

EN_Card_Data.csv をダウンロード後に実行して:
1. カードIDのマッピング表を作成
2. Dragapult ex/Dusknoir デッキの正確なカードIDを特定
3. デッキlistをdeck.csvとして出力

使い方:
  python analyze_cards.py --csv ../data/EN_Card_Data.csv --deck dragapult
"""

import argparse
import csv
import json
import os


def load_card_data(csv_path: str) -> list[dict]:
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def build_name_to_id(cards: list[dict]) -> dict[str, list[dict]]:
    """カード名 → カード情報リストのマッピング"""
    mapping: dict[str, list[dict]] = {}
    for card in cards:
        name = card.get("name", "").strip()
        if name not in mapping:
            mapping[name] = []
        mapping[name].append(card)
    return mapping


def search_cards(cards: list[dict], query: str) -> list[dict]:
    """名前の部分一致でカードを検索"""
    q = query.lower()
    return [c for c in cards if q in c.get("name", "").lower()]


def print_card_info(card: dict) -> None:
    print(f"  ID: {card.get('id', 'N/A')}")
    print(f"  Name: {card.get('name', 'N/A')}")
    print(f"  Set: {card.get('set', card.get('setCode', 'N/A'))}")
    print(f"  HP: {card.get('hp', 'N/A')}")
    print(f"  CardType: {card.get('cardType', 'N/A')}")
    print(f"  Regulation: {card.get('regulationMark', 'N/A')}")
    print()


# Dragapult ex/Dusknoir デッキのカード名リスト
DRAGAPULT_DECK = [
    # ポケモン
    ("Dreepy", 4),
    ("Drakloak", 3),
    ("Dragapult ex", 3),
    ("Duskull", 2),
    ("Dusclops", 1),
    ("Dusknoir", 1),
    ("Fezandipiti ex", 1),
    ("Munkidori", 1),
    ("Budew", 1),
    # トレーナー
    ("Hilda", 4),
    ("Crispin", 3),
    ("N", 3),
    ("Boss's Orders", 2),
    ("Unfair Stamp", 2),
    ("Ultra Ball", 4),
    ("Rare Candy", 3),
    ("Technical Machine: Devolution", 2),
    ("Path to the Peak", 3),
    # エネルギー
    ("Basic Fire Energy", 4),
    ("Basic Psychic Energy", 4),
]


def build_deck(cards: list[dict], deck_spec: list[tuple[str, int]], output_path: str) -> None:
    """デッキスペックからdeck.csvを生成"""
    name_to_id = build_name_to_id(cards)
    deck_ids = []
    missing = []

    print("=== デッキ構築 ===")
    for name, count in deck_spec:
        matches = name_to_id.get(name, [])
        if not matches:
            # 部分一致で検索
            matches = search_cards(cards, name)

        if matches:
            # 最初のマッチを使用（複数ある場合は最初のH以降の規制マーク優先）
            best = matches[0]
            for m in matches:
                reg = m.get("regulationMark", "")
                if reg in ("H", "I", "J"):
                    best = m
                    break
            card_id = best.get("id", "?")
            print(f"  {name} × {count}: ID={card_id} ({best.get('set', '')})")
            deck_ids.extend([str(card_id)] * count)
        else:
            print(f"  [NOT FOUND] {name} × {count}")
            missing.append(name)
            deck_ids.extend(["0"] * count)

    total = sum(count for _, count in deck_spec)
    print(f"\n合計: {total} 枚 (不足: {len(missing)} 種)")

    with open(output_path, "w") as f:
        for card_id in deck_ids:
            f.write(card_id + "\n")

    print(f"deck.csv を書き出し: {output_path}")
    if missing:
        print(f"見つからなかったカード: {missing}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="EN_Card_Data.csv のパス")
    parser.add_argument("--deck", type=str, default="dragapult", choices=["dragapult"],
                        help="生成するデッキ名")
    parser.add_argument("--search", type=str, default="", help="カード名検索クエリ")
    parser.add_argument("--output", type=str, default="deck.csv")
    args = parser.parse_args()

    cards = load_card_data(args.csv)
    print(f"カード総数: {len(cards)}")

    if args.search:
        results = search_cards(cards, args.search)
        print(f"\n'{args.search}' の検索結果 ({len(results)} 件):")
        for c in results[:20]:
            print_card_info(c)
        return

    # カラム名を確認
    if cards:
        print("CSVカラム:", list(cards[0].keys()))

    deck_spec = DRAGAPULT_DECK
    build_deck(cards, deck_spec, args.output)

    # カードIDのJSONマップを保存（RL学習時に使用）
    id_map = {c.get("name", ""): int(c.get("id", 0)) for c in cards if c.get("id")}
    map_path = os.path.join(os.path.dirname(args.output), "card_id_map.json")
    with open(map_path, "w") as f:
        json.dump(id_map, f, ensure_ascii=False, indent=2)
    print(f"カードIDマップを保存: {map_path}")


if __name__ == "__main__":
    main()
