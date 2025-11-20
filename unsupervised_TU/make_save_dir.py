import os
from datetime import datetime

def make_save_dir(base="./result", args=None, extra=None, add_timestamp=False):
    """
    自動根據 args 建立層級式 save_dir
    base: 最上層資料夾
    args: argparse.Namespace 或 具有屬性的物件
    extra: 額外自訂字典（例如 {'comment': 'debug'}）
    add_timestamp: 是否加上時間戳，防止覆蓋
    """

    # 主要屬性（可依需求調整順序）
    path_hierarchy = [
        ("DS", args.DS),
        ("mode", args.mode),
        ("neg_include_self", args.neg_include_self) if args.neg_include_self else None,
        ("aug", args.aug),
        ("shuffle_DBN", args.shuffle_DBN),
        ("or_loss", args.or_loss),
    ]

    # 這些用來放在最底層（用底線連接）
    tags = []
    if getattr(args, "use_kde", False):
        tags.append("kde")
    if getattr(args, "no_clamp", False):
        tags.append("no_clamp")
    if extra:
        for k, v in extra.items():
            tags.append(f"{k}_{v}")

    # 建立層級結構
    save_dir = base
    for key, val in path_hierarchy:
        save_dir = os.path.join(save_dir, f"{key}_{val}")

    # 最底層再加上一層合併標籤
    if tags:
        save_dir = os.path.join(save_dir, "_".join(tags))

    # 加上時間戳以避免覆蓋（可選）
    if add_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(save_dir, timestamp)

    # 建立資料夾
    os.makedirs(save_dir, exist_ok=True)
    return save_dir
