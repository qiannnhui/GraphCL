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
        ("mode", args.mode),
        ("RBO_p", args.RBO_p) if args.mode=="reweight_FNs_by_RBO" and hasattr(args, "RBO_p") else None,
        ("tn_weight", args.tn_weight) if args.mode=="reweight_FNs_by_RBO" and args.reweight_strategy=="boost_TNs" and hasattr(args, "tn_weight") else None,
        ("DS", args.DS),
        ("neg_include_self", args.neg_include_self) if args.neg_include_self else None,
        ("aug", args.aug) if args.rotate == "none" else None,
        ("rotate", args.rotate) if args.rotate in ['random', "by_degree"] and hasattr(args, "rotate") else None,
        ("rotate_angle_deg", args.rotate_angle_deg) if args.mode=="reweighted_by_angle" and hasattr(args, "rotate_angle_deg") and hasattr(args, "rotate") else None,
        ("shuffle_DBN", args.shuffle_DBN) if args.shuffle_DBN else None,
        ("or_loss", args.or_loss) if args.or_loss else None,
        ("EN_THRESHOLD", f"{args.base_en_threshold}-{args.max_en_threshold}") if args.mode in ["rm_FNs_by_ENs", "reweight_FNs_by_ENs"] and hasattr(args, "base_en_threshold") and hasattr(args, "max_en_threshold") else None,
        ("Reweight_Strategy", args.reweight_strategy) if args.mode=="reweight_FNs_by_ENs" and hasattr(args, "reweight_strategy") else None,
        ("Reweight_Strategy", args.reweight_strategy) if args.mode=="reweight_FNs_by_RBO" and args.reweight_strategy=="boost_TNs" else None,
        ("Coverage_Threshold", args.coverage_threshold) if args.mode=="rm_FNs_by_ENs" or (args.mode == "reweight_FNs_by_ENs" and getattr(args, "reweight_strategy", "") == "thresholded") and hasattr(args, "coverage_threshold") else None,
        ("Renormalization", args.renormalization) if args.renormalization else None,
        ("RBO_anchor", args.RBO_anchor) if args.RBO_anchor else None,
        ("denominator_anchor", args.denominator_anchor) if args.denominator_anchor else None,
    ]
    path_hierarchy = [item for item in path_hierarchy if item is not None]

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
