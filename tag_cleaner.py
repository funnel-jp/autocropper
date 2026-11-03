# file: tag_cleaner.py
# タグ一括整理: キャプションテキストをマスター辞書・ルール・設定ファイルに基づき整理
import os
import sys
import argparse
import re
import json
import csv
from pathlib import Path
from collections import Counter

def parse_tag_line(line):
    if ":" in line:
        return [t.strip().lower() for t in line.split(":", 1)[1].split(",") if t.strip()]
    return []

def load_character_settings(settings_path):
    settings = {"add": [], "remove_partial": [], "remove_exact": [], "keep_partial": [], "keep_exact": []}
    if not os.path.exists(settings_path): return settings
    with open(settings_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            upper_line = line.upper()
            if upper_line.startswith("ADD:"): 
                settings["add"].extend(parse_tag_line(line))
            elif upper_line.startswith("REMOVE_PARTIAL:") or upper_line.startswith("REMOVE:"): 
                settings["remove_partial"].extend(parse_tag_line(line))
            elif upper_line.startswith("REMOVE_EXACT:"): 
                settings["remove_exact"].extend(parse_tag_line(line))
            elif upper_line.startswith("KEEP_PARTIAL:") or upper_line.startswith("KEEP:"): 
                settings["keep_partial"].extend(parse_tag_line(line))
            elif upper_line.startswith("KEEP_EXACT:"): 
                settings["keep_exact"].extend(parse_tag_line(line))
    return settings

def load_character_tags_from_csv(csv_path="selected_tags.csv"):
    char_tags = set()
    if not os.path.exists(csv_path): return char_tags
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if str(row.get("category", "")).strip() == "4":
                    name = row.get("name", "").strip().replace("_", " ").lower()
                    if name: char_tags.add(name)
    except Exception as e:
        pass
    return char_tags

def normalize_master_dict(d):
    """辞書内のすべてのタグを小文字・半角スペース区切りに強制統一する"""
    if isinstance(d, dict):
        return {k: normalize_master_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [str(i).lower().replace("_", " ").strip() for i in d]
    elif isinstance(d, str):
        return d.lower().replace("_", " ").strip()
    return d

# 💡 [改修] エラーをキャッチして強制終了する安全装置の追加
def load_tag_master(master_path="tag_master.json"):
    if not os.path.exists(master_path):
        print(f"  -> [警告] {master_path} が見つかりません。自動マルゼン式フィルタリングは制限されます。")
        return {}
    try:
        with open(master_path, "r", encoding="utf-8") as f: 
            data = json.load(f)
            return normalize_master_dict(data)
    except Exception as e:
        print(f"  -> [エラー] マスター辞書の解析に失敗しました: {e}")
        print("  -> 処理を中断します。JSONの構文（カンマの抜け等）を確認してください。")
        sys.exit(1) # ここで裏側のプロセスを強制終了させ、UI側に異常を通知する

def load_char_tag_rules(rules_path="char_tag_rules.json"):
    if not os.path.exists(rules_path): return {}
    try:
        with open(rules_path, "r", encoding="utf-8") as f: return json.load(f)
    except: return {}

def get_ui_category_for_tag(tag_normalized, tag_master):
    if not tag_master: return "Unknown"
    
    pt = tag_master.get("physical_traits", {})
    if "hair_color" in pt and any(tag_normalized in v for v in pt["hair_color"].values()): return "Physical Traits"
    if "eye_color" in pt and any(tag_normalized in v for v in pt["eye_color"].values()): return "Physical Traits"
    if "hair_style" in pt and tag_normalized in pt["hair_style"]: return "Physical Traits"
    if "body_features" in pt and tag_normalized in pt["body_features"]: return "Physical Traits"
    
    idf = tag_master.get("identity_features", {})
    for tag_list in idf.values():
        if tag_normalized in tag_list: return "Identity Features"
    if tag_normalized in tag_master.get("character_accessories", []): return "Identity Features"
        
    cp = tag_master.get("clothing_and_props", {})
    if tag_normalized in cp.get("clothing", []) or tag_normalized in tag_master.get("clothing", []): return "Clothing"
    if tag_normalized in cp.get("props_and_weapons", []) or tag_normalized in tag_master.get("props_and_weapons", []): return "Props & Weapons"
    
    if tag_normalized in cp.get("ambiguous_decorations", []): return "Ambiguous"
    
    if tag_normalized in tag_master.get("facial_expressions", []): return "Facial Expressions"
    if tag_normalized in tag_master.get("composition_and_perspective", []): return "Composition"
    if tag_normalized in tag_master.get("background_and_scene", []): return "Background & Scene"
    if tag_normalized in tag_master.get("style_and_effects", []): return "Style & Effects"
    
    return "Unknown"

def build_character_profile(target_dir, char_name, tag_master):
    profile = {
        "Physical Traits": set(), "Identity Features": set(), "Clothing": set(),
        "Props & Weapons": set(), "hair_color_groups": set(), "eye_color_groups": set()
    }
    if not tag_master: return profile
        
    target_path = Path(target_dir)
    all_stand_tags = []
    for txt_path in target_path.glob(f"{char_name}_*.txt"):
        stem = txt_path.stem
        if "_face" in stem or "_upper" in stem or "_full" in stem:
            tags = [t.strip().lower() for t in txt_path.read_text(encoding="utf-8").split(",") if t.strip()]
            tags = [t.replace("_", " ") for t in tags]
            all_stand_tags.extend(tags)
            
    if not all_stand_tags: return profile
        
    tag_counts = Counter(all_stand_tags)
    reliable_tags = {tag for tag, count in tag_counts.items() if count >= 2}
    
    pt = tag_master.get("physical_traits", {})
    if "hair_color" in pt:
        for group_name, colors in pt["hair_color"].items():
            if any(c in reliable_tags for c in colors): profile["hair_color_groups"].add(group_name)
    if "eye_color" in pt:
        for group_name, colors in pt["eye_color"].items():
            if any(c in reliable_tags for c in colors): profile["eye_color_groups"].add(group_name)
                
    for tag in reliable_tags:
        ui_cat = get_ui_category_for_tag(tag, tag_master)
        if ui_cat in profile:
            profile[ui_cat].add(tag)
            
    return profile

def matches_profile(tag_normalized, tag_master, profile, ui_cat):
    if ui_cat == "Physical Traits":
        pt = tag_master.get("physical_traits", {})
        if "hair_color" in pt:
            for group_name, colors in pt["hair_color"].items():
                if tag_normalized in colors: return group_name in profile["hair_color_groups"]
        if "eye_color" in pt:
            for group_name, colors in pt["eye_color"].items():
                if tag_normalized in colors: return group_name in profile["eye_color_groups"]
        return tag_normalized in profile["Physical Traits"]
        
    elif ui_cat in ["Identity Features", "Clothing", "Props & Weapons"]:
        return tag_normalized in profile[ui_cat]
        
    return True

BACKGROUND_NOISE_TAGS = {
    "tree", "building", "cloud", "sky", "scenery", "outdoors", "indoors", "window", 
    "floor", "wall", "shadow", "sunlight", "mountain", "water", "plant", "nature", "day", "night"
}
LOWER_BODY_TAGS = {
    "shoes", "boots", "skirt", "thighhighs", "pantyhose", "legwear", "kneehighs", 
    "socks", "toes", "feet", "legs", "thighs", "pants", "shorts", "sneakers", "loafers"
}
CHARACTER_BODY_TAGS = {
    "1girl", "1boy", "solo", "breasts", "cleavage", "navel", "smile", "blush", 
    "open mouth", "looking at viewer", "shirt", "jacket", "dress"
}

def get_args():
    parser = argparse.ArgumentParser(description="キャプションタグ一括整理ツール")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--tag_settings", type=str, default="tag_settings.txt")
    return parser.parse_args()

def clean_tags(target_dir, tag_settings_file):
    target_path = Path(target_dir)
    settings = load_character_settings(tag_settings_file)
    csv_character_tags = load_character_tags_from_csv("selected_tags.csv")
    tag_master = load_tag_master("tag_master.json")
    char_rules = load_char_tag_rules()
            
    char_groups = set()
    for txt_path in target_path.glob("*.txt"):
        char_groups.add(txt_path.stem.split("_")[0])
        
    count = 0
    for char_name in char_groups:
        profile = build_character_profile(target_dir, char_name, tag_master)
        
        fallback_rules = char_rules.get("ALL", ["Physical Traits", "Identity Features"])
        categories_to_drop = char_rules.get(char_name, fallback_rules)
        
        for txt_path in target_path.glob(f"{char_name}_*.txt"):
            is_face = "_face" in txt_path.stem
            is_upper = "_upper" in txt_path.stem
            is_illust = "_illust" in txt_path.stem
            is_eye = "eye" in txt_path.stem
            is_full = "_full" in txt_path.stem
            is_obj = not (is_face or is_upper or is_illust or is_eye or is_full)
            
            raw_tags = [t.strip().lower() for t in txt_path.read_text(encoding="utf-8").split(",") if t.strip()]
            has_multiple_chars = any(t in ["2girls", "3girls", "multiple girls", "boys", "group"] for t in [rt.replace("_", " ") for rt in raw_tags])
            
            cleaned_tags = []
            for original_tag in raw_tags:
                normalized_tag = original_tag.replace("_", " ")
                
                if any(k == normalized_tag or k == original_tag for k in settings["keep_exact"]):
                    cleaned_tags.append(normalized_tag); continue
                if any(re.search(rf'(^|[\s_]){re.escape(k)}(?:s|es)?([\s_]|$)', original_tag, re.IGNORECASE) for k in settings["keep_partial"]):
                    cleaned_tags.append(normalized_tag); continue

                if normalized_tag in csv_character_tags: continue
                    
                if any(r == normalized_tag or r == original_tag for r in settings["remove_exact"]): continue
                if any(re.search(rf'(^|[\s_]){re.escape(p)}(?:s|es)?([\s_]|$)', original_tag, re.IGNORECASE) for p in settings["remove_partial"]):
                    continue
                
                ui_cat = get_ui_category_for_tag(normalized_tag, tag_master)
                
                if ui_cat == "Ambiguous":
                    pass
                elif ui_cat in categories_to_drop:
                    if not is_illust:
                        continue 
                    else:
                        if has_multiple_chars:
                            if matches_profile(normalized_tag, tag_master, profile, ui_cat):
                                continue
                            else:
                                cleaned_tags.append(normalized_tag) 
                                continue
                        else:
                            continue 
                            
                if not is_illust and normalized_tag in BACKGROUND_NOISE_TAGS: continue
                if (is_face or is_upper) and normalized_tag in LOWER_BODY_TAGS: continue
                if (is_obj or is_eye) and normalized_tag in CHARACTER_BODY_TAGS: continue

                cleaned_tags.append(normalized_tag)
                
            final_tags = []
            if char_name not in final_tags: final_tags.append(char_name)
            if is_eye and "eye focus" not in final_tags: final_tags.append("eye focus")
            for t in settings["add"]:
                if t not in final_tags: final_tags.append(t)
            for t in cleaned_tags:
                if t not in final_tags and t != char_name: final_tags.append(t)
                    
            txt_path.write_text(", ".join(final_tags), encoding="utf-8")
            count += 1
            
    print(f"  -> [完了] {count} 件のテキストファイルのタグを整理・更新しました。")

def main():
    args = get_args()
    clean_tags(args.input_dir, args.tag_settings)

if __name__ == "__main__":
    main()
