# dataset_builder.py
import os
import sys
import cv2
import numpy as np
import shutil
import random
import argparse
import subprocess
import concurrent.futures
import json
from pathlib import Path

SIMPLE_BG_TAGS = {
    "simple background", "white background", "transparent background", 
    "grey background", "black background", "blue background", 
    "red background", "green background", "yellow background", "pink background", "purple background",
    "colorful background", "gradient background", "patterned background", 
    "plaid background", "striped background", "sunburst background", "polka dot background", "two-tone background"
}

def setup_dll_path(venv_path=None):
    if sys.platform != "win32": return
    import site
    paths = site.getsitepackages()
    if venv_path:
        lib_dir = Path(venv_path) / "Lib" / "site-packages"
        if lib_dir.exists(): paths.append(str(lib_dir))
    for p in paths:
        bin_p = Path(p) / "nvidia" / "cudnn" / "bin"
        if bin_p.exists():
            os.add_dll_directory(str(bin_p))
            os.environ["PATH"] = str(bin_p) + os.pathsep + os.environ["PATH"]

def load_json_defaults():
    try:
        return json.loads((Path(__file__).resolve().parent / "settings.json").read_text(encoding="utf-8"))
    except Exception:
        return {}

JSON_DEFAULTS = load_json_defaults()

def get_args():
    parser = argparse.ArgumentParser(description="LoRAデータセット構築ツール (画像・キャプション合成)")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--bg_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--target_total", type=int, default=JSON_DEFAULTS.get("db_total", 500))
    parser.add_argument("--jitter_prob", type=float, default=JSON_DEFAULTS.get("db_jitter", 0.15))
    parser.add_argument("--blur_prob", type=float, default=JSON_DEFAULTS.get("db_blur", 0.8))
    parser.add_argument("--sd_scripts_path", type=str, required=True)
    # メモリに画像を保持する高速処理モードのフラグ
    parser.add_argument("--keep_in_memory", action="store_true", help="画像をメモリに保持して高速化")
    
    # カテゴリごとの目標割合設定
    parser.add_argument("--ratio_face", type=float, default=JSON_DEFAULTS.get("db_ratio_face", 30.0))
    parser.add_argument("--ratio_upper", type=float, default=JSON_DEFAULTS.get("db_ratio_upper", 30.0))
    parser.add_argument("--ratio_full", type=float, default=JSON_DEFAULTS.get("db_ratio_full", 20.0))
    parser.add_argument("--ratio_illust", type=float, default=JSON_DEFAULTS.get("db_ratio_illust", 10.0))
    parser.add_argument("--ratio_right_eye", type=float, default=JSON_DEFAULTS.get("db_ratio_reye", 2.5))
    parser.add_argument("--ratio_left_eye", type=float, default=JSON_DEFAULTS.get("db_ratio_leye", 2.5))
    parser.add_argument("--ratio_obj", type=float, default=JSON_DEFAULTS.get("db_ratio_obj", 5.0))
    
    return parser.parse_args()

def prepare_background_images(bg_dir):
    bg_path = Path(bg_dir)
    if not bg_path.exists():
        bg_path.mkdir(parents=True)
        return
        
    raw_dir = bg_path / "raw_bg"
    raw_dir.mkdir(exist_ok=True)
    
    target_sizes = [(1024, 1024), (1216, 832), (832, 1216)]
    processed_any = False
    
    for img_path in bg_path.glob("*.*"):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
            
        stem = img_path.stem
        if any(stem.endswith(f"_{w}x{h}") for w, h in target_sizes):
            continue
            
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        h, w = img.shape[:2]
        processed_any = True
        
        for tw, th in target_sizes:
            scale = max(tw / w, th / h)
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            y_start = (new_h - th) // 2
            x_start = (new_w - tw) // 2
            cropped = resized[y_start:y_start+th, x_start:x_start+tw]
            
            save_name = f"{stem}_{tw}x{th}.jpg"
            cv2.imwrite(str(bg_path / save_name), cropped)
            
        shutil.move(str(img_path), str(raw_dir / img_path.name))
        
        txt_path = img_path.with_suffix(".txt")
        if txt_path.exists():
            shutil.move(str(txt_path), str(raw_dir / txt_path.name))

    if processed_any:
        print("  -> [情報] 新しい背景画像を3つのアスペクト比に分割・最適化しました。元画像は raw_bg フォルダに退避しました。")

def generate_caption(src_txt_path, category):
    tags = []
    if src_txt_path and Path(src_txt_path).exists():
        with open(src_txt_path, "r", encoding="utf-8") as f:
            tags = [t.strip().lower() for t in f.read().split(",") if t.strip()]
            
    final_tags = tags.copy()
        
    # --- 構図・画風の概念分離（メタタグの自動付与） ---
    if category in ['full', 'upper', 'face']:
        # 立ち絵合成特有の「平坦なノベルゲーム的構図」への過学習を防ぐ
        for t in ["visual novel", "fake screenshot", "looking at viewer"]:
            if t not in final_tags: final_tags.append(t)
            
        if category in ['full', 'upper'] and "standing" not in final_tags:
            final_tags.append("standing")
            
        if category == 'face' and "close-up" not in final_tags:
            final_tags.append("close-up")

    elif category in ['obj', 'left_eye', 'right_eye']:
        # 小物や目は単色背景のため、設定資料であることを明示
        for t in ["character design", "simple background"]:
            if t not in final_tags: final_tags.append(t)
        
        if "close-up" not in final_tags: final_tags.append("close-up")
        if category == 'obj' and "macro" not in final_tags: final_tags.append("macro")

    # illust（一枚絵）の場合は、Taggerが付けた元々のタグをそのまま尊重する（余計なメタタグは付けない）
            
    return ", ".join(final_tags)

def apply_color_jitter(img, prob):
    import math
    if random.random() > prob: return img
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    h_shift = random.uniform(-8, 8)
    s_scale = random.uniform(0.85, 1.15)
    v_scale = random.uniform(0.85, 1.15)
    hsv[:, :, 0] = (hsv[:, :, 0] + h_shift) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s_scale, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * v_scale, 0, 255)
    jittered = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    alpha = random.uniform(0.9, 1.1)
    beta = random.uniform(-10, 10)
    return cv2.convertScaleAbs(jittered, alpha=alpha, beta=beta)

def composite_smart(fg_img, category, bg_images_dict, jitter_prob, blur_prob, bbox=None):
    fh, fw = fg_img.shape[:2]
    has_alpha = (fg_img.shape[2] == 4)
    has_bg = any(len(v) > 0 for v in bg_images_dict.values())
    is_blurred = False

    # 一枚絵（透過なし）の場合は、アスペクト比を完全に維持してジッターのみ適用
    if not has_alpha or category == 'illust':
        return apply_color_jitter(fg_img, jitter_prob), None, is_blurred, None

    # 小物(obj)や目(eye)の処理：変形は行わず、過学習を防ぐため多様な単色背景を敷く
    if category not in ['full', 'upper', 'face']:
        bg_color_choice = random.choice([
            ((255, 255, 255), "white"), 
            ((0, 0, 0), "black"),       
            ((128, 128, 128), "grey"), 
            ((random.randint(50, 200), random.randint(50, 200), random.randint(50, 200)), "colorful")
        ])
        bg_color, bg_color_name = bg_color_choice

        fg_bgr = fg_img[:, :, :3]
        alpha = fg_img[:, :, 3:4] / 255.0
        bg_img_raw = np.full((fh, fw, 3), bg_color, dtype=np.uint8)
        merged_fg = (fg_bgr * alpha + bg_img_raw * (1 - alpha)).astype(np.uint8)
        return apply_color_jitter(merged_fg, jitter_prob), None, False, bg_color_name

    # --- 以下は full, upper, face の背景合成処理 ---

    # 【品質向上策1】合成エッジの平滑化 (Edge Blending)
    alpha_channel = fg_img[:, :, 3].copy()
    alpha_blurred = cv2.GaussianBlur(alpha_channel, (3, 3), 0)
    fg_img[:, :, 3] = np.where(alpha_channel > 0, alpha_blurred, 0)

    if not has_bg:
        fg_bgr = fg_img[:, :, :3]
        alpha = fg_img[:, :, 3:4] / 255.0
        bg_img_raw = np.full((fh, fw, 3), (128, 128, 128), dtype=np.uint8)
        merged_fg = (fg_bgr * alpha + bg_img_raw * (1 - alpha)).astype(np.uint8)
        return apply_color_jitter(merged_fg, jitter_prob), None, False, "grey"

    res_list = [(1024, 1024), (832, 1216), (1216, 832)]
    tw, th = random.choice(res_list)
    
    available_bgs = bg_images_dict.get((tw, th), [])
    if not available_bgs:
        all_bgs = sum(bg_images_dict.values(), [])
        bg_raw, bg_path = random.choice(all_bgs)
        res = cv2.resize(bg_raw, (tw, th), interpolation=cv2.INTER_LANCZOS4)
    else:
        bg_raw, bg_path = random.choice(available_bgs)
        res = bg_raw.copy()

    # 【品質向上策2】背景のアテンション低下 (Background Desaturation)
    if random.random() < 0.5:
        hsv_bg = cv2.cvtColor(res, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv_bg[:, :, 1] *= random.uniform(0.3, 0.6)
        res = cv2.cvtColor(hsv_bg.astype(np.uint8), cv2.COLOR_HSV2BGR)

    if random.random() < blur_prob:
        k_size = random.choice([15, 21, 27, 33, 41])
        res = cv2.GaussianBlur(res, (k_size, k_size), 0)
        is_blurred = True

    # スケール調整
    f_s = (th / fh) * random.uniform(0.85, 1.0)
    fg_s = cv2.resize(fg_img, (int(fw*f_s), int(fh*f_s)), interpolation=cv2.INTER_AREA)
    sh, sw = fg_s.shape[:2]

    x_off = random.randint(0, max(0, tw - sw)) if sw <= tw else random.randint(tw - sw, 0)
    
    # 垂直位置合わせ
    if category in ['upper', 'face']:
        y_off = th - sh
    else:  # category == 'full'
        margin = int(th * 0.1)
        y_off = random.randint(max(0, th - sh - margin), th - sh) if th > sh else th - sh

    y1, y2, x1, x2 = max(0, y_off), min(th, y_off + sh), max(0, x_off), min(tw, x_off + sw)
    ih, iw = y2 - y1, x2 - x1
    if ih <= 0 or iw <= 0: return apply_color_jitter(res, jitter_prob), bg_path, is_blurred, None
    
    fy1, fx1 = (0 if y_off >= 0 else -y_off), (0 if x_off >= 0 else -x_off)
    roi, fg_roi = res[y1:y2, x1:x2], fg_s[fy1:fy1+ih, fx1:fx1+iw]
    alpha_fg = fg_roi[:, :, 3:4] / 255.0
    res[y1:y2, x1:x2] = (fg_roi[:,:,:3] * alpha_fg + roi * (1-alpha_fg)).astype(np.uint8)

    return apply_color_jitter(res, jitter_prob), bg_path, is_blurred, None

def worker_process(args_tuple):
    img_data, cat_name, bg_images_dict, jitter_prob, blur_prob, bbox, save_p, src_txt_path = args_tuple
    try:
        if isinstance(img_data, (str, Path)):
            img_src = cv2.imread(str(img_data), cv2.IMREAD_UNCHANGED)
            if img_src is None:
                return 0, f"Error: Cannot read image {img_data}"
        else:
            img_src = img_data

        res_img, used_bg_path, is_blurred, solid_bg_color = composite_smart(img_src, cat_name, bg_images_dict, jitter_prob, blur_prob, bbox)
        cv2.imwrite(str(save_p), res_img)
        
        caption = generate_caption(src_txt_path, cat_name)
        
        with open(save_p.with_suffix('.txt'), "w", encoding="utf-8") as f:
            f.write(caption)
            
        return 1, f"Success"
    except Exception as e:
        return 0, f"Error: {e}"

def build_dataset():
    args = get_args()
    setup_dll_path(args.sd_scripts_path + "/venv")

    print("\n--- 1. 初期クリーンアップ ---")
    json_files = list(Path(args.input_dir).glob("*.json"))
    cleaned_count = 0
    for j_path in json_files:
        if not j_path.with_suffix('.png').exists():
            j_path.unlink()
            cleaned_count += 1
    if cleaned_count > 0: print(f"  -> {cleaned_count}個の孤立したメタデータを削除しました。")

    output_base = Path(args.output_dir)
    if output_base.exists(): shutil.rmtree(output_base)
    output_base.mkdir(parents=True, exist_ok=True)

    print("\n--- 2. 背景画像の準備 ---")
    prepare_background_images(args.bg_dir)
    
    bg_images_dict = {(1024, 1024): [], (1216, 832): [], (832, 1216): []}
    for p in Path(args.bg_dir).glob("*.*"):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            stem = p.stem
            matched_size = None
            for size in bg_images_dict.keys():
                if stem.endswith(f"_{size[0]}x{size[1]}"):
                    matched_size = size
                    break
            
            if matched_size:
                img = cv2.imread(str(p))
                if img is not None:
                    bg_images_dict[matched_size].append((img, p))

    input_files = list(Path(args.input_dir).glob("*.png"))
    input_files = [p for p in input_files if p.parent == Path(args.input_dir)]
    
    char_groups = {}
    for p in input_files:
        char_name = p.stem.split("_")[0]
        if char_name not in char_groups:
            char_groups[char_name] = {'full': [], 'upper': [], 'face': [], 'illust': [], 'left_eye': [], 'right_eye': [], 'obj': []}
            
        stem = p.stem
        if '_illust' in stem: cat = 'illust'
        elif '_full' in stem: cat = 'full'
        elif '_upper' in stem: cat = 'upper'
        elif '_face' in stem: cat = 'face'
        elif 'left_eye' in stem: cat = 'left_eye'
        elif 'right_eye' in stem: cat = 'right_eye'
        else: cat = 'obj'
        
        bbox = None
        if cat in ['obj', 'left_eye', 'right_eye'] and p.with_suffix('.json').exists():
            with open(p.with_suffix('.json'), 'r', encoding="utf-8") as f: bbox = json.load(f).get('bbox')
        char_groups[char_name][cat].append((p, bbox))

    print(f"\n==========================================")
    print(f" 検出キャラ数: {len(char_groups)}")
    print(f"==========================================\n")

    tasks = []

    for char_raw_name, cats in char_groups.items():
        target_folder = output_base / f"1_{char_raw_name}"
        target_folder.mkdir(parents=True, exist_ok=True)

        total_source_items = sum(len(v) for v in cats.values())
        if total_source_items == 0: continue
        
        # 割合の動的再配分 (エッジケースの考慮)
        r_f, r_u, r_fl, r_il = args.ratio_face, args.ratio_upper, args.ratio_full, args.ratio_illust
        r_re, r_le, r_o = args.ratio_right_eye, args.ratio_left_eye, args.ratio_obj
        
        if not cats['obj']: r_il += r_o; r_o = 0.0
        if not cats['illust']: r_u += r_il; r_il = 0.0
        if not cats['left_eye']: r_f += r_le; r_le = 0.0
        if not cats['right_eye']: r_f += r_re; r_re = 0.0
        
        # 目標枚数の算出
        total_target = args.target_total
        targets = {
            'face': int(total_target * (r_f / 100.0)),
            'upper': int(total_target * (r_u / 100.0)),
            'full': int(total_target * (r_fl / 100.0)),
            'illust': int(total_target * (r_il / 100.0)),
            'right_eye': int(total_target * (r_re / 100.0)),
            'left_eye': int(total_target * (r_le / 100.0)),
            'obj': int(total_target * (r_o / 100.0))
        }

        print(f"--- {char_raw_name} (ソース: {total_source_items}枚) ---")
        print(f"    目標配分: 顔={targets['face']} 上半身={targets['upper']} 全身={targets['full']} "
              f"一枚絵={targets['illust']} 目(左/右)={targets['left_eye']}/{targets['right_eye']} 小物={targets['obj']}")

        for cat_name, items in cats.items():
            num_items = len(items)
            if num_items == 0 or targets[cat_name] <= 0: continue
            
            # 均等割り振りのための商と余りを計算
            base_rep = targets[cat_name] // num_items
            remainder = targets[cat_name] % num_items
            
            for idx, (img_path, bbox) in enumerate(items):
                src_txt_path = img_path.with_suffix('.txt')
                
                if args.keep_in_memory:
                    img_data = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
                    if img_data is None: continue
                else:
                    img_data = str(img_path)
                
                # 余りの数だけ、先頭の画像から1枚多く増幅させることで正確な目標枚数に合わせる
                rep = base_rep + (1 if idx < remainder else 0)
                
                for i in range(rep):
                    # ファイル名の衝突を防ぐため、idxも加えて保存名を作成
                    save_p = target_folder / f"{img_path.stem}_aug_{idx+1:03d}_{i+1:03d}.jpg"
                    tasks.append((img_data, cat_name, bg_images_dict, args.jitter_prob, args.blur_prob, bbox, save_p, src_txt_path))
        
    total_tasks = len(tasks)
    if total_tasks > 0:
        print(f"\n--- 3. 画像合成＆キャプション生成 (マルチスレッド) ---")
        print(f"    -> 処理タスク数: {total_tasks}件")
        
        completed = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
            futures = [executor.submit(worker_process, t) for t in tasks]
            
            for future in concurrent.futures.as_completed(futures):
                status, result_data = future.result()
                if status == 1:
                    completed += 1
                    if completed % 100 == 0 or completed == total_tasks:
                        print(f"    進捗: {completed} / {total_tasks} 件完了")
                else:
                    print(f"    [エラー] {result_data}")

    print("\nすべてのデータセット構築処理が完了しました！")

if __name__ == "__main__":
    build_dataset()