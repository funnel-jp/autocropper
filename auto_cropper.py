# file: auto_cropper.py
import cv2
import numpy as np
import os
import sys
import argparse
import shutil
import json
import traceback
import subprocess
import concurrent.futures
from pathlib import Path
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from PIL import Image

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
    parser = argparse.ArgumentParser(description="LoRA学習用 自動切り抜き＆透過維持ツール")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=JSON_DEFAULTS.get("ac_res", 1024))
    parser.add_argument("--margin_top", type=float, default=JSON_DEFAULTS.get("ac_m_top", 0.8))
    parser.add_argument("--margin_bottom", type=float, default=JSON_DEFAULTS.get("ac_m_btm", 2.0))
    parser.add_argument("--margin_side", type=float, default=JSON_DEFAULTS.get("ac_m_side", 1.5))
    parser.add_argument("--obj_margin", type=float, default=JSON_DEFAULTS.get("ac_m_obj", 1.0))
    parser.add_argument("--use_upscaler", action="store_true")
    parser.add_argument("--remove_bg", action="store_true")
    parser.add_argument("--prompt_file", type=str, default="obj_prompts.txt")
    parser.add_argument("--obj_threshold", type=float, default=JSON_DEFAULTS.get("ac_thresh", 0.10))
    
    parser.add_argument("--max_area_ratio", type=float, default=JSON_DEFAULTS.get("ac_max_area", 0.35), help="小物のBBoxが画像全体に占める最大割合")
    parser.add_argument("--iou_threshold", type=float, default=JSON_DEFAULTS.get("ac_iou", 0.50), help="BBoxの重なり許容度 (重複検知防止)")
    parser.add_argument("--phash_threshold", type=int, default=JSON_DEFAULTS.get("ac_phash", 5), help="類似画像と判定するdHashのハミング距離")
    parser.add_argument("--hist_threshold", type=float, default=JSON_DEFAULTS.get("ac_hist", 0.98), help="クロップずれを吸収する色ヒストグラム相関のしきい値(1.0に近いほど厳しい)")
    parser.add_argument("--face_include_threshold", type=float, default=JSON_DEFAULTS.get("ac_face_inc", 0.50), help="小物のBBoxが顔を包含する許容度 (顔を含む誤検知防止)")
    
    parser.add_argument("--truncation_margin", type=int, default=5, help="見切れ判定とする画像端からの距離(px)")
    
    parser.add_argument("--sd_scripts_path", type=str, default="", help="WD14タガー実行用のsd-scriptsパス (未指定ならタグ付けはOpenCVフォールバック)")
    parser.add_argument("--overwrite_tags", action="store_true", help="既存のキャプションファイルを上書きする")
    parser.add_argument("--extract_eyes", action="store_true", help="左右の目を専用ロジックで個別に切り出す")
    parser.add_argument("--filter_head_decorations", action="store_true", help="頭部装飾品が顔より下にある場合を除外する")
    parser.add_argument("--rembg_model", type=str, default=JSON_DEFAULTS.get("ac_rembg_model", "isnet-anime"), help="rembgのセッションモデル名 (setup.pyで選択)")
    return parser.parse_args()

args = get_args()

if os.path.exists(args.output_dir):
    shutil.rmtree(args.output_dir)
os.makedirs(args.output_dir, exist_ok=True)

face_model = None
detector = None
upscaler = None
rmbg_session = None

global_saved_features = []
global_saved_eyes = {}

def convert_single_image(p, cmd_name):
    temp_out = p.with_name(p.stem + "_tmp_conv.png")
    final_out = p.with_suffix(".png")
    try:
        subprocess.run([cmd_name, str(p), "-depth", "8", str(temp_out)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        p.unlink()
        temp_out.rename(final_out)
        return 1
    except Exception as e:
        print(f"  -> [エラー] {p.name} の変換に失敗しました: {e}")
        if temp_out.exists():
            try: temp_out.unlink()
            except: pass
        return 0

def format_images_to_png(input_dir):
    try:
        subprocess.run(["magick", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        cmd_name = "magick"
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("\n[警告] ImageMagickがインストールされていません。画像形式の自動PNG統一処理をスキップします。")
        return

    print("\n--- 入力画像のPNGフォーマット統一を開始 (並列処理) ---")
    image_paths = [p for p in Path(input_dir).glob("*.*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".jfif"}]
    
    converted_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
        futures = [executor.submit(convert_single_image, p, cmd_name) for p in image_paths]
        for future in concurrent.futures.as_completed(futures):
            converted_count += future.result()
                        
    if converted_count > 0: print(f"  -> {converted_count}枚の画像のフォーマット確認・変換が完了しました。")

def run_tagger_for_background_check(input_dir, sd_scripts_path, overwrite_tags):
    if not sd_scripts_path:
        print("\n[情報] sd-scriptsパスが未指定のためタグ付けをスキップし、OpenCV判定にフォールバックします。")
        return
    py_exe = str(Path(sd_scripts_path) / "venv/Scripts/python.exe")
    tagger_script = Path(sd_scripts_path) / "finetune/tag_images_by_wd14_tagger.py"

    setup_dll_path(sd_scripts_path + "/venv")
    
    if Path(tagger_script).exists():
        print("\n--- WD14タガーによる事前タグ付け (背景判定＆キャプション保持用) を開始 ---")
        
        target_dir = input_dir
        tmp_dir = None
        
        if not overwrite_tags:
            input_path = Path(input_dir)
            untagged_images = []
            for ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".jfif"]:
                for img_path in input_path.glob(f"*{ext}"):
                    if not img_path.with_suffix(".txt").exists():
                        untagged_images.append(img_path)
            
            if len(untagged_images) == 0:
                print("  -> [情報] すべての画像にタグファイルが存在するため、推論計算を完全にスキップします。")
                return
                
            print(f"  -> [情報] タグが存在しない未処理の画像 {len(untagged_images)} 枚のみを対象に推論を実行します。")
            
            tmp_dir = input_path / "_tmp_tagging"
            tmp_dir.mkdir(exist_ok=True)
            for img_path in untagged_images:
                shutil.copy(img_path, tmp_dir / img_path.name)
            target_dir = str(tmp_dir.absolute())
        else:
            print("  -> [情報] 全画像に対してタグファイルを上書き再生成します。")

        try:
            env = os.environ.copy()
            subprocess.run([py_exe, str(tagger_script), "--batch_size", "8", "--repo_id", "SmilingWolf/wd-eva02-large-tagger-v3", "--onnx", str(Path(target_dir).absolute())], check=True, cwd=sd_scripts_path, env=env)
            print("  -> 事前タグ付け完了。")
        except Exception as e:
            print(f"  -> [エラー] タガーの実行に失敗しました: {e}")
            
        if tmp_dir and tmp_dir.exists():
            for txt_path in tmp_dir.glob("*.txt"):
                shutil.move(str(txt_path), str(Path(input_dir) / txt_path.name))
            shutil.rmtree(tmp_dir, ignore_errors=True)
            print("  -> [情報] 生成されたタグファイルを元のフォルダに統合しました。")
    else:
        print(f"\n[警告] {tagger_script} が見つかりません。タグベースの背景判定をスキップし、OpenCV判定にフォールバックします。")

def init_models():
    global face_model, detector, upscaler, rmbg_session
    print("\n--- モデルの初期化を開始 ---")
    
    try:
        print("1. YOLOv8顔検出モデルを準備中...")
        face_path = hf_hub_download(repo_id="Bingsu/adetailer", filename="face_yolov8n.pt")
        face_model = YOLO(face_path)
    except Exception as e:
        print(f"顔検出モデルのロードに失敗: {e}")

    if args.use_upscaler:
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            print("2. Real-ESRGANを準備中...")
            model_net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
            esrgan_path = hf_hub_download(repo_id="ac-pill/upscale_models", filename="RealESRGAN_x4plus_anime_6B.pth")
            upscaler = RealESRGANer(scale=4, model_path=esrgan_path, model=model_net, half=True)
        except Exception as e:
            print(f"超解像エンジンのロードに失敗: {e}")

    if args.remove_bg:
        try:
            from rembg import remove, new_session
            print("3. 背景削除モデルを準備中...")
            rmbg_session = new_session(args.rembg_model)
        except Exception as e:
            print(f"背景削除モデルのロードに失敗: {e}")

    if detector is None:
        try:
            from transformers import pipeline
            import torch
            print("4. 物体検知モデル (OWLv2) を準備中...")
            device = 0 if torch.cuda.is_available() else -1
            detector = pipeline(task="zero-shot-object-detection", 
                                model="google/owlv2-base-patch16-ensemble", 
                                device=device)
            print(f"  -> デバイス: {'GPU (cuda:0)' if device == 0 else 'CPU'}")
        except Exception as e:
            print(f"OWLv2のロード中にエラーが発生しました:\n{traceback.format_exc()}")

def parse_tags(txt_path):
    if not txt_path.exists():
        return None, False
        
    with open(txt_path, "r", encoding="utf-8") as f:
        tags = [t.strip().lower() for t in f.read().split(",")]
    
    simple_bg_tags = {
        "simple background", "white background", "transparent background", 
        "grey background", "black background", "blue background", 
        "red background", "green background", "yellow background", "pink background", "purple background",
        "colorful background", "gradient background", "patterned background", 
        "plaid background", "striped background", "sunburst background", "polka dot background", "two-tone background"
    }
    
    is_simple = any(t in simple_bg_tags for t in tags)
    has_glasses = any("glasses" in t for t in tags)
            
    return is_simple, has_glasses

def is_simple_background(img, ratio_threshold=0.40, border_threshold=0.65, bg_std_threshold=15.0, continuous_threshold=0.90):
    small = cv2.resize(img, (256, 256))
    quantized = (small // 32) * 32 
    pixels = quantized.reshape(-1, 3)
    colors, counts = np.unique(pixels, axis=0, return_counts=True)
    
    max_idx = np.argmax(counts)
    max_color = colors[max_idx]
    max_ratio = counts[max_idx] / len(pixels)
    
    border_pixels = np.concatenate([
        quantized[0, :], quantized[-1, :], quantized[1:-1, 0], quantized[1:-1, -1]
    ])
    
    border_match = np.all(border_pixels == max_color, axis=1)
    border_match_ratio = np.sum(border_match) / len(border_pixels)
    
    if max_ratio <= ratio_threshold or border_match_ratio <= border_threshold:
        return False, max_ratio, border_match_ratio

    def get_max_continuous_color_ratio(line_pixels):
        if len(line_pixels) == 0: return 0.0
        diffs = np.any(line_pixels[1:] != line_pixels[:-1], axis=1)
        change_indices = np.where(diffs)[0]
        if len(change_indices) == 0: return 1.0 
        splits = np.split(line_pixels, change_indices + 1)
        max_len = max(len(s) for s in splits)
        return max_len / len(line_pixels)

    top_edge = quantized[0, :]
    bottom_edge = quantized[-1, :]
    left_edge = quantized[:, 0]
    right_edge = quantized[:, -1]

    edge_ratios = [
        get_max_continuous_color_ratio(top_edge),
        get_max_continuous_color_ratio(bottom_edge),
        get_max_continuous_color_ratio(left_edge),
        get_max_continuous_color_ratio(right_edge)
    ]
    
    if max(edge_ratios) < continuous_threshold:
        return False, max_ratio, border_match_ratio

    bg_mask = np.all(quantized == max_color, axis=-1)
    bg_pixels_raw = small[bg_mask]
    
    if len(bg_pixels_raw) > 0:
        std_dev = np.std(bg_pixels_raw, axis=0)
        max_std_dev = np.max(std_dev)
        if max_std_dev > bg_std_threshold:
            return False, max_ratio, border_match_ratio
    
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    bg_edges = edges[bg_mask]
    bg_edge_ratio = np.sum(bg_edges > 0) / bg_edges.size if bg_edges.size > 0 else 1.0

    if bg_edge_ratio > 0.03:
        return False, max_ratio, border_match_ratio

    h_s, w_s = edges.shape
    margin = int(min(h_s, w_s) * 0.05)
    
    top_border_edges = edges[0:margin, :]
    left_border_edges = edges[:, 0:margin]
    right_border_edges = edges[:, -margin:]
    
    side_edges = np.concatenate([
        top_border_edges.flatten(),
        left_border_edges.flatten(),
        right_border_edges.flatten()
    ])
    
    side_edge_ratio = np.sum(side_edges > 0) / side_edges.size
    if side_edge_ratio > 0.015:
        return False, max_ratio, border_match_ratio

    return True, max_ratio, border_match_ratio

def crop_with_padding(img, y1, y2, x1, x2):
    h, w = img.shape[:2]
    
    # 画面外にはみ出したマージン量を計算
    pad_top = max(0, -y1)
    pad_bottom = max(0, y2 - h)
    pad_left = max(0, -x1)
    pad_right = max(0, x2 - w)
    
    # 元画像内で実際にクロップ可能な座標
    src_y1 = max(0, y1)
    src_y2 = min(h, y2)
    src_x1 = max(0, x1)
    src_x2 = min(w, x2)
    
    cropped = img[src_y1:src_y2, src_x1:src_x2]
    
    # パディングが必要な場合のみ追加
    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        if len(img.shape) == 3 and img.shape[2] == 4:
            pad_color = (0, 0, 0, 0) # 透過背景の場合は透明で埋める
        else:
            # 単色背景の場合は左上のピクセルの色で埋める
            pad_color = img[0, 0].tolist()
            
        cropped = cv2.copyMakeBorder(
            cropped, pad_top, pad_bottom, pad_left, pad_right, 
            cv2.BORDER_CONSTANT, value=pad_color
        )
        
    return cropped

def get_symmetric_crop(fx1, fy1, fx2, fy2, m_top_px, m_btm_px, m_side_px, img_w, img_h):
    # 理想のマージン座標
    x1 = fx1 - m_side_px
    x2 = fx2 + m_side_px
    y1 = fy1 - m_top_px
    y2 = fy2 + m_btm_px
    
    # はみ出し量を計算
    over_l = max(0, -x1)
    over_r = max(0, x2 - img_w)
    
    # 左右を対称に切り詰める (ズーム)
    if over_l > 0 or over_r > 0:
        shrink_x = max(over_l, over_r)
        shrink_x = min(shrink_x, m_side_px) # 被写体自体(fx1, fx2)には食い込まないように保護
        x1 += shrink_x
        x2 -= shrink_x
        
    # 上下を対称に切り詰める (ズーム)
    over_t = max(0, -y1)
    over_b = max(0, y2 - img_h)
    if over_t > 0 or over_b > 0:
        shrink_y = max(over_t, over_b)
        shrink_y = min(shrink_y, m_top_px, m_btm_px) # 被写体自体には食い込まないように保護
        y1 += shrink_y
        y2 -= shrink_y
        
    # 切り詰めてもなお画面外に出る場合（画像自体が小さすぎる等）は境界で最終クリップ
    x1 = max(0, x1)
    x2 = min(img_w, x2)
    y1 = max(0, y1)
    y2 = min(img_h, y2)
    
    return int(x1), int(y1), int(x2), int(y2)

def resize_by_area(image, target_resolution):
    h, w = image.shape[:2]
    target_area = target_resolution * target_resolution
    current_area = h * w
    
    import math
    scale = math.sqrt(target_area / current_area)
    
    if scale < 1.0:
        scale = 1.0
    
    if scale > 1.0 and upscaler is not None:
        try:
            if image.shape[2] == 4:
                bgr = image[:, :, :3]
                mask = image[:, :, 3]
                up_bgr, _ = upscaler.enhance(bgr, outscale=scale)
                up_mask = cv2.resize(mask, (up_bgr.shape[1], up_bgr.shape[0]), interpolation=cv2.INTER_LANCZOS4)
                return cv2.merge([up_bgr[:,:,0], up_bgr[:,:,1], up_bgr[:,:,2], up_mask]), scale
            else:
                output, _ = upscaler.enhance(image, outscale=scale)
                return output, scale
        except Exception: pass
        
    if scale == 1.0:
        return image, scale
        
    interp = cv2.INTER_LANCZOS4
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=interp), scale

def calculate_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    denominator = float(boxAArea + boxBArea - interArea)
    return interArea / denominator if denominator > 0 else 0.0

def calc_dhash(image, hash_size=8):
    if image is None or image.size == 0:
        return np.zeros((hash_size * hash_size,), dtype=bool)
    if len(image.shape) == 3:
        if image.shape[2] == 4:
            alpha = image[:, :, 3] / 255.0
            bg = np.ones_like(image[:, :, :3]) * 255
            bgr = (image[:, :, :3] * alpha[:, :, np.newaxis] + bg * (1 - alpha[:, :, np.newaxis])).astype(np.uint8)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    resized = cv2.resize(gray, (hash_size + 1, hash_size))
    diff = resized[:, 1:] > resized[:, :-1]
    return diff.flatten()

def hamming_distance(hash1, hash2):
    return np.count_nonzero(hash1 != hash2)

def calc_histogram(image):
    if len(image.shape) == 3:
        if image.shape[2] == 4:
            mask = image[:, :, 3]
            bgr = image[:, :, :3]
        else:
            mask = None
            bgr = image
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    else:
        mask = None
        hsv = image
        
    hist = cv2.calcHist([hsv], [0, 1], mask, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist

def process_image(img_path):
    global global_saved_features
    global global_saved_eyes
    
    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if img is None: return
    base_name = img_path.stem
    print(f"\n--- 処理中: {img_path.name} ---")

    txt_path = img_path.with_suffix('.txt')
    tag_is_simple, has_glasses = parse_tags(txt_path)
    
    if img.dtype == np.uint16:
        print("  -> [情報] 16-bit画像を検出しました。8-bitに変換します。")
        img = (img / 256).astype(np.uint8)

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if img.shape[2] == 4:
        if np.all(img[:, :, 3] == 255):
            print("  -> [情報] 透過情報を持たないRGBA画像を検出しました。RGB画像として判定フローに回します。")
            img = img[:, :, :3]

    is_simple_bg = True

    if img.shape[2] != 4:
        if tag_is_simple is not None:
            is_simple_bg = tag_is_simple
            if is_simple_bg:
                print("  -> [背景判定(タガー)] 立ち絵系タグを検出: 単色背景(立ち絵)と判定しました。")
            else:
                print("  -> [背景判定(タガー)] 風景系/背景タグを検出: 複雑な背景(一枚絵)と判定しました。")
        else:
            is_simple_bg, max_ratio, border_ratio = is_simple_background(img)
            print(f"  -> [背景判定(OpenCV)] 最頻色: {max_ratio:.1%}, 外周一致: {border_ratio:.1%}")
        
        if is_simple_bg:
            if args.remove_bg and rmbg_session is not None:
                from rembg import remove
                print("  -> 単色背景(立ち絵)と判定: 背景を透過処理します...")
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.cvtColor(remove(img_rgb, session=rmbg_session), cv2.COLOR_RGBA2BGRA)
            else:
                print("  -> 単色背景(立ち絵)と判定: (背景除去オプションOFF)")
        else:
            print("  -> 複雑な背景(一枚絵)と判定: 透過処理および部分切り抜きをスキップします。")
    else:
        print("  -> [背景判定] 透過画像: 立ち絵として処理します。")

    # 元画像の透過オートトリム (全体に対する処理)
    if is_simple_bg and len(img.shape) == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3]
        coords = cv2.findNonZero((alpha > 10).astype(np.uint8))
        if coords is not None:
            x, y, trim_w, trim_h = cv2.boundingRect(coords)
            if trim_w < img.shape[1] or trim_h < img.shape[0]:
                img = img[y:y+trim_h, x:x+trim_w]
                print(f"  -> [情報] 透明な余白領域をトリミングしました (新サイズ: {trim_w}x{trim_h})")

    if not is_simple_bg:
        full_crop, _ = resize_by_area(img, args.resolution)
        cv2.imwrite(os.path.join(args.output_dir, f"{base_name}_illust.png"), full_crop)
        
        if txt_path.exists():
            shutil.copy(txt_path, os.path.join(args.output_dir, f"{base_name}_illust.txt"))
            
        return

    if has_glasses:
        print("  -> [情報] 眼鏡着用(glassesタグ)を検出しました。眼鏡ありの目として個別に選出します。")

    h, w = img.shape[:2]
    img_area = h * w
    if img.shape[2] == 4:
        alpha = img[:,:,3]/255.0
        detect_img = (img[:,:,:3]*alpha[:,:,np.newaxis] + 255*(1-alpha[:,:,np.newaxis])).astype(np.uint8)
    else:
        detect_img = img

    face_boxes = []
    if face_model:
        results = face_model(detect_img, verbose=False)
        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                face_boxes.append(box.xyxy[0].cpu().numpy())
            
            box = face_boxes[0]
            fx1, fy1, fx2, fy2 = map(int, box)
            fw_box, fh_box = fx2-fx1, fy2-fy1
            
            # _face: 顔アップは余白が消えすぎない範囲で対称にズーム（従来通り）
            face_m_top = int(fh_box * (args.margin_top * 0.6))
            face_m_btm = int(fh_box * 0.4) 
            face_m_side = int(fw_box * (args.margin_side * 0.35))
            
            face_x1, face_y1, face_x2, face_y2 = get_symmetric_crop(
                fx1, fy1, fx2, fy2, face_m_top, face_m_btm, face_m_side, w, h
            )
            face_crop, _ = resize_by_area(img[face_y1:face_y2, face_x1:face_x2], args.resolution)
            cv2.imwrite(os.path.join(args.output_dir, f"{base_name}_face.png"), face_crop)
            
            # _upper: 上半身も対称にズーム（従来通り）
            upper_m_top = int(fh_box * args.margin_top)
            upper_m_btm = int(fh_box * args.margin_bottom)
            upper_m_side = int(fw_box * args.margin_side)
            
            upper_x1, upper_y1, upper_x2, upper_y2 = get_symmetric_crop(
                fx1, fy1, fx2, fy2, upper_m_top, upper_m_btm, upper_m_side, w, h
            )
            upper_crop, _ = resize_by_area(img[upper_y1:upper_y2, upper_x1:upper_x2], args.resolution)
            cv2.imwrite(os.path.join(args.output_dir, f"{base_name}_upper.png"), upper_crop)
            
            # _full (全体)
            full_crop, _ = resize_by_area(img, args.resolution)
            cv2.imwrite(os.path.join(args.output_dir, f"{base_name}_full.png"), full_crop)

            if txt_path.exists():
                shutil.copy(txt_path, os.path.join(args.output_dir, f"{base_name}_full.txt"))

    if detector:
        prompts = []
        if os.path.exists(args.prompt_file):
            with open(args.prompt_file, "r", encoding="utf-8") as f:
                prompts = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        
        if args.extract_eyes:
            if not any(p.lower() == "eye" for p in prompts):
                prompts.append("eye")
                
        if prompts:
            pil_detect = Image.fromarray(cv2.cvtColor(detect_img, cv2.COLOR_BGR2RGB))
            try:
                predictions = detector(pil_detect, candidate_labels=prompts, threshold=args.obj_threshold)
                predictions = sorted(predictions, key=lambda x: x.get("score", 0.0), reverse=True)
                kept_predictions = []
                
                best_left_eye_box = None
                best_right_eye_box = None
                
                for pred in predictions:
                    if pred["label"].lower() == "eye":
                        b = pred["box"]
                        eye_cx = (b["xmin"] + b["xmax"]) / 2.0
                        eye_cy = (b["ymin"] + b["ymax"]) / 2.0
                        
                        target_face = None
                        for fb in face_boxes:
                            if fb[0] <= eye_cx <= fb[2] and fb[1] <= eye_cy <= fb[3]:
                                target_face = fb
                                break
                        
                        if target_face is not None:
                            face_cx = (target_face[0] + target_face[2]) / 2.0
                        elif len(face_boxes) > 0:
                            face_cx = (face_boxes[0][0] + face_boxes[0][2]) / 2.0
                        else:
                            face_cx = w / 2.0
                            
                        if eye_cx < face_cx:
                            pred["label"] = "left_eye"
                            if best_left_eye_box is None:
                                best_left_eye_box = b
                        else:
                            pred["label"] = "right_eye"
                            if best_right_eye_box is None:
                                best_right_eye_box = b
                
                for i, pred in enumerate(predictions):
                    label = pred["label"]
                    b = pred["box"]
                    ox1, oy1, ox2, oy2 = b["xmin"], b["ymin"], b["xmax"], b["ymax"]
                    
                    t_margin = args.truncation_margin
                    is_truncated = (ox1 <= t_margin) or (oy1 <= t_margin) or (ox2 >= w - t_margin) or (oy2 >= h - t_margin)
                    
                    # 💡 [修正] _obj: get_symmetric_cropは使わず、はみ出しパディングを前提とした広めの座標を一旦保持
                    m_w = int((ox2-ox1)*args.obj_margin)
                    m_h = int((oy2-oy1)*args.obj_margin)
                    cx1, cy1, cx2, cy2 = ox1 - m_w, oy1 - m_h, ox2 + m_w, oy2 + m_h

                    box_area = (ox2 - ox1) * (oy2 - oy1)
                    reject_reason = None
                    
                    if box_area / img_area > args.max_area_ratio:
                        reject_reason = "area"
                    else:
                        for kept_pred in kept_predictions:
                            if kept_pred["status"] == "kept" and kept_pred["pred"]["label"] == label:
                                kb = kept_pred["pred"]["box"]
                                iou = calculate_iou([ox1, oy1, ox2, oy2], [kb["xmin"], kb["ymin"], kb["xmax"], kb["ymax"]])
                                if iou > args.iou_threshold:
                                    reject_reason = "iou"
                                    break
                    
                    is_eye = label in ["left_eye", "right_eye"]

                    if reject_reason is None and not is_eye:
                        for fb in face_boxes:
                            face_area = (fb[2] - fb[0]) * (fb[3] - fb[1])
                            xA, yA = max(ox1, fb[0]), max(oy1, fb[1])
                            xB, yB = min(ox2, fb[2]), min(oy2, fb[3])
                            interArea = max(0, xB - xA) * max(0, yB - yA)
                            if face_area > 0 and (interArea / face_area) > args.face_include_threshold:
                                reject_reason = "face_included"
                                break

                    if reject_reason is None and args.filter_head_decorations and not is_eye:
                        label_lower = label.lower().replace('_', ' ')
                        head_keywords = ["hair", "head", "hat", "cap", "crown", "tiara", "halo", "horn", "ear", "glasses", "eyepatch", "goggles"]
                        if any(kw in label_lower for kw in head_keywords):
                            obj_cx = (ox1 + ox2) / 2.0
                            obj_cy = (oy1 + oy2) / 2.0
                            target_face = None
                            min_dist = float('inf')
                            for fb in face_boxes:
                                face_cx = (fb[0] + fb[2]) / 2.0
                                dist = abs(obj_cx - face_cx)
                                if dist < min_dist:
                                    min_dist = dist
                                    target_face = fb
                            if target_face is not None:
                                face_bottom = target_face[3]
                                face_height = target_face[3] - target_face[1]
                                limit_y = face_bottom + (face_height * 0.50)
                                if obj_cy > limit_y:
                                    reject_reason = "not_on_head"

                    if reject_reason is None and not is_eye and (best_left_eye_box is not None and best_right_eye_box is not None):
                        le_cx = (best_left_eye_box["xmin"] + best_left_eye_box["xmax"]) / 2.0
                        le_cy = (best_left_eye_box["ymin"] + best_left_eye_box["ymax"]) / 2.0
                        re_cx = (best_right_eye_box["xmin"] + best_right_eye_box["xmax"]) / 2.0
                        re_cy = (best_right_eye_box["ymin"] + best_right_eye_box["ymax"]) / 2.0
                        
                        if (cx1 <= le_cx <= cx2 and cy1 <= le_cy <= cy2) and (cx1 <= re_cx <= cx2 and cy1 <= re_cy <= cy2):
                            reject_reason = "both_eyes_included"

                    if reject_reason is None and is_truncated:
                        reject_reason = "truncated"

                    kept_predictions.append({
                        "pred": pred, "status": "kept" if reject_reason is None else "rejected",
                        "reason": reject_reason, "original_index": i,
                        "is_truncated": is_truncated,
                        "crop_coords": (cx1, cy1, cx2, cy2)
                    })

                for item in kept_predictions:
                    pred = item["pred"]
                    label = pred["label"]
                    b = pred["box"]
                    ox1, oy1, ox2, oy2 = b["xmin"], b["ymin"], b["xmax"], b["ymax"]
                    cx1, cy1, cx2, cy2 = item["crop_coords"]
                    
                    # 💡 [新規] 小物のオートトリム＆強制パディング処理
                    
                    # 1. パディングを許容して広めにクロップ
                    obj_crop_raw = crop_with_padding(img, cy1, cy2, cx1, cx2)
                    
                    rel_ox1 = ox1 - cx1
                    rel_oy1 = oy1 - cy1
                    rel_ox2 = ox2 - cx1
                    rel_oy2 = oy2 - cy1
                    
                    # 2. 透過領域のオートトリム (不要な背景を完全に削る)
                    if len(obj_crop_raw.shape) == 3 and obj_crop_raw.shape[2] == 4:
                        alpha = obj_crop_raw[:, :, 3]
                        coords = cv2.findNonZero((alpha > 10).astype(np.uint8))
                        if coords is not None:
                            tx, ty, tw, th = cv2.boundingRect(coords)
                            obj_crop_raw = obj_crop_raw[ty:ty+th, tx:tx+tw]
                            # BBox座標もシフト
                            rel_ox1 -= tx
                            rel_oy1 -= ty
                            rel_ox2 -= tx
                            rel_oy2 -= ty
                            
                    # 3. AIのエッジタッチ判定を回避するための強制パディング (必ず15%の余白を足す)
                    c_h, c_w = obj_crop_raw.shape[:2]
                    pad_y = max(10, int(c_h * 0.15))
                    pad_x = max(10, int(c_w * 0.15))
                    
                    if len(obj_crop_raw.shape) == 3 and obj_crop_raw.shape[2] == 4:
                        pad_color = (0, 0, 0, 0)
                    else:
                        pad_color = obj_crop_raw[0, 0].tolist()
                        
                    obj_padded = cv2.copyMakeBorder(obj_crop_raw, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=pad_color)
                    rel_ox1 += pad_x
                    rel_oy1 += pad_y
                    rel_ox2 += pad_x
                    rel_oy2 += pad_y
                    
                    # 4. 目標解像度にリサイズ
                    obj_crop, scale = resize_by_area(obj_padded, args.resolution)
                    reject_reason = item["reason"]
                    
                    if item["status"] == "kept":
                        current_hash = calc_dhash(obj_crop)
                        current_hist = calc_histogram(obj_crop)
                        
                        for saved_label, shash, shist in global_saved_features:
                            if saved_label == label:
                                if hamming_distance(current_hash, shash) <= args.phash_threshold:
                                    item["status"] = "rejected"
                                    reject_reason = "phash"
                                    break
                                    
                                if cv2.compareHist(current_hist, shist, cv2.HISTCMP_CORREL) >= args.hist_threshold:
                                    item["status"] = "rejected"
                                    reject_reason = "hist_sim"
                                    break

                        if item["status"] == "kept" and args.extract_eyes and label in ["left_eye", "right_eye"]:
                            char_name = base_name.split("_")[0]
                            eye_key = f"{char_name}_glasses" if has_glasses else f"{char_name}_noglasses"
                            
                            if eye_key not in global_saved_eyes:
                                global_saved_eyes[eye_key] = set()
                                
                            if label in global_saved_eyes[eye_key]:
                                item["status"] = "rejected"
                                reject_reason = "eye_limit"
                            else:
                                global_saved_eyes[eye_key].add(label)
                                
                        if item["status"] == "kept":
                            global_saved_features.append((label, current_hash, current_hist))
                    
                    save_name_base = f"{base_name}_{label.replace(' ','_')}_{item['original_index']+1:02d}"
                    
                    obj_cx = (ox1 + ox2) / 2.0
                    obj_cy = (oy1 + oy2) / 2.0
                    rel_cx = round(obj_cx / w, 3)
                    rel_cy = round(obj_cy / h, 3)
                    
                    bbox_data = {
                        "label": label, 
                        "bbox": [int(rel_ox1 * scale), int(rel_oy1 * scale), int(rel_ox2 * scale), int(rel_oy2 * scale)], 
                        "original_center_rel": [rel_cx, rel_cy],
                        "status": item["status"], 
                        "reason": reject_reason,
                        "is_truncated": bool(item["is_truncated"])
                    }
                    
                    if item["status"] == "rejected":
                        save_name = f"{save_name_base}_del_{reject_reason}"
                        save_dir = os.path.join(args.output_dir, "filtered")
                        os.makedirs(save_dir, exist_ok=True)
                        print(f"  -> [Filter] 除外({reject_reason}): 隔離フォルダへ移動 -> {save_name}.png")
                        cv2.imwrite(os.path.join(save_dir, f"{save_name}.png"), obj_crop)
                        with open(os.path.join(save_dir, f"{save_name}.json"), "w", encoding="utf-8") as f:
                            json.dump(bbox_data, f, ensure_ascii=False, indent=4)
                    else:
                        save_name = save_name_base
                        save_dir = args.output_dir
                        cv2.imwrite(os.path.join(save_dir, f"{save_name}.png"), obj_crop)
                        with open(os.path.join(save_dir, f"{save_name}.json"), "w", encoding="utf-8") as f:
                            json.dump(bbox_data, f, ensure_ascii=False, indent=4)

            except Exception as e:
                print(f"  -> 小物検知実行中にエラー: {e}")

if __name__ == "__main__":
    format_images_to_png(args.input_dir)
    run_tagger_for_background_check(args.input_dir, args.sd_scripts_path, args.overwrite_tags)
    
    init_models()
    image_paths = list(Path(args.input_dir).glob("*.*"))
    for p in image_paths:
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            process_image(p)
    print("\nすべての処理が完了しました。")