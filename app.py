# app.py
import urllib.error
import gradio as gr
import subprocess
import sys
import os
import queue
import shutil
import urllib.parse
import time
import json
import threading
import traceback
from pathlib import Path

# 分割したAI判定モジュールをインポート
from ai_selector import run_ai_selection_gen

DEFAULT_INPUT_DIR = "input_images"
DEFAULT_OUTPUT_DIR = "output_images"
DEFAULT_TRAIN_DIR = ""
SD_SCRIPTS_PATH = ""

PROMPTS_FILE = "obj_prompts.txt"
TAG_SETTINGS_FILE = "tag_settings.txt"
AI_PROMPT_FILE = "ai_prompt.txt"
SETTINGS_FILE = "settings.json"
CHAR_TAG_RULES_FILE = "char_tag_rules.json"

DEFAULT_TAG_SETTINGS = """# 自動整理設定
# 追加したいタグ (カンマ区切り)
ADD: 
# 削除したいタグ（部分一致/単語境界：指定単語を含むタグを一掃します）
REMOVE_PARTIAL: eye, hair, brow, nose, teeth, ear, skin, ribbon, bow
# 削除したいタグ（完全一致：指定タグと完全に一致する場合のみ削除）
REMOVE_EXACT: left eye, right eye
# 絶対に残したいタグ（部分一致/単語境界：Removeより優先されます）
KEEP_PARTIAL: brown, hair ornament, hairband, hairpin, earrings
# 絶対に残したいタグ（完全一致）
KEEP_EXACT: eye focus
"""

DEFAULT_SETTINGS = {
    "ac_input": DEFAULT_INPUT_DIR, "ac_output": DEFAULT_OUTPUT_DIR, "te_sd_path_s1": SD_SCRIPTS_PATH,
    "ac_res": 1024, "ac_thresh": 0.15,
    "ac_rembg": True, "ac_upscale": True, "ac_extract_eyes": True, "ac_filter_head": True,
    "ac_rembg_model": "isnet-anime",
    "ac_m_top": 1.0, "ac_m_btm": 1.5, "ac_m_side": 1.5, "ac_m_obj": 1.0,
    "ac_max_area": 0.35, "ac_iou": 0.50, "ac_face_inc": 0.50, "ac_phash": 5, "ac_hist": 0.98,
    "ac_prompts": "hair ornament\nhairband",
    
    "man_target_dir": DEFAULT_OUTPUT_DIR,
    "man_filter": ["小物 (_obj)"],
    "man_columns": 4,
    "ai_api_url": "http://localhost:11434",
    "ai_model_name": "qwen3.5:9b",
    "ai_translate_model": "",
    "ai_search_more": False,
    "ai_prompt": "You are a professional image quality assessor...",
    
    "te_target_dir_inp": DEFAULT_OUTPUT_DIR,
    "te_tags_config": DEFAULT_TAG_SETTINGS,
    
    "db_input": DEFAULT_OUTPUT_DIR, "db_output": DEFAULT_TRAIN_DIR,
    "db_bg": "bg_images", "db_sd_inp": SD_SCRIPTS_PATH,
    "db_total": 500, "db_jitter": 0.15, "db_blur": 0.8, "db_keep_in_memory": True,
    "db_ratio_face": 40.0, "db_ratio_upper": 30.0, "db_ratio_full": 15.0, "db_ratio_illust": 10.0,
    "db_ratio_reye": 0.0, "db_ratio_leye": 5.0, "db_ratio_obj": 0.0
}

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
                result = DEFAULT_SETTINGS.copy()
                result.update(saved)
                return result
        except:
            pass
    return DEFAULT_SETTINGS.copy()

def save_settings(settings_dict):
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings_dict, f, ensure_ascii=False, indent=4)
    except: pass

def update_setting(key, value):
    settings = load_settings()
    settings[key] = value
    save_settings(settings)

current_settings = load_settings()
log_queue = queue.Queue()
log_history = ""

def get_utf8_env():
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env

def read_logs():
    global log_history
    has_new = False
    while not log_queue.empty():
        try:
            log_history += log_queue.get_nowait()
            has_new = True
        except queue.Empty: break
    return log_history if has_new else gr.update()

def save_text_file(path, content):
    with open(path, "w", encoding="utf-8") as f: f.write(content)

def start_combined_process_gen(input_dir, output_dir, resolution, remove_bg, use_upscaler, obj_threshold, m_top, m_bottom, m_side, m_obj, prompts_content, max_area, iou, phash, hist, face_inc, overwrite_tags, extract_eyes, filter_head, sd_path):
    save_text_file(PROMPTS_FILE, prompts_content)
    log_queue.put("🚀 Phase 1: 素材切り出しを開始...\n")
    yield gr.update(value="⏳ 処理中 (Crop)...", variant="secondary", interactive=False)
    
    # 実行するスクリプトの絶対パスを解決し、sys.executableを使用
    base_dir = Path(__file__).resolve().parent
    script_crop = str(base_dir / "auto_cropper.py")
    
    cmd_crop = [
        sys.executable, script_crop,
        "--input_dir", input_dir,
        "--output_dir", output_dir,
        "--resolution", str(resolution),
        "--obj_threshold", str(obj_threshold),
        "--margin_top", str(m_top),
        "--margin_bottom", str(m_bottom),
        "--margin_side", str(m_side),
        "--obj_margin", str(m_obj),
        "--prompt_file", PROMPTS_FILE,
        "--max_area_ratio", str(max_area),
        "--iou_threshold", str(iou),
        "--phash_threshold", str(phash),
        "--hist_threshold", str(hist),
        "--face_include_threshold", str(face_inc),
        "--sd_scripts_path", str(sd_path)
    ]
    if remove_bg: cmd_crop.append("--remove_bg")
    if use_upscaler: cmd_crop.append("--use_upscaler")
    if overwrite_tags: cmd_crop.append("--overwrite_tags")
    if extract_eyes: cmd_crop.append("--extract_eyes")
    if filter_head: cmd_crop.append("--filter_head_decorations")
    
    try:
        process = subprocess.Popen(
            cmd_crop,
            cwd=str(base_dir),
            env=get_utf8_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        
        def reader():
            for line in process.stdout:
                # フィルタリングせず、サブプロセスの全標準出力を通知キューへ転送
                log_queue.put(line)
                
        t = threading.Thread(target=reader, daemon=True)
        t.start()
        
        while process.poll() is None:
            time.sleep(0.5)
            yield gr.update()
        t.join()
        
        if process.returncode != 0:
            log_queue.put(f"\n❌ クロップ処理が異常終了しました (終了コード: {process.returncode})\n")
            gr.Warning(f"クロップ処理に失敗しました (コード: {process.returncode})")
            yield gr.update(value="🚀 処理を開始", variant="primary", interactive=True)
            return
    except Exception as e:
        # Popen自体の起動失敗（WinError等）をスタックトレース付きで出力
        err_detail = traceback.format_exc()
        log_queue.put(f"⚠️ サブプロセス起動エラー:\n{err_detail}\n")
        gr.Warning(f"実行エラー: {e}")
        yield gr.update(value="🚀 処理を開始", variant="primary", interactive=True)
        return

    log_queue.put("\n🚀 Phase 2: 自動タグ再生成を開始...\n")
    yield gr.update(value="⏳ 処理中 (Retag)...")
    
    script_retag = str(base_dir / "retagger.py")
    cmd_retag = [
        sys.executable, script_retag,
        "--input_dir", output_dir,
        "--sd_scripts_path", sd_path
    ]
    try:
        process = subprocess.Popen(
            cmd_retag,
            cwd=str(base_dir),
            env=get_utf8_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        
        def reader2():
            for line in process.stdout:
                log_queue.put(line)
                
        t2 = threading.Thread(target=reader2, daemon=True)
        t2.start()
        
        while process.poll() is None:
            time.sleep(0.5)
            yield gr.update()
        t2.join()
        
        if process.returncode != 0:
            log_queue.put(f"\n❌ タグ再生成が異常終了しました (終了コード: {process.returncode})\n")
            gr.Warning(f"タグ再生成に失敗しました (コード: {process.returncode})")
        else:
            log_queue.put("✅ 自動工程完了\n")
            gr.Info("自動処理が完了しました。")
    except Exception as e: 
        err_detail = traceback.format_exc()
        log_queue.put(f"⚠️ サブプロセス起動エラー:\n{err_detail}\n")
        gr.Warning(f"実行エラー: {e}")
        
    yield gr.update(value="🚀 処理を開始", variant="primary", interactive=True)

def start_cleaner_gen(input_dir, tag_settings_content):
    save_text_file(TAG_SETTINGS_FILE, tag_settings_content)
    log_queue.put("🚀 タグ整理を実行中...\n")
    yield gr.update(value="⏳ 処理中...", variant="secondary", interactive=False)
    cmd = [sys.executable, "tag_cleaner.py", "--input_dir", input_dir, "--tag_settings", TAG_SETTINGS_FILE]
    try:
        process = subprocess.Popen(cmd, env=get_utf8_env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')
        out_lines = []
        def reader():
            for line in process.stdout:
                out_lines.append(line)
                if "[エラー]" in line or "[警告]" in line:
                    log_queue.put(f"⚠️ {line.strip()}\n")
                    
        t = threading.Thread(target=reader)
        t.start()
        
        while process.poll() is None:
            time.sleep(0.5)
            yield gr.update()
            
        t.join()
        
        if process.returncode != 0:
            err = next((l.strip() for l in out_lines if "[エラー]" in l or "Error" in l or "Exception" in l), f"終了コード: {process.returncode}")
            log_queue.put(f"❌ タグ整理が異常終了しました: {err}\n")
            gr.Warning(f"エラーが発生し、処理が中断されました:\n{err}")
        else:
            log_queue.put("✅ 完了\n")
            gr.Info("タグの一括整理が正常に完了しました。")
            
    except Exception as e: 
        log_queue.put(f"⚠️ 実行エラー: {e}\n")
        gr.Warning(f"実行に失敗しました: {e}")
        
    yield gr.update(value="⚙️ タグ一括整理を実行", variant="primary", interactive=True)

def start_builder_gen(input_dir, output_dir, bg_dir, target_total, jitter, blur, keep_in_memory, ratio_face, ratio_upper, ratio_full, ratio_illust, ratio_reye, ratio_leye, ratio_obj, sd_path):
    if not output_dir or not str(output_dir).strip():
        log_queue.put("⚠️ 出力先ディレクトリ (学習データ出力) が未指定のため中止しました。\n")
        gr.Warning("学習データセットの出力先ディレクトリを指定してください。")
        yield gr.update(value="✨ 学習データセットを生成", variant="primary", interactive=True)
        return
    log_queue.put("🚀 Dataset Builder 開始...\n")
    yield gr.update(value="⏳ 処理中...", variant="secondary", interactive=False)
    cmd = [
        sys.executable, "dataset_builder.py", 
        "--input_dir", input_dir, "--output_dir", output_dir, "--bg_dir", bg_dir, 
        "--target_total", str(int(target_total)), "--jitter_prob", str(jitter), "--blur_prob", str(blur), 
        "--ratio_face", str(ratio_face), "--ratio_upper", str(ratio_upper), 
        "--ratio_full", str(ratio_full), "--ratio_illust", str(ratio_illust), 
        "--ratio_right_eye", str(ratio_reye), "--ratio_left_eye", str(ratio_leye), 
        "--ratio_obj", str(ratio_obj), "--sd_scripts_path", sd_path
    ]
    if keep_in_memory: cmd.append("--keep_in_memory")
    try:
        process = subprocess.Popen(cmd, env=get_utf8_env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')
        out_lines = []
        def reader():
            for line in process.stdout:
                out_lines.append(line)
                if "[エラー]" in line or "[警告]" in line:
                    log_queue.put(f"⚠️ {line.strip()}\n")
        t = threading.Thread(target=reader)
        t.start()
        while process.poll() is None: time.sleep(0.5); yield gr.update()
        t.join()
        
        if process.returncode != 0:
            err = next((l.strip() for l in out_lines if "[エラー]" in l or "Error" in l), f"コード {process.returncode}")
            log_queue.put(f"❌ 構築に失敗しました: {err}\n")
            gr.Warning(f"データセット構築に失敗しました:\n{err}")
        else:
            log_queue.put("✅ 完了\n")
            gr.Info("データセット生成が完了しました！")
    except Exception as e: 
        log_queue.put(f"⚠️ 実行エラー: {e}\n")
        gr.Warning(f"実行に失敗しました: {e}")
    yield gr.update(value="✨ 学習データセットを生成", variant="primary", interactive=True)

def get_filtered_paths(target_dir, selected_filters):
    all_paths = [str(p) for p in Path(target_dir).glob("*.*") if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]]
    all_paths.sort()
    filtered = []
    for p in all_paths:
        stem = Path(p).stem
        is_illust = "_illust" in stem
        is_full = "_full" in stem
        is_upper = "_upper" in stem
        is_face = "_face" in stem
        is_eye = "eye" in stem
        is_obj = not (is_illust or is_full or is_upper or is_face or is_eye)
        if "一枚絵 (_illust)" in selected_filters and is_illust: filtered.append(p)
        elif "全身 (_full)" in selected_filters and is_full: filtered.append(p)
        elif "上半身 (_upper)" in selected_filters and is_upper: filtered.append(p)
        elif "顔 (_face)" in selected_filters and is_face: filtered.append(p)
        elif "目 (_eye)" in selected_filters and is_eye: filtered.append(p)
        elif "小物 (_obj)" in selected_filters and is_obj: filtered.append(p)
    return filtered

def load_manual_selection(target_dir, columns, selected_filters):
    image_paths = get_filtered_paths(target_dir, selected_filters)
    gallery_items = [(p, Path(p).name) for p in image_paths]
    return gr.update(value=gallery_items, columns=columns), image_paths, None, "", -1

def on_gallery_select(evt: gr.SelectData, image_paths, ai_captions):
    if not image_paths or evt.index >= len(image_paths): return None, "", -1
    img_path = image_paths[evt.index]
    filename = Path(img_path).name
    if ai_captions and img_path in ai_captions: filename += f"\n\n🤖 {ai_captions[img_path]}"
    return img_path, filename, evt.index

def delete_selected_image(target_dir, current_idx, image_paths, ai_captions, columns):
    if current_idx < 0 or current_idx >= len(image_paths): return gr.update(), gr.update(), "未選択", current_idx, image_paths, ai_captions
    img_path = Path(image_paths[current_idx])
    for ext in [".txt", ".json", ".npz"]:
        related = img_path.with_suffix(ext)
        if related.exists(): related.unlink()
    if img_path.exists(): img_path.unlink()
    removed_path_str = str(img_path)
    if removed_path_str in ai_captions: del ai_captions[removed_path_str]
    image_paths.pop(current_idx)
    if not image_paths: return [], None, "該当する画像がありません", -1, [], {}
    next_idx = min(current_idx, len(image_paths) - 1)
    next_img = image_paths[next_idx]
    gallery_items = [(p, ai_captions.get(p, Path(p).name)) for p in image_paths]
    next_fn = Path(next_img).name + (f"\n\n🤖 {ai_captions[next_img]}" if next_img in ai_captions else "")
    return gr.update(value=gallery_items, columns=columns), next_img, next_fn, next_idx, image_paths, ai_captions

def rescue_selected_image(current_idx, image_paths, ai_captions, columns):
    if current_idx < 0 or current_idx >= len(image_paths): return gr.update(), gr.update(), ai_captions
    img_path = str(image_paths[current_idx])
    caption = ai_captions.get(img_path, "")
    if caption.startswith("[NG]"):
        new_caption = caption.replace("[NG]", "[OK] 手動保護:", 1)
        ai_captions[img_path] = new_caption
        gallery_items = [(p, ai_captions.get(str(p), Path(p).name)) for p in image_paths]
        new_fn = Path(img_path).name + f"\n\n🤖 {new_caption}"
        log_queue.put(f"🛡️ 画像を手動保護しました: {Path(img_path).name}\n")
        return gr.update(value=gallery_items, columns=columns), new_fn, ai_captions
    return gr.update(), gr.update(), ai_captions

def isolate_ng_images(target_dir, image_paths, ai_captions, columns):
    if not image_paths: return gr.update(), image_paths, ai_captions
    filtered_dir = Path(target_dir) / "filtered_by_ai"
    filtered_dir.mkdir(exist_ok=True)
    new_image_paths, moved_count = [], 0
    for img_p in image_paths:
        caption = ai_captions.get(img_p, "")
        if caption.startswith("[NG]"):
            path_obj = Path(img_p)
            for ext in [".png", ".jpg", ".jpeg", ".webp", ".txt", ".json", ".npz"]:
                target_file = path_obj.with_suffix(ext)
                if target_file.exists(): shutil.move(str(target_file), str(filtered_dir / target_file.name))
            moved_count += 1
            if img_p in ai_captions: del ai_captions[img_p]
        else: new_image_paths.append(img_p)
    log_queue.put(f"🗑️ {moved_count} 枚のNG判定画像を隔離しました。\n")
    gallery_items = [(p, ai_captions.get(str(p), Path(p).name)) for p in new_image_paths]
    return gr.update(value=gallery_items, columns=columns), new_image_paths, ai_captions

def run_ai_wrapper(api_url, model_name, system_prompt, target_dir, image_paths, ai_captions, columns, search_more_if_compromise, translate_model):
    save_text_file(AI_PROMPT_FILE, system_prompt)
    yield from run_ai_selection_gen(api_url, model_name, system_prompt, target_dir, image_paths, ai_captions, columns, search_more_if_compromise, log_queue, translate_model)

def load_char_tag_rules():
    if not os.path.exists(CHAR_TAG_RULES_FILE): return {}
    try:
        with open(CHAR_TAG_RULES_FILE, "r", encoding="utf-8") as f: return json.load(f)
    except: return {}

def save_char_tag_rule(char_name, selected_categories):
    if not char_name: return "⚠️ キャラクターを選択してください。"
    rules = load_char_tag_rules()
    
    if char_name == "ALL":
        rules = {"ALL": selected_categories}
    else:
        rules[char_name] = selected_categories
        
    with open(CHAR_TAG_RULES_FILE, "w", encoding="utf-8") as f:
        json.dump(rules, f, indent=4, ensure_ascii=False)
        
    if char_name == "ALL":
        return f"✅ 全キャラクターの設定を {', '.join(selected_categories)} で上書き・統一しました。"
    return f"✅ {char_name} の設定を保存しました: {', '.join(selected_categories)}"

def load_rule_for_char(char_name):
    rules = load_char_tag_rules()
    if char_name == "ALL": return rules.get("ALL", ["Physical Traits", "Identity Features"])
    return rules.get(char_name, rules.get("ALL", ["Physical Traits", "Identity Features"]))

def update_rule_chars(target_dir):
    chars = []
    target_path = Path(target_dir)
    if target_path.exists():
        for p in target_path.glob("*.txt"):
            c = p.stem.split('_')[0]
            if c not in chars: chars.append(c)
    choices = ["ALL"] + sorted(chars)
    return gr.update(choices=choices)

def load_tag_stats(target_dir, char_name):
    if not char_name: return [], [], "選択してください", [], [], [], gr.update(choices=[], value=[])
    target_path = Path(target_dir); tag_counts = {}
    
    if char_name == "ALL":
        txt_paths = list(target_path.glob("*.txt"))
        img_paths = [str(p) for p in target_path.glob("*.*") if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]]
    else:
        txt_paths = list(target_path.glob(f"{char_name}_*.txt"))
        img_paths = [str(p) for p in target_path.glob(f"{char_name}_*.*") if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]]
        
    for txt_path in txt_paths:
        for t in [t.strip().lower() for t in txt_path.read_text(encoding="utf-8").split(",") if t.strip()]: tag_counts[t] = tag_counts.get(t, 0) + 1
    
    sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
    formatted = [(t, c, f"<a href='https://danbooru.donmai.us/wiki_pages/{urllib.parse.quote(t.replace(' ', '_'))}' target='_blank'>Wiki ↗</a>") for t, c in sorted_tags]
    imgs = sorted(img_paths)
    
    return formatted, formatted, "※ 自動削除ログには対応していません。削除カテゴリの設定は char_tag_rules.json に保存されます", imgs, imgs, [], gr.update(choices=[], value=[])

def toggle_select_image(evt: gr.SelectData, current_selected, image_paths):
    if not image_paths or evt.index >= len(image_paths): return current_selected, gr.update(), "なし"
    img_path = image_paths[evt.index]
    
    new_selected = list(current_selected)
    if img_path in new_selected:
        new_selected.remove(img_path)
    else:
        new_selected.append(img_path)
        
    choices = [Path(p).name for p in new_selected]
    txt = Path(img_path).with_suffix(".txt")
    tags_text = txt.read_text(encoding="utf-8") if txt.exists() else "なし"
    
    return new_selected, gr.update(choices=choices, value=choices), tags_text

def clear_selection():
    return [], gr.update(choices=[], value=[]), ""

def batch_select_images(query, current_selected, image_paths):
    new_selected = list(current_selected)
    if query and image_paths:
        for p in image_paths:
            if query.lower() in Path(p).name.lower() and p not in new_selected:
                new_selected.append(p)
                
    choices = [Path(p).name for p in new_selected]
    return new_selected, gr.update(choices=choices, value=choices)

def sync_selection_from_checkbox(checked_names, image_paths):
    name_to_path = {Path(p).name: p for p in image_paths}
    new_selected = [name_to_path[name] for name in checked_names if name in name_to_path]
    return new_selected

def filter_tags(query, raw_stats):
    return [row for row in raw_stats if query.lower() in str(row[0]).lower()] if query else raw_stats

def add_tag_from_table(evt: gr.SelectData, current_tags):
    if evt.index[1] == 0:
        curr = [t.strip() for t in current_tags.split(",") if t.strip()]
        if str(evt.value) not in curr: curr.append(str(evt.value))
        return ", ".join(curr)
    return current_tags

def batch_edit_tags(target_dir, char_name, add_tags_str, remove_tags_str, insert_pos, exclude_tags_str, selected_images):
    if not char_name: return "未選択"
    adds = [t.strip() for t in add_tags_str.split(",") if t.strip()]
    rems = [t.strip() for t in remove_tags_str.split(",") if t.strip()]
    excls = [t.strip().lower().replace("_", " ") for t in exclude_tags_str.split(",") if t.strip()]
    try: pos = int(insert_pos)
    except: pos = -1
    count = 0
    
    if selected_images and len(selected_images) > 0:
        target_txts = [Path(p).with_suffix(".txt") for p in selected_images]
    else:
        if char_name == "ALL":
            target_txts = list(Path(target_dir).glob("*.txt"))
        else:
            target_txts = list(Path(target_dir).glob(f"{char_name}_*.txt"))
            
    for txt in target_txts:
        if not txt.exists(): continue
        shutil.copy(txt, str(txt) + ".bak")
        tags = [t.strip() for t in txt.read_text(encoding="utf-8").split(",") if t.strip()]
        lowers = [t.lower().replace("_", " ") for t in tags]
        skip_add = any(et in lowers for et in excls) if excls else False
        new_tags = [t for t in tags if t not in rems]
        if not skip_add and adds:
            for t in adds:
                if t in new_tags: new_tags.remove(t)
            if pos == -1 or pos >= len(new_tags): new_tags.extend(adds)
            else: p = max(0, pos); new_tags = new_tags[:p] + adds + new_tags[p:]
        txt.write_text(", ".join(new_tags), encoding="utf-8"); count += 1
    return f"✅ {count}件更新"

def undo_batch_edit(target_dir, char_name, selected_images):
    count = 0
    if selected_images and len(selected_images) > 0:
        target_baks = [Path(p).with_suffix(".txt.bak") for p in selected_images]
    else:
        if char_name == "ALL":
            target_baks = list(Path(target_dir).glob("*.txt.bak"))
        else:
            target_baks = list(Path(target_dir).glob(f"{char_name}_*.txt.bak"))
            
    for bak in target_baks: 
        if bak.exists():
            shutil.copy(bak, bak.with_suffix('')); count += 1
    return f"↩️ {count}件復元" if count > 0 else "なし"


# ==========================================
# 💡 マスター辞書流用型 オートコンプリート機能 (サジェスト)
# ==========================================
autocomplete_tags_cache = []

def load_autocomplete_tags():
    global autocomplete_tags_cache
    if autocomplete_tags_cache: return autocomplete_tags_cache
    
    tag_set = set()
    
    # 1. メインの tag_master.json からタグを抽出
    if os.path.exists("tag_master.json"):
        try:
            with open("tag_master.json", "r", encoding="utf-8") as f:
                data = json.load(f)
                
                def _extract_tags(node):
                    if isinstance(node, dict):
                        for v in node.values(): _extract_tags(v)
                    elif isinstance(node, list):
                        for item in node:
                            if isinstance(item, str): 
                                tag_set.add(item.strip().lower().replace("_", " "))
                            else: 
                                _extract_tags(item)
                    elif isinstance(node, str):
                        tag_set.add(node.strip().lower().replace("_", " "))
                        
                _extract_tags(data)
        except Exception as e:
            print(f"Error loading tag_master.json for autocomplete: {e}")
            
    # 2. (拡張用フォールバック) CSVがあれば補足として読み込む
    tag_paths = ["tags/danbooru.csv", "tags/danbooru_e621_merged.csv", "danbooru.csv"]
    for p in tag_paths:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split(",")
                        if parts and len(parts[0]) > 0:
                            tag_name = parts[0].replace('"', '').strip().lower().replace("_", " ")
                            if tag_name not in ['name', 'tag']:
                                tag_set.add(tag_name)
            except Exception:
                pass
                
    autocomplete_tags_cache = sorted(list(tag_set))
    return autocomplete_tags_cache

def suggest_tags(text):
    tags = load_autocomplete_tags()
    if not text or not tags: 
        return gr.update(visible=False, value=[])
    
    parts = text.split(",")
    last_part = parts[-1].strip().lower()
    
    if len(last_part) < 2: 
        return gr.update(visible=False, value=[])
    
    suggestions = [t for t in tags if t.startswith(last_part)]
    
    if len(suggestions) < 15:
        part_matches = [t for t in tags if last_part in t and not t.startswith(last_part)]
        suggestions.extend(part_matches)
        
    suggestions = suggestions[:15] 
    
    if suggestions:
        return gr.update(value=[[t] for t in suggestions], visible=True)
    return gr.update(visible=False, value=[])

def apply_suggestion(text, evt: gr.SelectData):
    suggested_tag = str(evt.value)
    if not text:
        return suggested_tag + ", ", gr.update(visible=False)
        
    parts = text.split(",")
    if len(parts) > 0:
        prefix_space = " " if parts[-1].startswith(" ") else ""
        parts[-1] = prefix_space + suggested_tag
        
    new_text = ",".join(parts) + ", "
    return new_text.lstrip(), gr.update(visible=False)


with gr.Blocks(title="LoRA Dataset Maker") as app:
    gr.Markdown("# 🎨 LoRA Dataset Maker GUI")
    with gr.Tabs():
        with gr.TabItem("Step 1: 素材切り出し & 自動タグ付与"):
            gr.Markdown("入力画像からキャラ素材を自動で切り出し（一枚絵/全身/上半身/顔/目/小物）、切り分けて保存します。続けてWD14による自動タグ付与も実行されます（タグ付与には有効なsd-scriptsパスが必要です）。")
            with gr.Row():
                ac_input = gr.Textbox(label="入力", value=current_settings["ac_input"])
                ac_output = gr.Textbox(label="出力", value=current_settings["ac_output"])
                te_sd_path_s1 = gr.Textbox(label="sd-scriptsパス", value=current_settings["te_sd_path_s1"])
            with gr.Row():
                ac_res = gr.Slider(label="解像度", minimum=512, maximum=2048, step=64, value=current_settings["ac_res"])
                ac_thresh = gr.Slider(label="小物しきい値", minimum=0.01, maximum=1.0, value=current_settings["ac_thresh"])
            with gr.Row():
                ac_rembg = gr.Checkbox(label="背景除去", value=current_settings["ac_rembg"])
                ac_upscale = gr.Checkbox(label="超解像", value=current_settings["ac_upscale"])
                ac_extract_eyes = gr.Checkbox(label="目の個別切り出し", value=current_settings["ac_extract_eyes"])
                ac_filter_head = gr.Checkbox(label="頭部装飾フィルタ", value=current_settings["ac_filter_head"])
            with gr.Accordion("詳細クロップ設定", open=False):
                with gr.Row(): 
                    ac_m_top = gr.Slider(label="頭上", value=current_settings["ac_m_top"])
                    ac_m_btm = gr.Slider(label="足元", value=current_settings["ac_m_btm"])
                with gr.Row(): 
                    ac_m_side = gr.Slider(label="左右", value=current_settings["ac_m_side"])
                    ac_m_obj = gr.Slider(label="小物倍率", value=current_settings["ac_m_obj"])
                with gr.Row(): 
                    ac_max_area = gr.Slider(label="最大面積", value=current_settings["ac_max_area"])
                    ac_iou = gr.Slider(label="IoU", value=current_settings["ac_iou"])
                    ac_face_inc = gr.Slider(label="顔混入", value=current_settings["ac_face_inc"])
                with gr.Row(): 
                    ac_phash = gr.Slider(label="pHash", value=current_settings["ac_phash"])
                    ac_hist = gr.Slider(label="色分布", value=current_settings["ac_hist"])
            ac_prompts = gr.Textbox(label="小物プロンプト", lines=3, value=current_settings["ac_prompts"])
            ac_btn = gr.Button("🚀 処理を開始", variant="primary")

        with gr.TabItem("Step 2: 手動選別・AIアシスト"):
            with gr.Row():
                man_target_dir = gr.Textbox(label="対象フォルダ", value=current_settings["man_target_dir"])
                man_filter = gr.CheckboxGroup(
                    choices=["一枚絵 (_illust)", "全身 (_full)", "上半身 (_upper)", "顔 (_face)", "目 (_eye)", "小物 (_obj)"],
                    value=current_settings["man_filter"],
                    label="表示フィルター (チェックした画像のみ表示・AI評価対象となります)"
                )
            with gr.Row():
                man_columns = gr.Slider(label="一覧の列数", minimum=1, maximum=10, step=1, value=current_settings["man_columns"])
                man_load_btn = gr.Button("🔄 フォルダ読み込み")
                
            with gr.Accordion("🤖 AI選別アシスト (座標ベース高度判定)", open=True):
                gr.Markdown(f"※ 中心座標のX・Yがそれぞれ5%以内の小物を同じクラスタとして分析をまとめ、後段で総合的にOK/NGと要判定します。")
                with gr.Row(): 
                    ai_api_url = gr.Textbox(label="Ollama URL", value=current_settings["ai_api_url"])
                    ai_model_name = gr.Textbox(label="VLMモデル", value=current_settings["ai_model_name"])
                    ai_translate_model = gr.Textbox(label="翻訳モデル (空欄なら既定: gemma3:1b)", value=current_settings["ai_translate_model"])
                
                ai_search_more = gr.Checkbox(label="妥協合格(スコア4)時は、クラスタ内で完璧な画像(スコア5)を探し続ける", value=current_settings["ai_search_more"])
                ai_prompt = gr.Textbox(label="AI指示プロンプト", lines=10, value=current_settings["ai_prompt"])
                ai_run_btn = gr.Button("🤖 表示中の画像にAI判定を実行 (バッチ推論)", variant="primary")
            with gr.Row():
                with gr.Column(scale=1): 
                    man_gallery = gr.Gallery(label="画像一覧", height=700, object_fit="contain", allow_preview=False, columns=current_settings["man_columns"])
                    man_isolate_btn = gr.Button("🗑️ NG判定画像を一括隔離", variant="stop")
                with gr.Column(scale=1): 
                    man_preview = gr.Image(label="プレビュー", interactive=False)
                    man_filename = gr.Textbox(label="ファイル名", interactive=False)
                    with gr.Row():
                        man_delete_btn = gr.Button("🗑️ この画像を削除", variant="stop")
                        man_rescue_btn = gr.Button("✅ NG判定を取り消す（保護）", variant="primary")
                    man_state, man_current_idx, man_ai_captions = gr.State([]), gr.State(-1), gr.State({})

        with gr.TabItem("Step 3: タグ編集・クリーン"):
            gr.Markdown("選別済みの素材に対し、マルゼン式の「概念の削除」や手動でのタグ調整を行います。")
            with gr.Row():
                te_target_dir_inp = gr.Textbox(label="対象フォルダ", value=current_settings["te_target_dir_inp"])
            
            gr.Markdown("💡 **AIマスター辞書フィルタリング:** `tag_master.json` を使用した自動マルゼン式が適用されます。")
            
            with gr.Row(): 
                te_refresh_btn = gr.Button("🔄 1. キャラクタ一覧を更新")
                te_char_select = gr.Dropdown(label="プレビュー対象キャラ", choices=[])
            gr.Markdown("---")
            
            with gr.Accordion("🎭 キャラクター別 タグ学習（削除）カテゴリ設定", open=False):
                gr.Markdown("キャラクターごとに、AIに学習させる（＝キャプションから除去する）タグのカテゴリを選択します。")
                with gr.Row():
                    rule_char_select = gr.Dropdown(label="設定対象キャラクター", choices=[], interactive=True)
                rule_categories = gr.CheckboxGroup(
                    choices=[
                        "Physical Traits", "Identity Features", "Clothing", 
                        "Props & Weapons", "Facial Expressions", "Composition", 
                        "Background & Scene", "Style & Effects"
                    ],
                    value=["Physical Traits", "Identity Features"],
                    label="AIに学習させる（キャプションから削除する）カテゴリ"
                )
                rule_save_btn = gr.Button("💾 このキャラクターの設定を保存")
                rule_status = gr.Markdown("")
            
            with gr.Accordion("⚙️ 従来の正規表現ベースの整理ルール (通常は使用不要)", open=False):
                gr.Markdown("※ マスター辞書で処理しきれない特殊なタグや、明示的な保護/削除ルールを追加したい場合のみ使用します。")
                te_tags_config = gr.Textbox(label="自動整理設定", lines=6, max_lines=15, value=current_settings["te_tags_config"])
                
            te_clean_btn = gr.Button("⚙️ タグ一括整理を実行", variant="primary")
            
            gr.Markdown("---")
            with gr.Row():
                with gr.Column(scale=4):
                    te_search = gr.Textbox(label="🔍 タグ検索")
                    te_tag_stats = gr.Dataframe(headers=["タグ名", "数", "Danbooru"], datatype=["str", "number", "markdown"])
                    te_deleted_log = gr.Textbox(label="自動削除ログ", interactive=False, lines=3)
                with gr.Column(scale=6):
                    gr.Markdown("💡 画像をクリックすると複数選択できます（プレビュー無効化済）。空欄の場合は「表示中の全画像」に適用されます。")
                    with gr.Row():
                        te_batch_select_text = gr.Textbox(label="ファイル名で一括選択", placeholder="例: _face", scale=3)
                        te_batch_select_btn = gr.Button("➕ 一括選択", scale=1)
                    te_gallery = gr.Gallery(label="プレビュー", height=300, object_fit="contain", columns=4, allow_preview=False)
                    te_selected_images_chk = gr.CheckboxGroup(label="現在選択中の画像 (チェックを外して個別解除 / 空なら全対象)", choices=[], value=[], interactive=True)
                    te_img_tags = gr.Textbox(label="最後に選択した画像のタグ", interactive=False, lines=2)
                    
                    with gr.Row(): 
                        te_add_tags = gr.Textbox(label="追加タグ (カンマ区切りで入力)", scale=3)
                        te_insert_pos = gr.Number(label="位置", value=-1, scale=1)
                    # 💡 [修正] col_count を column_count に変更
                    te_add_suggest = gr.Dataframe(headers=["💡 候補 (クリックで補完)"], datatype=["str"], column_count=(1, "fixed"), visible=False, interactive=False)
                    
                    with gr.Row(): 
                        te_remove_tags = gr.Textbox(label="削除タグ (カンマ区切りで入力)", scale=3)
                        te_exclude_tags = gr.Textbox(label="追加除外条件", scale=2)
                    with gr.Row():
                        with gr.Column(scale=3):
                            # 💡 [修正] col_count を column_count に変更
                            te_remove_suggest = gr.Dataframe(headers=["💡 候補 (クリックで補完)"], datatype=["str"], column_count=(1, "fixed"), visible=False, interactive=False)
                        with gr.Column(scale=2):
                            # 💡 [修正] col_count を column_count に変更
                            te_exclude_suggest = gr.Dataframe(headers=["💡 候補 (クリックで補完)"], datatype=["str"], column_count=(1, "fixed"), visible=False, interactive=False)
                            
                    with gr.Row(): 
                        te_exec_btn = gr.Button("⚙️ 2. タグ更新", variant="primary")
                        te_undo_btn = gr.Button("↩️ 戻す")
                        te_clear_selection_btn = gr.Button("✖️ 選択クリア", variant="secondary")
                    te_result = gr.Textbox(label="結果", interactive=False)
                    te_raw_stats_state, te_image_paths_state, te_selected_images_state = gr.State([]), gr.State([]), gr.State([])

        with gr.TabItem("Step 4: 最終構築"):
            gr.Markdown("Step 1〜3で整備した素材を、比率・枚数設定に沿って背景画像（bg_images）へ合成し、色補正やブラーを掛けながらサンプリングして学習用データセット（画像+キャプション）を生成します。")
            with gr.Row(): 
                db_input = gr.Textbox(label="入力", value=current_settings["db_input"])
                db_output = gr.Textbox(label="出力先", value=current_settings["db_output"])
            with gr.Row(): 
                db_bg = gr.Textbox(label="背景", value=current_settings["db_bg"])
                db_sd_inp = gr.Textbox(label="sd-scripts", value=current_settings["db_sd_inp"])
            with gr.Row(): 
                db_total = gr.Number(label="目標枚数", value=current_settings["db_total"])
                db_jitter = gr.Slider(label="色補正", value=current_settings["db_jitter"])
                db_blur = gr.Slider(label="ブラー", value=current_settings["db_blur"])
            with gr.Row(): 
                db_keep_in_memory = gr.Checkbox(label="⚡ 高速処理モード (推奨: 32GB RAM以上)", value=current_settings["db_keep_in_memory"])
                
            with gr.Accordion("⚙️ 比率設定 (カテゴリごとの目標割合)", open=True):
                with gr.Row():
                    db_ratio_face = gr.Slider(label="顔 (%)", value=current_settings["db_ratio_face"], maximum=100)
                    db_ratio_upper = gr.Slider(label="上半身 (%)", value=current_settings["db_ratio_upper"], maximum=100)
                    db_ratio_full = gr.Slider(label="全身 (%)", value=current_settings["db_ratio_full"], maximum=100)
                    db_ratio_illust = gr.Slider(label="一枚絵 (%)", value=current_settings["db_ratio_illust"], maximum=100)
                with gr.Row():
                    db_ratio_reye = gr.Slider(label="右目 (%)", value=current_settings["db_ratio_reye"], maximum=100)
                    db_ratio_leye = gr.Slider(label="左目 (%)", value=current_settings["db_ratio_leye"], maximum=100)
                    db_ratio_obj = gr.Slider(label="小物 (%)", value=current_settings["db_ratio_obj"], maximum=100)
                    
            db_btn = gr.Button("✨ 学習データセットを生成", variant="primary")

    gr.Markdown("---"); console_log = gr.Textbox(label="通知", lines=3, interactive=False); gr.Timer(1).tick(read_logs, outputs=console_log)

    ac_btn.click(start_combined_process_gen, inputs=[ac_input, ac_output, ac_res, ac_rembg, ac_upscale, ac_thresh, ac_m_top, ac_m_btm, ac_m_side, ac_m_obj, ac_prompts, ac_max_area, ac_iou, ac_phash, ac_hist, ac_face_inc, gr.State(True), ac_extract_eyes, ac_filter_head, te_sd_path_s1], outputs=[ac_btn])
    man_columns.change(lambda c: gr.update(columns=c), inputs=[man_columns], outputs=[man_gallery])
    man_filter.change(load_manual_selection, inputs=[man_target_dir, man_columns, man_filter], outputs=[man_gallery, man_state, man_preview, man_filename, man_current_idx])
    man_load_btn.click(load_manual_selection, inputs=[man_target_dir, man_columns, man_filter], outputs=[man_gallery, man_state, man_preview, man_filename, man_current_idx])
    man_gallery.select(on_gallery_select, inputs=[man_state, man_ai_captions], outputs=[man_preview, man_filename, man_current_idx])
    man_delete_btn.click(delete_selected_image, inputs=[man_target_dir, man_current_idx, man_state, man_ai_captions, man_columns], outputs=[man_gallery, man_preview, man_filename, man_current_idx, man_state, man_ai_captions])
    man_rescue_btn.click(rescue_selected_image, inputs=[man_current_idx, man_state, man_ai_captions, man_columns], outputs=[man_gallery, man_filename, man_ai_captions])
    ai_run_btn.click(run_ai_wrapper, inputs=[ai_api_url, ai_model_name, ai_prompt, man_target_dir, man_state, man_ai_captions, man_columns, ai_search_more, ai_translate_model], outputs=[ai_run_btn, man_ai_captions, man_gallery])
    man_isolate_btn.click(isolate_ng_images, inputs=[man_target_dir, man_state, man_ai_captions, man_columns], outputs=[man_gallery, man_state, man_ai_captions])

    te_clean_btn.click(start_cleaner_gen, inputs=[te_target_dir_inp, te_tags_config], outputs=[te_clean_btn])
    te_refresh_btn.click(update_rule_chars, inputs=[te_target_dir_inp], outputs=[rule_char_select]).then(update_rule_chars, inputs=[te_target_dir_inp], outputs=[te_char_select])
    rule_char_select.change(load_rule_for_char, inputs=[rule_char_select], outputs=[rule_categories])
    rule_save_btn.click(save_char_tag_rule, inputs=[rule_char_select, rule_categories], outputs=[rule_status])
    
    te_char_select.change(load_tag_stats, inputs=[te_target_dir_inp, te_char_select], outputs=[te_raw_stats_state, te_tag_stats, te_deleted_log, te_gallery, te_image_paths_state, te_selected_images_state, te_selected_images_chk])
    te_search.change(filter_tags, inputs=[te_search, te_raw_stats_state], outputs=[te_tag_stats])
    te_tag_stats.select(add_tag_from_table, inputs=[te_remove_tags], outputs=[te_remove_tags])
    
    te_gallery.select(toggle_select_image, inputs=[te_selected_images_state, te_image_paths_state], outputs=[te_selected_images_state, te_selected_images_chk, te_img_tags])
    te_selected_images_chk.change(sync_selection_from_checkbox, inputs=[te_selected_images_chk, te_image_paths_state], outputs=[te_selected_images_state])
    te_batch_select_btn.click(batch_select_images, inputs=[te_batch_select_text, te_selected_images_state, te_image_paths_state], outputs=[te_selected_images_state, te_selected_images_chk])
    te_clear_selection_btn.click(clear_selection, outputs=[te_selected_images_state, te_selected_images_chk, te_img_tags])
    
    te_add_tags.change(suggest_tags, inputs=[te_add_tags], outputs=[te_add_suggest])
    te_add_suggest.select(apply_suggestion, inputs=[te_add_tags], outputs=[te_add_tags, te_add_suggest])
    
    te_remove_tags.change(suggest_tags, inputs=[te_remove_tags], outputs=[te_remove_suggest])
    te_remove_suggest.select(apply_suggestion, inputs=[te_remove_tags], outputs=[te_remove_tags, te_remove_suggest])
    
    te_exclude_tags.change(suggest_tags, inputs=[te_exclude_tags], outputs=[te_exclude_suggest])
    te_exclude_suggest.select(apply_suggestion, inputs=[te_exclude_tags], outputs=[te_exclude_tags, te_exclude_suggest])
    
    te_exec_btn.click(batch_edit_tags, inputs=[te_target_dir_inp, te_char_select, te_add_tags, te_remove_tags, te_insert_pos, te_exclude_tags, te_selected_images_state], outputs=[te_result]).then(load_tag_stats, inputs=[te_target_dir_inp, te_char_select], outputs=[te_raw_stats_state, te_tag_stats, te_deleted_log, te_gallery, te_image_paths_state, te_selected_images_state, te_selected_images_chk])
    te_undo_btn.click(undo_batch_edit, inputs=[te_target_dir_inp, te_char_select, te_selected_images_state], outputs=[te_result]).then(load_tag_stats, inputs=[te_target_dir_inp, te_char_select], outputs=[te_raw_stats_state, te_tag_stats, te_deleted_log, te_gallery, te_image_paths_state, te_selected_images_state, te_selected_images_chk])
    
    db_btn.click(start_builder_gen, inputs=[db_input, db_output, db_bg, db_total, db_jitter, db_blur, db_keep_in_memory, db_ratio_face, db_ratio_upper, db_ratio_full, db_ratio_illust, db_ratio_reye, db_ratio_leye, db_ratio_obj, db_sd_inp], outputs=[db_btn])

    components_to_save = {
        "ac_input": ac_input, "ac_output": ac_output, "te_sd_path_s1": te_sd_path_s1,
        "ac_res": ac_res, "ac_thresh": ac_thresh,
        "ac_rembg": ac_rembg, "ac_upscale": ac_upscale, "ac_extract_eyes": ac_extract_eyes, "ac_filter_head": ac_filter_head,
        "ac_m_top": ac_m_top, "ac_m_btm": ac_m_btm, "ac_m_side": ac_m_side, "ac_m_obj": ac_m_obj,
        "ac_max_area": ac_max_area, "ac_iou": ac_iou, "ac_face_inc": ac_face_inc, "ac_phash": ac_phash, "ac_hist": ac_hist,
        "ac_prompts": ac_prompts,
        "man_target_dir": man_target_dir, "man_filter": man_filter, "man_columns": man_columns,
        "ai_api_url": ai_api_url, "ai_model_name": ai_model_name, "ai_translate_model": ai_translate_model, "ai_search_more": ai_search_more, "ai_prompt": ai_prompt,
        "te_target_dir_inp": te_target_dir_inp, "te_tags_config": te_tags_config,
        "db_input": db_input, "db_output": db_output, "db_bg": db_bg, "db_sd_inp": db_sd_inp,
        "db_total": db_total, "db_jitter": db_jitter, "db_blur": db_blur, "db_keep_in_memory": db_keep_in_memory,
        "db_ratio_face": db_ratio_face, "db_ratio_upper": db_ratio_upper, "db_ratio_full": db_ratio_full,
        "db_ratio_illust": db_ratio_illust, "db_ratio_reye": db_ratio_reye, "db_ratio_leye": db_ratio_leye, "db_ratio_obj": db_ratio_obj
    }
    def create_update_fn(key): return lambda val: update_setting(key, val)
    for key, comp in components_to_save.items(): comp.change(create_update_fn(key), inputs=[comp], outputs=[])

if __name__ == "__main__":
    app.launch(inbrowser=True)