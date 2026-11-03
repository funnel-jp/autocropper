# file: ai_selector.py
import os
import time
import base64
import requests
import json
import re
import cv2
import numpy as np
import urllib.request
import urllib.error
import gradio as gr
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# CPU翻訳モデル用のグローバル変数を撤廃
# translator_pipeline = None

class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
    def find(self, i):
        if self.parent[i] == i: return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]
    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j

def group_images_by_center(image_paths, dist_threshold_x=0.05, dist_threshold_y=0.05):
    clusters = {}
    char_groups = {}
    
    for img_path in image_paths:
        path_obj = Path(img_path)
        char_name = path_obj.stem.split("_")[0]
        
        center_rel = None
        label = "unknown"
        json_path = path_obj.with_suffix(".json")
        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    center_rel = data.get("original_center_rel")
                    label = data.get("label", "unknown")
            except: pass
            
        if char_name not in char_groups:
            char_groups[char_name] = []
        char_groups[char_name].append({"path": img_path, "label": label, "center_rel": center_rel})
        
    cluster_offset = 0
    for char_name, items in char_groups.items():
        n = len(items)
        uf = UnionFind(n)
        for i in range(n):
            for j in range(i + 1, n):
                if items[i]["center_rel"] and items[j]["center_rel"] and items[i]["label"] == items[j]["label"]:
                    cx1, cy1 = items[i]["center_rel"]
                    cx2, cy2 = items[j]["center_rel"]
                    if abs(cx1 - cx2) <= dist_threshold_x and abs(cy1 - cy2) <= dist_threshold_y:
                        uf.union(i, j)
        
        for i in range(n):
            root = uf.find(i)
            clusters[items[i]["path"]] = f"{char_name}_cluster_{cluster_offset + root}"
        cluster_offset += n
            
    return clusters

def analyze_image_with_ollama(api_url, model_name, image_paths, prompt, max_sizes=None, jpeg_quality=80, temperature=0.2):
    import numpy as np  # 念のためのローカルインポート
    
    # [変更] 解像度を元の512pxに復旧
    if max_sizes is None:
        max_sizes = [512] + [336] * max(0, len(image_paths) - 1)
        
    used_bgs = [] # 背景情報のリスト
    
    try:
        images_b64 = []
        for i, p in enumerate(image_paths):
            if os.path.exists(p):
                img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                if img is not None:
                    bg_info = "No Alpha (Opaque)" # 透過情報がない場合のデフォルト
                    
                    # --- エッジ色を判定し、同化を防ぐ動的背景色決定 ---
                    if len(img.shape) == 3 and img.shape[2] == 4:
                        alpha_channel = img[:, :, 3]
                        rgb_channels = img[:, :, :3]
                        
                        mask = alpha_channel > 0
                        if np.any(mask):
                            # アルファチャンネルを収縮させてエッジ（輪郭）マスクを作成
                            kernel = np.ones((5, 5), np.uint8)
                            eroded = cv2.erode(alpha_channel, kernel, iterations=1)
                            edge_mask = mask & (eroded == 0)
                            
                            # 万が一エッジが少なすぎる場合は、非透過領域全体を対象にする
                            if np.sum(edge_mask) < 10:
                                edge_mask = mask
                                
                            # エッジ部分の平均BGR色を算出
                            mean_bgr = rgb_channels[edge_mask].mean(axis=0)
                            
                            # VLMのノイズになりにくい、くすんだ候補色リスト (BGRフォーマット)
                            candidates = [
                                np.array([128, 128, 128]),  # ニュートラルグレー
                                np.array([80,  80,  80 ]),  # 暗いグレー
                                np.array([180, 180, 180]),  # 明るいグレー
                                np.array([110, 160, 110]),  # くすんだ緑 (グリーンバック風)
                                np.array([160, 110, 110]),  # くすんだ青 (ブルーバック風)
                                np.array([110, 110, 160])   # くすんだ赤紫
                            ]
                            
                            # 平均色と最も距離が遠い（コントラストが高い）色を選択
                            best_bg = candidates[0]
                            max_dist = -1
                            for c in candidates:
                                dist = np.linalg.norm(mean_bgr - c)
                                if dist > max_dist:
                                    max_dist = dist
                                    best_bg = c
                                    
                            bg_info = f"Added BG: BGR{best_bg.tolist()} (Edge Mean: BGR{[int(x) for x in mean_bgr]})"
                        else:
                            best_bg = np.array([128, 128, 128]) # 完全に透過の場合はデフォルトグレー
                            bg_info = f"Added BG: BGR{best_bg.tolist()} (Fully Transparent)"
                            
                        # 動的に決定した背景色でアルファブレンド合成
                        bg_img = np.full_like(rgb_channels, best_bg, dtype=np.uint8)
                        alpha_factor = alpha_channel[:, :, np.newaxis] / 255.0
                        img = (rgb_channels * alpha_factor + bg_img * (1 - alpha_factor)).astype(np.uint8)
                    # --------------------------------------------------------
                    
                    used_bgs.append(bg_info)
                    
                    h, w = img.shape[:2]
                    max_s = max_sizes[i] if i < len(max_sizes) else max_sizes[-1]
                    
                    if max(h, w) > max_s:
                        scale = max_s / float(max(h, w))
                        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                    
                    _, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
                    images_b64.append(base64.b64encode(buffer).decode('utf-8'))
            else:
                used_bgs.append("File not found")
        
        if not images_b64: return "Error: No images", used_bgs
        
        payload = {
            "model": model_name, 
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": images_b64
                }
            ],
            "stream": False,
            "keep_alive": "5m",
            "options": {
                "temperature": temperature,
                "num_ctx": 8192  # [変更] トークン枯渇を防ぐためコンテキスト長を4096から8192へ拡張
            }
        }
        
        response = requests.post(f"{api_url}/api/chat", json=payload, timeout=600)
        response.raise_for_status()
        
        return response.json().get("message", {}).get("content", ""), used_bgs
    except Exception as e: return f"Error: {str(e)}", used_bgs

DEFAULT_TRANSLATE_MODEL = "gemma3:1b"

def translate_text_with_ollama(api_url, model_name, text):
    model = model_name if model_name else DEFAULT_TRANSLATE_MODEL
    
    if "plamo" in model.lower():
        # plamo-2-translateはチャットファインチューニングされていないため、
        # 翻訳専用のopトークン形式でrawリクエストを送る
        prompt = (
            "<|plamo:op|>dataset\ntranslation\n"
            "<|plamo:op|>input lang=Japanese|English\n"
            f"{text}\n"
            "<|plamo:op|>output\n"
        )
        req_body = {
            "model": model,
            "prompt": prompt,
            "raw": True,
            "stream": False,
            "keep_alive": "5m",
            "options": {"temperature": 0.1, "num_predict": 400, "num_gpu": 0}
        }
    else:
        # プロンプトを強化し、中国語の混入をシステムプロンプトレベルで強制的に禁止する
        prompt = (
            "You are a professional English-to-Japanese translator. "
            "Translate the following English text to brief, natural Japanese. "
            "CRITICAL RULES:\n"
            "- Output ONLY the translated Japanese text.\n"
            "- DO NOT use any Simplified or Traditional Chinese characters.\n"
            "- Use ONLY standard Japanese characters (Hiragana, Katakana, Kanji).\n"
            "- Do not include explanations, notes, or markdown.\n\n"
            f"Text to translate:\n{text}"
        )
        req_body = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": "5m",
            "options": {"temperature": 0.1, "num_predict": 400}
        }
    try:
        req = urllib.request.Request(
            f"{api_url.rstrip('/')}/api/generate", 
            data=json.dumps(req_body).encode('utf-8'), 
            headers={'Content-Type': 'application/json'}
        )
        with urllib.request.urlopen(req, timeout=300) as response:
            result = json.loads(response.read().decode('utf-8'))
            translated = result.get('response', text).strip()
            if "plamo" in model.lower():
                translated = translated.split("<|plamo:op|>")[0].strip()
            return translated if translated else text
    except Exception as e:
        print(f"Translation failed: {e}")
        return text

def run_ai_selection_gen(api_url, model_name, system_prompt, target_dir, image_paths, ai_captions, columns, search_more_if_compromise, log_queue, translate_model=None):
    
    if not image_paths:
        log_queue.put("⚠️ 画像なし\n")
        yield gr.update(value="🤖 表示中の画像にAI判定を実行", interactive=True), ai_captions, gr.update()
        return
        
    if not system_prompt or len(system_prompt.strip()) < 10:
        log_queue.put("⚠️ エラー: 「AI指示プロンプト」が空、または短すぎます。\n")
        yield gr.update(value="🤖 表示中の画像にAI判定を実行", interactive=True), ai_captions, gr.update()
        return
        
    yield gr.update(value="⏳ AI評価中...", interactive=False), ai_captions, gr.update()
    
    target_images = [str(p) for p in image_paths]
    total = len(target_images)
        
    log_queue.put(f"🤖 VLM評価開始 (スマート・フォールバックモード): {total} 枚\n")
    
    debug_log_path = "ai_debug_log.txt"
    debug_content = f"=== AI Debug Log: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} ===\n"
    debug_content += f"Model: {model_name}\nTarget Count: {total}\n\n"
    
    clusters = group_images_by_center(target_images, dist_threshold_x=0.05, dist_threshold_y=0.05)
    
    cluster_to_imgs = {}
    for img_p in target_images:
        c_id = clusters.get(img_p, f"unclustered_{Path(img_p).name}")
        if c_id not in cluster_to_imgs:
            cluster_to_imgs[c_id] = []
        cluster_to_imgs[c_id].append(img_p)
    
    translation_pool = ThreadPoolExecutor(max_workers=1)
    pending_translations = []
    
    def harvest_translations():
        updated = False
        remaining = []
        for entry in pending_translations:
            if entry["fut"].done():
                ja = entry["fut"].result()
                prefix = "[OK]" if entry["ok"] else "[NG]"
                ai_captions[entry["img"]] = f"{prefix} {ja} (Score:{entry['score']})"
                updated = True
            else:
                remaining.append(entry)
        pending_translations[:] = remaining
        return updated
    
    for c_id, imgs_in_cluster in cluster_to_imgs.items():
        if pending_translations:
            if harvest_translations():
                gallery_items = [(p, ai_captions.get(str(p), Path(p).name)) for p in target_images]
                yield gr.update(), ai_captions, gr.update(value=gallery_items, columns=columns)
        
        log_queue.put(f"🧠 クラスタ処理開始: {c_id} (総数: {len(imgs_in_cluster)}枚)\n")
        debug_content += f"\n{'='*40}\nCluster: {c_id} ({len(imgs_in_cluster)} files)\n{'='*40}\n"
        
        ref_image = None
        for img_path in imgs_in_cluster:
            path_obj = Path(img_path)
            original_stem = path_obj.stem
            json_path = path_obj.with_suffix(".json")
            
            if json_path.exists():
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        label = data.get("label", "").replace(" ", "_")
                        if label:
                            import re
                            match = re.match(rf'^(.*)_{re.escape(label)}_\d{{2}}', original_stem)
                            if match:
                                original_stem = match.group(1)
                except Exception:
                    pass
            
            ref_image = next((str(p) for ext in [".png", ".jpg", ".jpeg", ".webp"] if (p := path_obj.parent / f"{original_stem}_upper{ext}").exists()), None)
            if ref_image: break
            
        best_img = None
        best_score = -1
        cluster_evaluations = {}
        
        for img_path in imgs_in_cluster:
            imgs_to_send = [img_path]
            # [変更] リクエスト解像度を512pxに復旧
            sizes = [512]
            
            max_retries = 3 
            score = -1
            reason = "Parsing Error"
            trans_fut = None
            
            for attempt in range(max_retries):
                current_prompt = system_prompt
                
                if attempt == 0:
                    current_temp = 0.4
                    debug_content += f"\n--- Evaluating Candidate: {Path(img_path).name} (Temp: {current_temp}) ---\n"
                    log_queue.put(f"  -> 画像を評価中: {Path(img_path).name}\n")
                elif attempt == 1:
                    current_temp = 0.6
                    current_prompt += "\n\n[WARNING] Format required: 'Reason: [text]' then 'Score: [number]'."
                    debug_content += f"\n--- Retry {attempt+1} for Candidate: {Path(img_path).name} (Temp: {current_temp}) ---\n"
                    log_queue.put(f"  -> 🔄 再評価を実行中 ({attempt+1}/{max_retries}): {Path(img_path).name}\n")
                else:
                    current_temp = 0.8
                    current_prompt += "\n\n[CRITICAL] No markdown. EXACTLY 'Reason: [text]' then 'Score: [number]'."
                    debug_content += f"\n--- Retry {attempt+1} for Candidate: {Path(img_path).name} (Temp: {current_temp}) ---\n"
                    log_queue.put(f"  -> 🔄 最終再評価を実行中 ({attempt+1}/{max_retries}): {Path(img_path).name}\n")
                
                response, used_bgs = analyze_image_with_ollama(
                    api_url, model_name, imgs_to_send, current_prompt,
                    max_sizes=sizes, jpeg_quality=80, temperature=current_temp
                )
                
                if attempt == 0 and used_bgs:
                    debug_content += f"Background Info: {used_bgs[0]}\n"
                    
                debug_content += f"Raw Response:\n{response}\n"
                
                try:
                    if str(response).strip().startswith("Error:"):
                        raise ValueError(f"API Communication Error: {response}")
                        
                    clean_response = str(response).replace('*', '').replace('#', '')
                    
                    import re
                    reason_match = re.search(r'(?:理由|Reason|reason)[\s:：]*([\s\S]*?)(?:Score|スコア|score|合計)', clean_response, re.IGNORECASE)
                    if reason_match:
                        parsed_reason = reason_match.group(1).strip()
                    else:
                        split_res = re.split(r'(?:Score|スコア|score|合計)[\s:：]*\d+', clean_response, flags=re.IGNORECASE)
                        if len(split_res) > 0 and split_res[0].strip():
                            parsed_reason = split_res[0].strip().replace("Reason:", "").strip()
                        else:
                            parsed_reason = clean_response.replace('\n', ' ')[:60].strip() + "..."

                    score_matches = re.findall(r'(?:スコア|Score|score|合計)[\s:：]*(\d+)', clean_response, re.IGNORECASE)
                    if score_matches:
                        parsed_score = int(score_matches[-1])
                    else:
                        fallback_match = re.findall(r'\d+', clean_response)
                        if fallback_match:
                            parsed_score = int(fallback_match[-1])
                        else:
                            raise ValueError("レスポンスからスコア(数値)を抽出できませんでした。")
                    
                    parsed_score = max(0, min(5, parsed_score))
                            
                    log_queue.put(f"  -> 英語での推論完了。日本語翻訳はバックグラウンドで並行実行中...\n")
                    trans_fut = translation_pool.submit(translate_text_with_ollama, api_url, translate_model, parsed_reason)
                    
                    debug_content += f"Parsed English Reason: {parsed_reason}\n"
                    
                    score = parsed_score
                    reason = parsed_reason
                    log_queue.put(f"  -> 判定完了 (Score:{score}) 理由: {reason[:20]}...\n")
                    break 
                         
                except Exception as e:
                    debug_content += f"Text Parse Error: {e}\n"
                    reason = f"Parsing Error: {e}"
                    if attempt < max_retries - 1:
                        log_queue.put(f"  -> ⚠️ 解析エラー: {e} (条件を変更して再評価します)\n")
                    else:
                        log_queue.put(f"  -> ⚠️ 解析エラー: {e} (エラーとして確定します)\n")
                        score = -1

            cluster_evaluations[img_path] = {"score": score, "reason": reason, "trans_fut": trans_fut}
            
            if score > best_score:
                best_score = score
                best_img = img_path
                
            if score == 0:
                log_queue.put(f"  -> [不合格] 対象外カテゴリ(服飾品など)と判定(スコア0)。他画像の可能性を考慮し、クラスタ内の次の候補を評価します。\n")
                continue
                
            if score == 5:
                log_queue.put(f"  -> [採用] 完璧な画像(スコア5)を見つけました。同クラスタの残りはスキップします。\n")
                break
            
            if score == 4:
                if not search_more_if_compromise:
                    log_queue.put(f"  -> [採用] 妥協合格(スコア4)として採用します。同クラスタの残りはスキップします。\n")
                    break
                else:
                    log_queue.put(f"  -> [保留] 妥協合格(スコア4)。さらに高品質な画像を探すためクラスタの残りを評価します。\n")
            
            if score == -1:
                log_queue.put(f"  -> [保留] 解析失敗。クラスタ内の次の候補を評価します。\n")
            elif score < 4 and score != 0:
                log_queue.put(f"  -> [保留] スコア不足(Score:{score})。クラスタ内の次の候補を評価します。\n")

        for img_path in imgs_in_cluster:
            if img_path in cluster_evaluations:
                eval_data = cluster_evaluations[img_path]
                s = eval_data["score"]
                r = eval_data["reason"]
                
                if s < 4:
                    prefix = "[NG]"
                elif img_path == best_img:
                    prefix = "[OK]"
                else:
                    ai_captions[img_path] = f"[NG] 重複: 同じ位置により高品質な画像が採用されました"
                    continue
                
                ai_captions[img_path] = f"{prefix} {r} (Score:{s})"
                fut = eval_data.get("trans_fut")
                if fut is not None:
                    pending_translations.append({"img": img_path, "fut": fut, "score": s, "ok": prefix == "[OK]"})
            else:
                ai_captions[img_path] = f"[NG] 重複: 先行画像が採用されたため間引き"
                
        gallery_items = [(p, ai_captions.get(str(p), Path(p).name)) for p in target_images]
        yield gr.update(), ai_captions, gr.update(value=gallery_items, columns=columns)

    while pending_translations:
        entry = pending_translations.pop(0)
        ja = entry["fut"].result()
        prefix = "[OK]" if entry["ok"] else "[NG]"
        ai_captions[entry["img"]] = f"{prefix} {ja} (Score:{entry['score']})"
    translation_pool.shutdown(wait=False)

    with open(debug_log_path, "w", encoding="utf-8") as f:
        f.write(debug_content)
    log_queue.put(f"📝 デバッグログを {debug_log_path} に保存しました。\n")
    log_queue.put("✨ 全てのAI判定処理が完了しました。\n")
    gallery_items = [(p, ai_captions.get(str(p), Path(p).name)) for p in target_images]
    yield gr.update(value="🤖 表示中の画像にAI判定を実行", interactive=True), ai_captions, gr.update(value=gallery_items, columns=columns)