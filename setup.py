# -*- coding: utf-8 -*-
"""autocropper 環境構築スクリプト

使用するモデルはすべてダウンロード前に一覧提示し、ユーザーの明示的な同意・
別モデル指定・スキップを受け付けます。ライセンスは各配布元ページで各自確認の
うえ、自己責任で利用してください。手順はすべて冪等 (再実行可能) です。

使い方:
    python setup.py                     # 対話形式でモデルを選択
    python setup.py --yes               # settings.json の既定値で全取得
    python setup.py --vlm NAME --translate NAME  # 対話せず指定モデルを取得
    python setup.py --skip-hf --skip-ollama      # 各ダウンロード工程をスキップ
    python setup.py --sd-scripts PATH   # sd-scripts の WD14 タガーモデルを事前取得
"""
import argparse
import json
import shutil
import subprocess
import sys
import venv
from pathlib import Path

BASE = Path(__file__).resolve().parent
VENV = BASE / ".venv"
REQ = BASE / "requirements.txt"
SETTINGS = BASE / "settings.json"
IS_WIN = sys.platform == "win32"
PY_BIN = VENV / ("Scripts" if IS_WIN else "bin") / ("python.exe" if IS_WIN else "python")

DEFAULT_VLM = "qwen3.5:9b"
DEFAULT_TRANSLATE = "gemma3:1b"
DEFAULT_REMBG = "isnet-anime"
WD14_REPO = "SmilingWolf/wd-eva02-large-tagger-v3"
FACE_ID = "Bingsu/adetailer:face_yolov8n.pt"
ESRGAN_ID = "ac-pill/upscale_models:RealESRGAN_x4plus_anime_6B.pth"
OWLV2_ID = "google/owlv2-base-patch16-ensemble"

VLM_OPTIONS = [
    ("qwen3.5:9b", "約6.5GB / 日本語理解が高く推奨。VRAM12GB目安"),
    ("qwen2.5vl:7b", "約6GB / 画像評価実績が豊富な安定選択"),
    ("gemma3:12b", "約8GB / 日本語強く視覚対応。VRAM12GBではやや苦しい"),
    ("llama3.2-vision:11b", "約8GB / 視覚対応だが日本語推論はいまひとつ"),
    ("gemma3:4b", "約3GB / 低VRAM・低性能PC向けの軽量選択"),
]
TRANSLATE_OPTIONS = [
    ("gemma3:1b", "0.8GB / GPU常駐。12GB VRAMでもVLMと共存しやすい既定"),
    ("hf.co/mitmul/plamo-2-translate-GGUF:IQ4_XS", "5.7GB / 翻訳特化。実行時はCPU動作でVRAM消費ゼロ。HFでPLaMoコミュニティライセンス同意要"),
    ("hf.co/mitmul/plamo-2-translate-GGUF:Q4_K_M", "6.3GB / plamo-2-translate の高品質量子化版。同上の同意要"),
]
REMBG_OPTIONS = [("isnet-anime", "アニメ向け。推奨"), ("u2net", "汎用")]


def banner(t):
    print(f"\n{'='*60}\n{t}\n{'='*60}")


def run(cmd, **kw):
    print("  $ " + " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, **kw)


def load_settings():
    try:
        return json.loads(SETTINGS.read_text(encoding="utf-8"))
    except Exception:
        return {}


def choose(label, default, options, assume_yes, allow_skip=True, note=""):
    print(f"\n--- {label} ---")
    if note:
        print("  " + note)
    print(f"  既定: {default}")
    for i, (mid, desc) in enumerate(options, 1):
        print(f"  [{i}] {mid}  {desc}")
    if assume_yes:
        print("  (--yes 指定のため既定を採用)")
        return default
    keys = f"[1-{len(options)}]=一覧から選択 / [Enter]=既定を使用 / [o]=モデル名を直接入力" + (" / [s]=スキップ" if allow_skip else "")
    while True:
        try:
            ans = input(f"  このモデルをダウンロードして利用しますか? {keys}: ").strip()
            if ans.lower() == "o":
                val = input("  モデル名を直接入力: ").strip()
                if not val:
                    continue
                return val
        except EOFError:
            print("  (標準入力が終了したため既定値を採用します)")
            return default
        if ans == "":
            return default
        if ans.lower() == "s" and allow_skip:
            return None
        if ans.isdigit() and 1 <= int(ans) <= len(options):
            return options[int(ans) - 1][0]
        print(f"  1〜{len(options)} の番号 / o / s のいずれかを入力してください。")


USER_DIRS = ["input_images", "output_images", "bg_images"]


def step_folders():
    print("\n--- 作業フォルダの作成 ---")
    for d in USER_DIRS:
        p = BASE / d
        if not p.exists():
            p.mkdir(parents=True)
            print(f"  -> 作成しました: {d}")
        else:
            print(f"  -> 既に存在します: {d}")
    print("  -> サンプル画像の同梱はありません。input_images にご自身の生成モデル等で用意した画像を置いてください。")


def step_venv():
    banner("[1/5] venv と依存パッケージの準備")
    if not PY_BIN.exists():
        print(f"venv を作成しています: {VENV}")
        venv.create(str(VENV), with_pip=True)
    run([str(PY_BIN), "-m", "pip", "install", "--upgrade", "pip"], check=True)
    run([str(PY_BIN), "-m", "pip", "install", "-r", str(REQ)], check=True)


def step_model_menu(args):
    banner("[2/5] モデル選択 (ライセンスを確認し自己責任で同意してください)")
    print("各モデルの再配布・商用利用等の条件は HuggingFace / Ollama のモデルページで")
    print("確認のうえ、同意できたもののみダウンロードされます。")
    s = load_settings()
    picks = {}
    if args.vlm:
        picks["vlm"] = args.vlm
    else:
        picks["vlm"] = choose("VLM評価モデル (画像スコアリング)", s.get("ai_model_name") or DEFAULT_VLM, VLM_OPTIONS, args.yes,
                              note="Ollama経由。GPUで実行。")
    if args.translate:
        picks["translate"] = args.translate
    else:
        picks["translate"] = choose("翻訳モデル (評価理由の英→日)", s.get("ai_translate_model") or DEFAULT_TRANSLATE, TRANSLATE_OPTIONS, args.yes,
                                    note="Ollama経由。plamo系は自動でCPU実行されます (VRAMを空けるため)。")
    picks["wd14"] = choose("WD14タガー (事前タグ付け・背景判定)", WD14_REPO, [(WD14_REPO, "")], args.yes,
                           note="sd-scripts の venv/モデル配置が前提。未導入の場合は本番実行時に OpenCV フォールバックします。")
    picks["face"] = choose("顔検出モデル (切り抜き精度)", FACE_ID, [(FACE_ID, "ultralytics系ウェイト。AGPL-3.0 の性質理解要")], args.yes)
    picks["esrgan"] = choose("Real-ESRGAN (超解像)", ESRGAN_ID, [(ESRGAN_ID, "アニメ調 x4")], args.yes,
                             note="スキップすると超解像機能が使えません。")
    picks["owlv2"] = choose("OWLv2 (物体検知: 小物切り抜き)", OWLV2_ID, [(OWLV2_ID, "ゼロショット物体検知")], args.yes,
                            note="スキップすると小物(_obj)切り出し機能が使いにくくなります。")
    picks["rembg"] = choose("背景除去モデル (rembg)", s.get("ac_rembg_model") or DEFAULT_REMBG, REMBG_OPTIONS, args.yes)
    return picks


def step_hf_weights(picks):
    banner("[3/5] HuggingFace モデルファイルの取得")
    if not PY_BIN.exists():
        print("[警告] venv がないためスキップします。")
        return
    lines = ["from huggingface_hub import hf_hub_download, snapshot_download"]
    for key in ("face", "esrgan"):
        pid = picks.get(key)
        if not pid:
            print(f"  -> {key} モデルはスキップされました (実行時に自動取得される場合あり)。")
            continue
        repo, filename = pid.split(":", 1)
        lines.append(f"hf_hub_download({repo!r}, filename={filename!r})")
    if picks.get("owlv2"):
        lines.append(f"snapshot_download({OWLV2_ID!r})")
    else:
        print("  -> OWLv2 はスキップされました。")
    if picks.get("rembg"):
        lines.append("from rembg import new_session")
        lines.append(f"new_session({picks['rembg']!r})")
    run([str(PY_BIN), "-c", "\n".join(lines)], check=False)
    print("  -> HFモデル取得処理が完了しました (失敗分は実行時に再試行されます)。")


def step_ollama_models(picks):
    banner("[4/5] Ollama モデルの取得")
    needed = [m for m in (picks.get("translate"), picks.get("vlm")) if m]
    if not needed:
        print("取得対象がありません。")
        return
    if not shutil.which("ollama"):
        print("[警告] ollama コマンドが見つかりません。https://ollama.com/download から")
        print("       インストール後、このスクリプトを再実行してください。")
        return
    for m in needed:
        if "plamo" in m:
            print(f"\n{m}: HuggingFace上の PLaMo コミュニティライセンス同意済みのアカウントが必要です。")
            print(" 未取得の場合は `ollama signin` (ollama.com連携) が必要です。")
        run(["ollama", "pull", m], check=False)


def step_wd14(picks, sd_scripts):
    banner("[5/5] sd-scripts WD14 タガーモデルの事前取得")
    if not picks.get("wd14"):
        print("スキップされました。")
        return
    if not sd_scripts:
        print("--sd-scripts を指定しなかったためスキップ (実行時に自動取得されます)。")
        return
    sd = Path(sd_scripts)
    tagger = sd / "finetune" / "tag_images_by_wd14_tagger.py"
    if not tagger.exists():
        print(f"[警告] タガースクリプトが見つかりません: {tagger}")
        return
    py_exe = sd / ("venv/Scripts/python.exe" if IS_WIN else "venv/bin/python")
    if not py_exe.exists():
        py_exe = sys.executable
    tmp = sd / "_autocropper_setup_empty"
    tmp.mkdir(exist_ok=True)
    env = dict(__import__("os").environ, PYTHONPATH=str(sd) + __import__("os").pathsep + __import__("os").environ.get("PYTHONPATH", ""))
    print("空フォルダへのタグ付けでモデルダウンロードのみ発生させます...")
    run([str(py_exe), str(tagger), "--batch_size", "1", "--model_dir", str(sd / "wd14_tagger_model"),
         "--onnx", "--repo_id", WD14_REPO, str(tmp.absolute())], check=False, cwd=str(sd), env=env)
    shutil.rmtree(tmp, ignore_errors=True)


def save_choices(s, picks):
    changed = False
    if picks.get("vlm") and picks["vlm"] != s.get("ai_model_name"):
        s["ai_model_name"] = picks["vlm"]
        changed = True
    if picks.get("translate") and picks["translate"] != s.get("ai_translate_model"):
        s["ai_translate_model"] = picks["translate"]
        changed = True
    if picks.get("rembg") and picks["rembg"] != s.get("ac_rembg_model"):
        s["ac_rembg_model"] = picks["rembg"]
        changed = True
    if changed:
        SETTINGS.write_text(json.dumps(s, ensure_ascii=False, indent=4), encoding="utf-8")
        print(f"  -> 選択結果を settings.json に保存しました。")


def main():
    parser = argparse.ArgumentParser(description="autocropper 環境構築スクリプト")
    parser.add_argument("--yes", action="store_true", help="既定値で全確認を進める")
    parser.add_argument("--vlm", default=None, help="VLM評価モデル名 (指定時は対話省略)")
    parser.add_argument("--translate", default=None, help="翻訳モデル名 (指定時は対話省略)")
    parser.add_argument("--sd-scripts", default=None, help="sd-scripts のインストール先パス")
    parser.add_argument("--skip-hf", action="store_true", help="HuggingFaceモデル取得をスキップ")
    parser.add_argument("--skip-ollama", action="store_true", help="Ollamaモデル取得をスキップ")
    args = parser.parse_args()

    step_folders()
    step_venv()
    picks = step_model_menu(args)
    if not args.skip_hf:
        step_hf_weights(picks)
    if not args.skip_ollama:
        step_ollama_models(picks)
    step_wd14(picks, args.sd_scripts)

    if shutil.which("magick"):
        print("\nImageMagick: 見つかりました (任意)。")
    else:
        print("\n[情報] ImageMagick は任意インストールです (未導入でもOpenCVで代替されます)。")

    banner("セットアップ完了")
    print(f"VLM評価モデル : {picks.get('vlm') or '(未選択: AI判定でエラーになります)'}")
    print(f"翻訳モデル    : {picks.get('translate') or '(未選択: 英語理由のままになります)'}")
    if picks.get("translate") and "plamo" in picks["translate"]:
        print("                (plamoは実行時に num_gpu=0 でCPU動作します)")
    save_choices(load_settings(), picks)
    print("run.bat / setup.bat で起動・再構成できます。")


if __name__ == "__main__":
    main()
