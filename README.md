# autocropper — LoRA Dataset Maker

[日本語](README.md) | [English](README.en.md)

イラスト画像から Stable Diffusion 向け LoRA 学習データセットまでの工程を、Gradio ベースの WebUI で自動化するツールです。
切り抜き・タグ付け・AI 選別・タグ整形・背景合成を 4 ステップのパイプラインとして提供します。

## パイプライン

| Step | 処理 | 実行スクリプト |
|------|------|----------------|
| Step 1 | 素材切り出し & 自動タグ付与。入力画像からキャラ素材をカテゴリ別（`_illust` / `_full` / `_upper` / `_face` / `_eye` / `_obj`）に自動クロップ。rembg 背景除去・Real-ESRGAN 超解像・目の個別切り出し・WD14 自動タグ付与 | `auto_cropper.py` → `retagger.py` |
| Step 2 | 手動選別 & AI アシスト。Ollama の VLM がクラスタ単位で OK/NG・スコアを判定し、翻訳モデルが評価理由を日本語で提示。NG 画像の一括隔離・保護・削除 | `app.py`（`ai_selector.py`） |
| Step 3 | タグ編集・クリーン。`tag_master.json` によるカテゴリ単位のタグ削除（AI に学習させる特性の分離）、一括タグ編集（Undo 付き）、Danbooru Wiki リンク | `tag_cleaner.py` |
| Step 4 | 最終構築。比率・枚数設定に沿って素材を背景画像（3 アスペクト比に自動整形）へ合成し、色補正・ブラーを適用しながら学習データセットを生成 | `dataset_builder.py` |

## 動作要件

- Windows 10/11（`setup.bat` / `run.bat` は Windows 用）
- Python 3.11 推奨（`setup.py` が venv を作成します）
- NVIDIA GPU 推奨（VLM 評価に VRAM 約 3〜8GB。翻訳モデルに plamo 系を選ぶと翻訳のみ CPU 実行され VRAM を消費しません）
- [Ollama](https://ollama.com/download)（Step 2 の AI 判定・翻訳に必要）
- [sd-scripts](https://github.com/kohya-ss/sd-scripts)（WD14 自動タグ付与に必要。未導入の場合は背景判定が OpenCV フォールバックになります）

## セットアップ

```
git clone <このリポジトリ>
cd autocropper
setup.bat
```

`setup.py`（対話型）が以下を実行します。各モデルは**ライセンスを確認し同意できたもののみ**ダウンロードされます。

1. 作業フォルダ（`input_images` / `output_images` / `bg_images`）の作成、venv 作成と `requirements.txt` のインストール
2. モデル選択メニュー（VLM / 翻訳 / WD14 / 顔検出 / Real-ESRGAN / OWLv2 / rembg）— 既定・別モデル直接入力・スキップを選択可
3. HuggingFace モデルファイルの取得
4. Ollama モデルの `ollama pull`（plamo 系は HF の PLaMo コミュニティライセンス同意済みアカウントが必要）
5. sd-scripts の WD14 タガーモデル事前取得（`--sd-scripts <パス>` 指定時）

主なオプション: `--yes`（既定値で一括同意） / `--vlm <name>` / `--translate <name>` / `--sd-scripts <path>` / `--skip-hf` / `--skip-ollama`

## 使い方

```
run.bat
```

Ollama サーバーを自動起動後、ブラウザで `http://127.0.0.1:7860` が開きます（初回起動のライブラリ読み込みに 1 分ほどかかることがあります）。

1. **Step 1** に元画像を `input_images/` に置き（元画像は各ユーザーが自身の生成モデル等で用意します。サンプルの同梱はありません）、sd-scripts パスを入力して実行 → `output_images/` にカテゴリ別素材とタグが出力されます
2. **Step 2** でフォルダを読み込み、AI 判定 → NG 隔離 or 手動選別
3. **Step 3** でキャラごとの削除カテゴリ設定と一括整理、必要ならタグの手動編集
4. **Step 4** で出力先（sd-scripts の学習用フォルダ）を指定して生成 → LoRA 学習へ

## 使用するモデル

| 用途 | 既定モデル | 備考 |
|------|-----------|------|
| VLM 評価 | `qwen3.5:9b` | `gemma3:4b` などの低 VRAM 代替可（setup.py で選択） |
| 翻訳（評価理由） | `gemma3:1b` | `plamo-2-translate`（IQ4_XS/Q4_K_M）選択時は CPU 実行で高品質 |
| タグ付与 | `SmilingWolf/wd-eva02-large-tagger-v3` | sd-scripts 経由 |
| 顔検出 | `face_yolov8n.pt` | ultralytics 系ウェイト |
| 超解像 | `RealESRGAN_x4plus_anime_6B.pth` | |
| 物体検知 | `google/owlv2-base-patch16-ensemble` | 小物 `_obj` 切り出し |
| 背景除去 | `isnet-anime` | `u2net` 選択可 |

各モデルのライセンス・利用条件はそれぞれのモデルページで確認してください。本ツールは同意を促すだけで、保証はしません。

## 設定ファイル

| ファイル | 内容 |
|----------|------|
| `settings.json` | UI の全入力値とモデル選択の保存（マシン固有のため Git 除外） |
| `tag_master.json` | タグの概念カテゴリ辞書（Step 3 のカテゴリ削除に使用） |
| `char_tag_rules.json` | キャラ別に「学習させる（= caption から削除する）」カテゴリ |
| `tag_settings.txt` | 正規表現ベースの追加/削除/保護ルール |
| `obj_prompts.txt` | 小物切り出し (OWLv2) 用の対象ワードリスト（1行1語。UI の「小物プロンプト」から編集） |
| `ai_prompt.txt` | AI 判定プロンプトの写し（UI から編集・自動保存） |

`tag_settings.txt` / `ai_prompt.txt` / `obj_prompts.txt` は UI 実行時に自動生成される引き渡しファイルで、Git には含めません（初期値は app.py に定義）。

## 注意事項

- 学習に使用する画像は、権利・利用規約を確認できるもののみを使用してください
- AI 判定はあくまでアシストです。最終判断は必ず人間が行ってください
- `filtered_by_ai/` に隔離された画像はいつでも戻せます。一括タグ編集は `.bak` で Undo 可能です

## License

本リポジトリのコードは [Apache License 2.0](LICENSE) の下で提供されます。ダウンロードされる各モデルの利用条件はこのライセンスに優先して適用されるため、各自で確認してください。
