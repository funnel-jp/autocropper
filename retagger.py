# file: retagger.py
# Phase 2: WD14 Tagger による自動タグ再生成 (クロップ処理後のキャプション生成)
import os
import sys
import argparse
import subprocess
from pathlib import Path

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

def get_args():
    parser = argparse.ArgumentParser(description="WD14 Tagger による自動タグ再生成ツール")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--sd_scripts_path", type=str, required=True)
    return parser.parse_args()

def run_retag(target_dir, sd_scripts_path):
    print("--- WD14 Tagger による自動タグ付与を開始 ---")
    script_path = os.path.join(sd_scripts_path, "finetune", "tag_images_by_wd14_tagger.py")
    if not os.path.exists(script_path):
        print(f"⚠️ sd-scriptsが見つかりません: {script_path}")
        return
        
    py_exe = str(Path(sd_scripts_path) / "venv/Scripts/python.exe")
    if not os.path.exists(py_exe):
        py_exe = "python"
        
    model_dir = str(Path(sd_scripts_path) / "wd14_tagger_model")
    target_dir_abs = str(Path(target_dir).absolute())
        
    cmd = [
        py_exe, script_path,
        "--batch_size", "4",
        "--thresh", "0.35",
        "--general_threshold", "0.35",
        "--character_threshold", "0.35",
        "--caption_extension", ".txt",
        "--model_dir", model_dir,
        "--onnx",
        "--repo_id", "SmilingWolf/wd-eva02-large-tagger-v3",
        target_dir_abs
    ]
    
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(sd_scripts_path).absolute()) + os.pathsep + env.get("PYTHONPATH", "")
    
    try:
        subprocess.run(cmd, env=env, cwd=sd_scripts_path, check=True)
    except Exception as e:
        print(f"  -> Retag実行時エラー: {e}")

def main():
    args = get_args()
    setup_dll_path(args.sd_scripts_path + "/venv")
    run_retag(args.input_dir, args.sd_scripts_path)

if __name__ == "__main__":
    main()
