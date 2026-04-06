# RGT - Realtime GGUF Translate

ローカルVLM (Vision Language Model) を使って、ゲーム画面のテキストをリアルタイムでOCR＆日本語翻訳するツールです。サーバー不要、完全オフラインで動作します。

## 特徴

- **完全ローカル実行** - インターネット接続不要、プライバシー安全
- **リアルタイム翻訳** - ゲーム画面を自動キャプチャ → OCR → 翻訳 → オーバーレイ表示
- **3層キャッシュ** - 同じ画面の再処理を回避して高速化
- **コンテキスト記憶** - 直近の翻訳を記憶して一貫性のある翻訳
- **用語辞書** - ゲーム固有の固有名詞を統一翻訳
- **カスタマイズ可能なオーバーレイ** - フォント、色、位置、タイプ別表示設定

## 対応モデル

| モデル | サイズ | VRAM目安 | 状態 |
|--------|--------|----------|------|
| **Qwen3.5-4B** (推奨) | ~4.8GB | 4-6GB | 動作確認済 |
| Qwen3-VL-8B | ~6GB | 8-10GB | 動作確認済 |
| Gemma-4 E2B/E4B | ~3GB | 3-6GB | llama-cpp-python 対応待ち |
| Gemma-3 | ~3-5GB | 4-8GB | llama-cpp-python 対応待ち |

## 必要環境

- **OS**: Windows 10/11
- **Python**: 3.11
- **GPU**: NVIDIA GPU (VRAM 6GB以上推奨)
- **CUDA**: 12.x 以上

## セットアップ

### 1. リポジトリのクローン

```bash
git clone https://github.com/<your-username>/RGT.git
cd RGT
```

### 2. 仮想環境の作成

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. llama-cpp-python のインストール

VLM対応の [JamePeng フォーク版](https://github.com/JamePeng/llama-cpp-python/releases) が必要です。

CUDAバージョンに合ったwheelをダウンロードしてインストール:

```bash
pip install llama_cpp_python-<version>+cu<ver>-cp311-cp311-win_amd64.whl
```

### 4. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

### 5. モデルファイルの準備

GGUFモデルと対応するmmprojファイルをダウンロードして、任意のフォルダに配置してください。

**Qwen3.5-4B (推奨):**
- `Qwen3.5-4B-Q8_0.gguf` (本体 ~4.2GB)
- `Qwen3.5-4B-mmproj-F16.gguf` (mmproj ~0.6GB)

> HuggingFaceなどからダウンロード可能です。

### 6. 起動

```bash
python rgt.py
```

## 使い方

1. **モデル設定** - GUIでモデルファイルとmmprojファイルのパスを指定
2. **モデルロード** - 「ロード」ボタンでモデルを読み込み
3. **キャプチャ対象** - 翻訳したいゲームウィンドウを選択
4. **翻訳開始** - 「開始」ボタンでリアルタイム翻訳スタート
5. **オーバーレイ** - 翻訳結果がゲーム画面上に自動表示

### キャプチャモード

- **カーソル追従** - マウスカーソル周辺を自動キャプチャ
- **ゲームモード** - 指定ウィンドウ全体をキャプチャ

### 用語辞書

設定画面から英→日の用語辞書を登録できます。キャラクター名や地名などの固有名詞を統一的に翻訳できます。

## ビルド済みexe

[Releases](../../releases) からビルド済みexeをダウンロードできます。
Python環境なしで実行可能ですが、モデルファイルは別途必要です。

### 自分でビルドする場合

```bash
pip install pyinstaller
pyinstaller RGT.spec
```

`dist/RGT.exe` が生成されます。

## 設定ファイル

`rgt_config.json` が自動生成されます。GUI から変更した設定は自動保存されます。

## ライセンス

MIT License
