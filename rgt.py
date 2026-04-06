"""
VLM リアルタイム翻訳ツール (v8.1.0 - GGUF/llama-cpp-python版)

特徴:
- llama-cpp-pythonでGGUFモデルを直接ロード（サーバー不要）
- llama.cppの高速推論をPythonから利用
- 3層キャッシュ（画像→翻訳、画像→抽出、テキスト→翻訳）で高速化
- コンテキスト記憶による翻訳の一貫性向上
- ゲーム用語辞書で固有名詞の統一翻訳
- 設定の自動保存/読み込み
- カスタマイズ可能なオーバーレイ

対応モデル:
- Qwen3.5-4B (統合VLM, 推奨)
- Qwen3-VL-8B / 2B
- Gemma-4 E2B/E4B / Gemma-3 (llama-cpp-python対応待ち)

必要条件:
1. llama-cpp-python (JamePengフォーク版):
   https://github.com/JamePeng/llama-cpp-python/releases

2. その他:
   pip install mss pynput opencv-python pillow numpy pywin32
"""

import os
import re
import time
import threading
import ctypes
import hashlib
import traceback
import tkinter as tk
from tkinter import ttk, colorchooser
import cv2
import numpy as np
import mss
import json
from pynput import mouse, keyboard
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from PIL import Image
import base64
import io

# 設定ファイルパス
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rgt_config.json")


@dataclass
class TypeOverlayConfig:
    """テキストタイプ別のオーバーレイ設定"""
    enabled: bool = True  # このタイプを表示するか
    use_custom: bool = False  # カスタム設定を使うか（Falseなら共通設定を継承）
    font_size: int = 14
    font_family: str = "Meiryo UI"
    text_color: str = "#FFFFFF"
    prefix: str = ""  # 先頭に付ける記号（選択肢なら "→ " など）
    suffix: str = ""  # 末尾に付ける記号


@dataclass
class OverlayConfig:
    """オーバーレイ設定"""
    # === 共通設定（全タイプに適用） ===
    font_size: int = 14
    font_family: str = "Meiryo UI"
    text_color: str = "#FFFFFF"
    bg_color: str = "#000000"
    opacity: float = 0.7
    position: str = "bottom"  # bottom, top, custom
    custom_x: int = 0
    custom_y: int = 0
    padding_x: int = 10
    padding_y: int = 5
    max_width: int = 600
    show_original: bool = False  # 原文も表示するか
    auto_hide: bool = True  # テキストなしで自動非表示
    
    # === タイプ別詳細設定 ===
    type_configs: Dict[str, dict] = field(default_factory=lambda: {
        "dialogue": {
            "enabled": True,
            "use_custom": False,
            "font_size": 14,
            "font_family": "Meiryo UI",
            "text_color": "#FFFFFF",
            "prefix": "",
            "suffix": ""
        },
        "choice": {
            "enabled": True,
            "use_custom": False,
            "font_size": 14,
            "font_family": "Meiryo UI",
            "text_color": "#FFFF00",  # 選択肢は黄色
            "prefix": "→ ",  # 選択肢の先頭に矢印
            "suffix": ""
        },
        "narration": {
            "enabled": True,
            "use_custom": False,
            "font_size": 14,
            "font_family": "Meiryo UI",
            "text_color": "#CCCCCC",  # ナレーションはグレー
            "prefix": "",
            "suffix": ""
        }
    })
    
    def get_type_config(self, text_type: str) -> TypeOverlayConfig:
        """指定タイプの設定を取得（存在しなければデフォルト）"""
        if text_type in self.type_configs:
            return TypeOverlayConfig(**self.type_configs[text_type])
        return TypeOverlayConfig()
    
    def set_type_config(self, text_type: str, config: TypeOverlayConfig):
        """指定タイプの設定を保存"""
        self.type_configs[text_type] = asdict(config)


@dataclass 
class AppConfig:
    """アプリ全体の設定"""
    # モデル設定
    model_path: str = ""
    mmproj_path: str = ""
    n_gpu_layers: int = -1
    n_ctx: int = 4096
    handler: str = "auto"  # VLMハンドラ（autoで自動検出）
    image_max_size: int = 0  # 画像リサイズ (0=なし, 512, 768, 1024等)
    
    # キャプチャ設定
    capture_mode: str = "cursor"
    capture_width: int = 800
    capture_height: int = 400
    interval: float = 1.0
    
    # オーバーレイ
    show_overlay: bool = True
    overlay: OverlayConfig = field(default_factory=OverlayConfig)
    
    # OCR補正辞書（ユーザー追加分）
    ocr_corrections: Dict[str, str] = field(default_factory=dict)

    # 用語辞書（ゲーム固有の英→日翻訳）
    term_dictionary: Dict[str, str] = field(default_factory=dict)

    # コンテキスト記憶数（直近の翻訳を記憶して一貫性を向上）
    context_memory_size: int = 10
    
    def save(self, path: str = CONFIG_FILE):
        """設定を保存"""
        try:
            data = asdict(self)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"[Config] Saved to {path}")
        except Exception as e:
            print(f"[Config] Save error: {e}")
    
    @classmethod
    def load(cls, path: str = CONFIG_FILE) -> 'AppConfig':
        """設定を読み込み"""
        try:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # OverlayConfigを復元
                if 'overlay' in data and isinstance(data['overlay'], dict):
                    overlay_data = data['overlay']
                    # type_configsがなければデフォルトを使用
                    if 'type_configs' not in overlay_data:
                        overlay_data['type_configs'] = OverlayConfig().type_configs
                    data['overlay'] = OverlayConfig(**overlay_data)
                # 未知フィールドをフィルタリング（後方互換）
                import dataclasses as _dc
                known_fields = {f.name for f in _dc.fields(cls)}
                filtered_data = {k: v for k, v in data.items() if k in known_fields}
                print(f"[Config] Loaded from {path}")
                return cls(**filtered_data)
        except Exception as e:
            print(f"[Config] Load error: {e}")
        return cls()

# Windows API
try:
    import ctypes.wintypes
    user32 = ctypes.windll.user32
    dwmapi = ctypes.windll.dwmapi
    HAS_WIN32 = True
except Exception:
    HAS_WIN32 = False

# PyWin32 (PrintWindow用)
try:
    import win32gui
    import win32ui
    import win32con
    HAS_PYWIN32 = True
except ImportError:
    HAS_PYWIN32 = False
    print("[警告] pywin32がインストールされていません。pip install pywin32")

# llama-cpp-python
# 対応VLMハンドラ
AVAILABLE_HANDLERS = {}
HANDLER_INFO = {
    'Qwen35': {
        'name': 'Qwen3.5',
        'description': '統合VLM。OCR・翻訳高品質。4B。VRAM 4-6GB推奨',
        'priority': 'quality'
    },
    'Qwen3VL': {
        'name': 'Qwen3-VL',
        'description': 'OCR精度・翻訳品質トップクラス。8B。VRAM 8-10GB推奨',
        'priority': 'quality'
    },
    'Gemma3': {
        'name': 'Gemma-4 / Gemma-3',
        'description': 'Google製VLM。Gemma-4 E2B/E4B・Gemma-3対応。VRAM 3-6GB推奨',
        'priority': 'speed'
    },
}

try:
    from llama_cpp import Llama
    HAS_LLAMA_CPP = True

    try:
        import llama_cpp.llama_chat_format as chat_format

        # Qwen3.5（統合VLM - 推奨）
        if hasattr(chat_format, 'Qwen35ChatHandler'):
            AVAILABLE_HANDLERS['Qwen35'] = chat_format.Qwen35ChatHandler
            print("[VLM] Qwen3.5 ハンドラ検出OK")

        # Qwen3-VL（レガシー）
        if hasattr(chat_format, 'Qwen3VLChatHandler'):
            AVAILABLE_HANDLERS['Qwen3VL'] = chat_format.Qwen3VLChatHandler
            print("[VLM] Qwen3-VL ハンドラ検出OK")

        # Gemma-3 / Gemma-4 (E2B/E4B)
        if hasattr(chat_format, 'Gemma3ChatHandler'):
            AVAILABLE_HANDLERS['Gemma3'] = chat_format.Gemma3ChatHandler
            print("[VLM] Gemma-3/4 ハンドラ検出OK")

        if not AVAILABLE_HANDLERS:
            print("[VLM] 警告: 対応ハンドラが見つかりません")

    except Exception as e:
        print(f"[VLM] ハンドラ検出エラー: {e}")

except ImportError:
    HAS_LLAMA_CPP = False
    print("[エラー] llama-cpp-pythonがインストールされていません")
    print("")
    print("JamePengフォーク版をインストールしてください:")
    print("  https://github.com/JamePeng/llama-cpp-python/releases")


def detect_handler_from_model(model_path: str) -> Optional[str]:
    """モデルファイル名からハンドラを推測"""
    name = os.path.basename(model_path).lower()
    if 'qwen3.5' in name or 'qwen35' in name:
        return 'Qwen35' if 'Qwen35' in AVAILABLE_HANDLERS else None
    if 'qwen3' in name and 'vl' in name:
        return 'Qwen3VL' if 'Qwen3VL' in AVAILABLE_HANDLERS else None
    if 'gemma' in name:
        return 'Gemma3' if 'Gemma3' in AVAILABLE_HANDLERS else None
    # デフォルト: 利用可能な最優先ハンドラ
    for h in ['Qwen35', 'Qwen3VL', 'Gemma3']:
        if h in AVAILABLE_HANDLERS:
            return h
    return None


# ---------------------------------------------------------
# ウィンドウヘルパー
# ---------------------------------------------------------
def get_window_list(include_hidden=False):
    """ウィンドウ一覧を取得
    
    include_hidden=True で非表示ウィンドウも含める（ゲーム用）
    """
    if not HAS_WIN32:
        return []
    windows = []
    
    # プロセス名取得用
    kernel32 = ctypes.windll.kernel32
    psapi = ctypes.windll.psapi
    PROCESS_QUERY_INFORMATION = 0x0400
    PROCESS_VM_READ = 0x0010
    
    def enum_callback(hwnd, _):
        # タイトル取得
        length = user32.GetWindowTextLengthW(hwnd)
        if length > 0:
            buff = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buff, length + 1)
            title = buff.value
            
            if title and title.strip() and title not in ['Program Manager', 'Windows Input Experience', 'MSCTFIME UI', 'Default IME']:
                # 可視チェック
                is_visible = user32.IsWindowVisible(hwnd)
                
                if is_visible or include_hidden:
                    # プロセス名も取得
                    process_name = ""
                    try:
                        pid = ctypes.wintypes.DWORD()
                        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                        
                        handle = kernel32.OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, pid.value)
                        if handle:
                            try:
                                name_buf = ctypes.create_unicode_buffer(260)
                                if psapi.GetModuleBaseNameW(handle, None, name_buf, 260):
                                    process_name = name_buf.value
                            finally:
                                kernel32.CloseHandle(handle)
                    except Exception:
                        pass

                    windows.append((hwnd, title, process_name, is_visible))
        return True
    
    WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)
    user32.EnumWindows(WNDENUMPROC(enum_callback), 0)
    return windows


def get_all_processes_with_windows():
    """全プロセスからウィンドウを持つものを取得（タスクマネージャー風）"""
    if not HAS_WIN32:
        return []
    
    try:
        psapi = ctypes.windll.psapi
        kernel32 = ctypes.windll.kernel32
        
        # 全プロセスIDを取得
        process_ids = (ctypes.wintypes.DWORD * 1024)()
        bytes_returned = ctypes.wintypes.DWORD()
        
        if not psapi.EnumProcesses(ctypes.byref(process_ids), ctypes.sizeof(process_ids), ctypes.byref(bytes_returned)):
            return []
        
        num_processes = bytes_returned.value // ctypes.sizeof(ctypes.wintypes.DWORD)
        
        processes = []
        PROCESS_QUERY_INFORMATION = 0x0400
        PROCESS_VM_READ = 0x0010
        
        for i in range(num_processes):
            pid = process_ids[i]
            if pid == 0:
                continue
            
            handle = kernel32.OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, pid)
            if handle:
                try:
                    name_buf = ctypes.create_unicode_buffer(260)
                    if psapi.GetModuleBaseNameW(handle, None, name_buf, 260):
                        process_name = name_buf.value
                        if process_name and not process_name.startswith('svchost'):
                            # このプロセスのウィンドウを探す
                            windows_for_process = []
                            current_pid = pid  # クロージャ用
                            
                            def find_windows(hwnd, _):
                                window_pid = ctypes.wintypes.DWORD()
                                user32.GetWindowThreadProcessId(hwnd, ctypes.byref(window_pid))
                                if window_pid.value == current_pid:
                                    length = user32.GetWindowTextLengthW(hwnd)
                                    if length > 0:
                                        buff = ctypes.create_unicode_buffer(length + 1)
                                        user32.GetWindowTextW(hwnd, buff, length + 1)
                                        title = buff.value
                                        if title and title.strip():
                                            windows_for_process.append((hwnd, title))
                                return True
                            
                            WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)
                            user32.EnumWindows(WNDENUMPROC(find_windows), 0)
                            
                            if windows_for_process:
                                for hwnd, title in windows_for_process:
                                    processes.append((hwnd, title, process_name, pid))
                finally:
                    kernel32.CloseHandle(handle)
        
        return processes
    except Exception as e:
        print(f"[Window] Process enum error: {e}")
        return []


def find_window_by_process_name(process_name: str):
    """プロセス名からウィンドウを検索"""
    processes = get_all_processes_with_windows()
    for hwnd, title, pname, pid in processes:
        if process_name.lower() in pname.lower():
            return hwnd, title
    return None, None


def get_window_rect(hwnd):
    if not HAS_WIN32:
        return None
    rect = ctypes.wintypes.RECT()
    if user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        return (rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top)
    return None


def get_client_rect(hwnd):
    if not HAS_WIN32:
        return None
    client_rect = ctypes.wintypes.RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(client_rect)):
        return None
    point = ctypes.wintypes.POINT(0, 0)
    if not ctypes.windll.user32.ClientToScreen(hwnd, ctypes.byref(point)):
        return None
    return (point.x, point.y, client_rect.right, client_rect.bottom)


# 高DPI対応
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


# ---------------------------------------------------------
# VLM クライアント (llama-cpp-python版)
# ---------------------------------------------------------
@dataclass
class TextBlock:
    """抽出されたテキストブロック"""
    type: str
    original: str
    translated: str


class VLMClient:
    """llama-cpp-pythonでGGUFを直接ロード（複数VLMモデル対応）"""
    
    def __init__(self):
        self.llm = None
        self.chat_handler = None
        self.initialized = False
        self.model_path = ""
        self.mmproj_path = ""
        self.handler_name = ""  # 使用中のハンドラ名
        
        # 画像リサイズ設定
        self.image_max_size = 0  # 0=リサイズなし, それ以外は最大辺のピクセル数
        
        # OCR補正辞書（フォントの癖による誤認識を補正）
        # ユーザーが必要に応じて追加（初期は空、LLMに任せる）
        self.ocr_corrections = {}
        
        # ============================================
        # プロンプト定義（Qwen3-VL専用）
        # ============================================
        
        self.prompt_extract_game = """You are an OCR tool for game screenshots.
Extract ONLY dialogue, narration, story text, and player choices.
Ignore all UI elements, buttons, menus, HUD, version numbers, labels.

You MUST respond with ONLY a JSON object, no other text:
{{"texts":["extracted text 1","extracted text 2"]}}

If no relevant text found:
{{"texts":[]}}"""

        self.prompt_translate_game = """Translate game text to Japanese. Output JSON only.

Type rules:
- narration: Descriptions, system text, 3rd person, ALL CAPS
- dialogue: Character speech, 1st/2nd person pronouns
- choice: Short options (1-4 words), imperative verbs

Fix OCR errors. Natural game-style Japanese.
{term_dict_section}{context_section}
Input:
{texts}

Output format example:
{{"blocks":[{{"type":"narration","original":"The door opened.","translated":"扉が開いた。"}},{{"type":"dialogue","original":"Who are you?","translated":"お前は誰だ？"}}]}}

Now translate the input above as JSON:"""
        
        self.prompt_extract_general = """Extract all readable text from this screenshot.
Return JSON only:
{"texts":["text1","text2"]}"""
        
        # 現在使用するプロンプト（デフォルトはゲームモード）
        self.prompt_extract = self.prompt_extract_game
        self.prompt_translate = self.prompt_translate_game
        self.game_mode = True  # ゲームモードかどうか
        
        # 選択肢検出パターンは _apply_choice_heuristic 内で定義
    
    def set_game_mode(self, is_game: bool):
        """ゲームモード/汎用モードを切り替え"""
        self.game_mode = is_game
        
        if is_game:
            self.prompt_extract = self.prompt_extract_game
            self.prompt_translate = self.prompt_translate_game
            print("[VLM] GAME mode")
        else:
            self.prompt_extract = self.prompt_extract_general
            self.prompt_translate = self.prompt_translate_game  # 翻訳は共通
            print("[VLM] GENERAL mode")
    
    def correct_ocr_text(self, text: str) -> str:
        """OCR誤認識を補正"""
        corrected = text
        corrections_made = []
        
        for wrong, correct in self.ocr_corrections.items():
            # 大文字小文字を保持しながら置換
            pattern = re.compile(re.escape(wrong), re.IGNORECASE)
            
            def replace_match(match):
                original = match.group(0)
                # 元の大文字小文字パターンを保持
                if original.isupper():
                    return correct.upper()
                elif original[0].isupper():
                    return correct.capitalize()
                else:
                    return correct
            
            new_text = pattern.sub(replace_match, corrected)
            if new_text != corrected:
                corrections_made.append(f"{wrong}->{correct}")
                corrected = new_text
        
        if corrections_made:
            print(f"[OCR Fix] {', '.join(corrections_made)}")
        
        return corrected
    
    def add_ocr_correction(self, wrong: str, correct: str):
        """OCR補正辞書に追加"""
        self.ocr_corrections[wrong.lower()] = correct.lower()
        print(f"[OCR] Added correction: {wrong} -> {correct}")
    
    def load_model(self, model_path: str, mmproj_path: str, 
                   n_gpu_layers: int = -1, n_ctx: int = 4096,
                   handler_name: str = "auto") -> Tuple[bool, str]:
        """モデルをロード
        
        Args:
            handler_name: "auto", "Qwen35", "Qwen3VL", "Gemma3"
        """
        if not HAS_LLAMA_CPP:
            return False, "llama-cpp-pythonがインストールされていません"
        
        if not AVAILABLE_HANDLERS:
            return False, "VLMハンドラが見つかりません。JamePengフォーク版をインストールしてください"
        
        try:
            self.model_path = model_path
            self.mmproj_path = mmproj_path
            
            print(f"[VLM] Loading model: {model_path}")
            print(f"[VLM] Loading mmproj: {mmproj_path}")
            print(f"[VLM] GPU layers: {n_gpu_layers}, Context: {n_ctx}")
            
            # ファイル存在確認
            if not os.path.exists(model_path):
                return False, f"モデルファイルが見つかりません: {model_path}"
            if not os.path.exists(mmproj_path):
                return False, f"mmprojファイルが見つかりません: {mmproj_path}"
            
            # ハンドラを決定
            if handler_name == "auto":
                detected = detect_handler_from_model(model_path)
                if detected and detected in AVAILABLE_HANDLERS:
                    handler_name = detected
                    print(f"[VLM] 自動検出: {handler_name}")
                else:
                    # デフォルト優先順位: Qwen35 > Qwen3VL > Gemma3
                    for h in ['Qwen35', 'Qwen3VL', 'Gemma3']:
                        if h in AVAILABLE_HANDLERS:
                            handler_name = h
                            break
                    print(f"[VLM] デフォルト使用: {handler_name}")
            
            if handler_name not in AVAILABLE_HANDLERS:
                available = list(AVAILABLE_HANDLERS.keys())
                return False, f"ハンドラ '{handler_name}' が見つかりません。利用可能: {available}"
            
            # チャットハンドラを作成
            handler_class = AVAILABLE_HANDLERS[handler_name]
            print(f"[VLM] Creating {handler_class.__name__}...")
            handler_kwargs = dict(
                clip_model_path=mmproj_path,
                verbose=False
            )
            # Qwen3.5: thinking無効化（翻訳速度優先）
            if handler_name == 'Qwen35':
                handler_kwargs['enable_thinking'] = False
            self.chat_handler = handler_class(**handler_kwargs)
            self.handler_name = handler_name
            
            # モデルをロード
            print("[VLM] Loading LLM...")
            self.llm = Llama(
                model_path=model_path,
                chat_handler=self.chat_handler,
                n_gpu_layers=n_gpu_layers,
                n_ctx=n_ctx,
                verbose=False,
            )
            
            self.initialized = True
            
            # モデルに合わせてプロンプトを設定
            self.set_game_mode(self.game_mode)
            
            info = HANDLER_INFO.get(handler_name, {})
            priority = info.get('priority', 'unknown')
            print(f"[VLM] ロード完了! ({handler_name} - {priority})")
            
            return True, f"{info.get('name', handler_name)}"
            
        except Exception as e:
            traceback.print_exc()
            return False, f"モデルロード失敗: {e}"
    
    def set_image_max_size(self, max_size: int):
        """画像の最大サイズを設定 (0=リサイズなし)"""
        self.image_max_size = max_size
        if max_size > 0:
            print(f"[VLM] 画像リサイズ: 最大 {max_size}px")
        else:
            print("[VLM] 画像リサイズ: オフ")
    
    def _image_to_base64(self, image: np.ndarray) -> str:
        """OpenCV画像をbase64に変換（リサイズ対応）"""
        # BGR -> RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        
        # リサイズ（設定されている場合）
        if self.image_max_size > 0:
            w, h = pil_img.size
            if w > self.image_max_size or h > self.image_max_size:
                # アスペクト比を維持してリサイズ
                if w > h:
                    new_w = self.image_max_size
                    new_h = int(h * self.image_max_size / w)
                else:
                    new_h = self.image_max_size
                    new_w = int(w * self.image_max_size / h)
                pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
                # print(f"[VLM] Resized: {w}x{h} -> {new_w}x{new_h}")
        
        # JPEG圧縮
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def extract_text(self, image: np.ndarray) -> Optional[List[str]]:
        """画像からテキスト抽出"""
        if not self.initialized:
            return None
        
        try:
            img_base64 = self._image_to_base64(image)
            
            # llama-cpp-pythonのchat completion形式
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_base64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": self.prompt_extract
                        }
                    ]
                }
            ]
            
            start_time = time.time()
            
            response = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=512,
                temperature=0.1,
            )
            
            elapsed = time.time() - start_time
            print(f"[VLM] Extract time: {elapsed:.2f}s")
            
            text = response['choices'][0]['message']['content']
            print(f"[VLM] Raw response: {text[:300]}")
            
            return self._parse_extract_response(text)
            
        except Exception as e:
            print(f"[VLM] Extract error: {e}")
            traceback.print_exc()
            return None
    
    def _parse_extract_response(self, text: str) -> List[str]:
        """抽出レスポンスをパース"""
        text = text.strip()
        
        # マークダウンコードブロック除去
        if text.startswith("```"):
            lines = text.split('\n')
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            text = '\n'.join(lines).strip()
        
        # JSON形式をパース
        if '{' in text:
            start = text.find('{')
            end = text.rfind('}') + 1
            json_str = text[start:end]
            
            if json_str and len(json_str) > 2:
                try:
                    data = json.loads(json_str)
                    texts = data.get('texts', [])
                    # OCR補正を適用
                    texts = [self.correct_ocr_text(t) for t in texts]
                    if texts:
                        print(f"[VLM] Extracted {len(texts)} texts")
                    return texts
                except json.JSONDecodeError as e:
                    print(f"[VLM] JSON parse failed: {e}")
        
        # フォールバック: 行ごとに抽出
        lines = []
        for l in text.split('\n'):
            l = l.strip()
            if l and not l.startswith(('{', '}', '[', ']', '"texts"')):
                if l.startswith(('-', '•', '*')):
                    l = l[1:].strip()
                if len(l) > 2:
                    lines.append(self.correct_ocr_text(l))
        
        if lines:
            print(f"[VLM] Fallback: extracted {len(lines)} lines")
            return lines
        
        return []
    
    def translate_text(self, texts: List[str],
                       context: Optional['TranslationContext'] = None,
                       term_dict: Optional['TermDictionary'] = None) -> List[TextBlock]:
        """テキストを翻訳（コンテキスト・用語辞書対応）"""
        if not self.initialized or not texts:
            return []

        try:
            # 動的プロンプト組み立て
            term_section = ""
            if term_dict and term_dict.terms:
                term_section = f"\nTERMINOLOGY (use these translations):\n{term_dict.get_prompt_string()}\n"

            context_section = ""
            if context and context.entries:
                context_section = f"\nRECENT TRANSLATIONS (for consistency):\n{context.get_context_string()}\n"

            texts_str = "\n".join([f"- {t}" for t in texts])
            prompt = self.prompt_translate.format(
                texts=texts_str,
                term_dict_section=term_section,
                context_section=context_section
            )

            messages = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]

            start_time = time.time()
            print(f"[LLM] Translating {len(texts)} texts...")

            response = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=2048,
                temperature=0.2,
            )
            
            elapsed = time.time() - start_time
            print(f"[LLM] Translate time: {elapsed:.2f}s")
            
            text = response['choices'][0]['message']['content']
            print(f"[LLM] Raw response: {text[:300]}")
            
            return self._parse_translation_response(text, texts)
            
        except Exception as e:
            print(f"[LLM] Translate error: {e}")
            traceback.print_exc()
            return []
    
    def _parse_translation_response(self, text: str, original_texts: List[str]) -> List[TextBlock]:
        """翻訳レスポンスをパース"""
        blocks = []
        
        try:
            text = text.strip()
            
            if text.startswith("```"):
                lines = text.split('\n')
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == '```':
                    lines = lines[:-1]
                text = '\n'.join(lines).strip()
            
            # JSON形式をパース
            if '{' in text:
                start = text.find('{')
                end = text.rfind('}') + 1
                json_str = text[start:end]
                
                if json_str and len(json_str) > 2:
                    try:
                        data = json.loads(json_str)
                        
                        for item in data.get('blocks', []):
                            block = TextBlock(
                                type=item.get('type', 'other'),
                                original=item.get('original', ''),
                                translated=item.get('translated', '')
                            )
                            # ヒューリスティックで選択肢判定を補正
                            block = self._apply_choice_heuristic(block)
                            blocks.append(block)
                        
                        if blocks:
                            return blocks
                    except json.JSONDecodeError as e:
                        print(f"[LLM] JSON parse error: {e}")
            
            # フォールバック: JSONパース失敗時、応答テキストから日本語部分を抽出
            print(f"[LLM] JSON parse failed, trying fallback extraction")

            # 応答が日本語を含む場合、それを翻訳結果として使用
            jp_lines = []
            for line in text.split('\n'):
                line = line.strip()
                # 日本語文字（ひらがな・カタカナ・漢字）を含む行を抽出
                if line and re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]', line):
                    # 先頭の記号や番号を除去
                    cleaned = re.sub(r'^[\-\*\d\.\)]+\s*', '', line).strip()
                    # 引用符を除去
                    cleaned = cleaned.strip('"\'')
                    if cleaned:
                        jp_lines.append(cleaned)

            if jp_lines:
                # 日本語行があれば、原文と対応付け
                for i, t in enumerate(original_texts):
                    translated = jp_lines[i] if i < len(jp_lines) else jp_lines[-1]
                    blocks.append(TextBlock(
                        type="dialogue",
                        original=t,
                        translated=translated
                    ))
            else:
                # 日本語が全くない場合は空を返す（「翻訳失敗」は表示しない）
                print(f"[LLM] No Japanese text found in response, skipping")
                return []

        except Exception as e:
            print(f"[LLM] Parse error: {e}")
            return []
        
        return blocks
    
    def _apply_choice_heuristic(self, block: TextBlock) -> TextBlock:
        """ヒューリスティックで選択肢かどうかを判定・補正"""
        original = block.original.strip()
        original_lower = original.lower()
        
        # === choiceと判定されたものを検証 ===
        if block.type == "choice":
            # 明らかに選択肢ではないパターン（NPCの発言っぽいもの）
            not_choice_patterns = [
                # 命令・勧誘形（NPCがプレイヤーに言う）
                r"^(go ahead|come on|let's|you gotta|ya gotta|you should|you need to|you have to)\b",
                r"\b(give it a (go|try|shot))\b",
                r"\b(try it|do it|check it out)\b",
                # 長い文（選択肢は通常短い）
                # カンマで区切られた複合文
            ]
            
            for pattern in not_choice_patterns:
                if re.search(pattern, original_lower, re.IGNORECASE):
                    print(f"[Heuristic] '{original[:30]}...' -> dialogue (not a choice)")
                    return TextBlock(
                        type="dialogue",
                        original=block.original,
                        translated=block.translated
                    )
            
            # 長すぎる文は選択肢ではない（40文字以上）
            if len(original) > 40:
                print(f"[Heuristic] '{original[:30]}...' -> dialogue (too long for choice)")
                return TextBlock(
                    type="dialogue",
                    original=block.original,
                    translated=block.translated
                )
            
            return block
        
        # === dialogue/narrationを選択肢に補正 ===
        # 短いテキスト（35文字以下）で以下のパターンに当てはまれば選択肢
        if len(original) <= 35:
            choice_patterns = [
                # 疑問文パターン（プレイヤーが聞く質問）- 主語がない or "I"
                r"^(who|what|where|when|why|how)\s+(are you|is (this|that|it)|do you|did you)",
                r"^(do you|are you|can i|will you|should i|may i|could you)\b",
                # Yes/No系・短い応答（単語のみ）
                r"^(yes|no|ok|okay|sure|fine|accept|decline|agree|refuse|cancel)[.!?]?$",
                r"^(leave|stay|fight|run|flee|help|ignore|wait|stop|go|enter|exit)[.!?]?$",
                # 「〜について聞く」系
                r"^ask (about|him|her|them)",
            ]
            
            for pattern in choice_patterns:
                if re.search(pattern, original_lower, re.IGNORECASE):
                    print(f"[Heuristic] '{original[:30]}...' -> choice (pattern match)")
                    return TextBlock(
                        type="choice",
                        original=block.original,
                        translated=block.translated
                    )
        
        return block


# ---------------------------------------------------------
# キャッシュ
# ---------------------------------------------------------
class ImageCache:
    """3層キャッシュ: 画像→翻訳, 画像→抽出, テキスト→翻訳"""

    def __init__(self, max_size: int = 100):
        self.translation_cache: Dict[str, List[TextBlock]] = {}  # 画像ハッシュ→翻訳済みブロック
        self.extraction_cache: Dict[str, List[str]] = {}  # 画像ハッシュ→抽出テキスト
        self.text_translation_cache: Dict[str, List[TextBlock]] = {}  # テキスト内容→翻訳済みブロック
        self.max_size = max_size
        self.access_order: List[str] = []

    def _compute_hash(self, image: np.ndarray) -> str:
        small = cv2.resize(image, (48, 48))
        return hashlib.md5(small.tobytes()).hexdigest()

    def _touch(self, h: str):
        if h in self.access_order:
            self.access_order.remove(h)
        self.access_order.append(h)

    def _evict_if_full(self):
        while len(self.access_order) >= self.max_size and self.access_order:
            old = self.access_order.pop(0)
            self.translation_cache.pop(old, None)
            self.extraction_cache.pop(old, None)

    # --- 画像→翻訳済みブロック ---
    def get_translations(self, image: np.ndarray) -> Optional[List[TextBlock]]:
        h = self._compute_hash(image)
        if h in self.translation_cache:
            self._touch(h)
            return self.translation_cache[h]
        return None

    def set_translations(self, image: np.ndarray, blocks: List[TextBlock]):
        h = self._compute_hash(image)
        self._evict_if_full()
        self.translation_cache[h] = blocks
        self._touch(h)

    # --- 画像→抽出テキスト ---
    def get_extractions(self, image: np.ndarray) -> Optional[List[str]]:
        h = self._compute_hash(image)
        if h in self.extraction_cache:
            self._touch(h)
            return self.extraction_cache[h]
        return None

    def set_extractions(self, image: np.ndarray, texts: List[str]):
        h = self._compute_hash(image)
        self._evict_if_full()
        self.extraction_cache[h] = texts
        self._touch(h)

    # --- テキスト内容→翻訳済みブロック ---
    def get_by_text(self, texts: List[str]) -> Optional[List[TextBlock]]:
        key = "||".join(sorted(texts))
        return self.text_translation_cache.get(key)

    def set_by_text(self, texts: List[str], blocks: List[TextBlock]):
        key = "||".join(sorted(texts))
        self.text_translation_cache[key] = blocks
        # テキストキャッシュは別途制限（最大200エントリ）
        if len(self.text_translation_cache) > 200:
            # 古い半分を削除
            keys = list(self.text_translation_cache.keys())
            for k in keys[:100]:
                del self.text_translation_cache[k]

    # --- 後方互換（旧APIをマッピング） ---
    def get(self, image: np.ndarray) -> Optional[List[TextBlock]]:
        return self.get_translations(image)

    def set(self, image: np.ndarray, blocks: List[TextBlock]):
        self.set_translations(image, blocks)


class TranslationContext:
    """直近の翻訳結果を記憶してコンテキストの一貫性を保つ"""

    def __init__(self, max_entries: int = 10):
        self.max_entries = max_entries
        self.entries: List[Dict[str, str]] = []
        self.empty_count = 0  # 空テキスト連続カウント

    def add(self, blocks: List['TextBlock']):
        """翻訳結果を記憶に追加"""
        for b in blocks:
            if b.translated.strip():
                self.entries.append({
                    "original": b.original,
                    "translated": b.translated,
                    "type": b.type
                })
        # 最大数を超えたら古いものを削除
        while len(self.entries) > self.max_entries:
            self.entries.pop(0)
        self.empty_count = 0

    def on_empty(self):
        """テキストが空だったときに呼ぶ。連続5回でコンテキストクリア"""
        self.empty_count += 1
        if self.empty_count >= 5:
            self.clear()

    def get_context_string(self, max_items: int = 5) -> str:
        """プロンプト注入用のコンテキスト文字列を生成"""
        if not self.entries:
            return ""
        recent = self.entries[-max_items:]
        lines = []
        for e in recent:
            lines.append(f'  "{e["original"]}" -> "{e["translated"]}"')
        return "\n".join(lines)

    def clear(self):
        self.entries.clear()
        self.empty_count = 0


class TermDictionary:
    """ゲーム固有の用語辞書（英→日）"""

    def __init__(self):
        self.terms: Dict[str, str] = {}

    def add(self, english: str, japanese: str):
        self.terms[english.strip()] = japanese.strip()

    def remove(self, english: str):
        self.terms.pop(english.strip(), None)

    def get_prompt_string(self) -> str:
        """プロンプト注入用の用語辞書文字列を生成"""
        if not self.terms:
            return ""
        lines = [f'  "{en}" = "{ja}"' for en, ja in self.terms.items()]
        return "\n".join(lines)


# ---------------------------------------------------------
# 翻訳オーバーレイ（カスタマイズ対応）
# ---------------------------------------------------------
class TranslationOverlay:
    def __init__(self, parent, config: OverlayConfig = None):
        self.parent = parent
        self.window = None
        self.label = None
        self.hwnd = None
        self.config = config or OverlayConfig()
        self.last_capture_region = None
        self.is_visible = False
        
    def create(self):
        if self.window:
            return
        self.window = tk.Toplevel(self.parent)
        self.window.title("Translation Overlay")
        self.window.overrideredirect(True)
        self.window.attributes('-topmost', True)
        self.window.attributes('-alpha', self.config.opacity)
        self.window.configure(bg=self.config.bg_color)
        self.label = tk.Label(
            self.window,
            text="",
            font=(self.config.font_family, self.config.font_size, "bold"),
            fg=self.config.text_color,
            bg=self.config.bg_color,
            wraplength=self.config.max_width,
            justify=tk.LEFT,
            padx=self.config.padding_x,
            pady=self.config.padding_y
        )
        self.label.pack()
        self.window.geometry("+50+50")
        self.window.withdraw()  # 最初は非表示
        self.window.update()
        try:
            self.hwnd = int(self.window.wm_frame(), 16)
            self._set_click_through(self.hwnd)
        except Exception as e:
            print(f"[Overlay] Setup Error: {e}")

    def _set_click_through(self, hwnd):
        if not HAS_WIN32: return
        try:
            GWL_EXSTYLE = -20
            WS_EX_LAYERED = 0x00080000
            WS_EX_TRANSPARENT = 0x00000020
            WS_EX_NOACTIVATE = 0x08000000
            style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_NOACTIVATE)
        except Exception:
            pass

    def _force_topmost(self):
        if not HAS_WIN32 or not self.hwnd: return
        try:
            HWND_TOPMOST = -1
            SWP_NOMOVE = 0x0002
            SWP_NOSIZE = 0x0001
            SWP_NOACTIVATE = 0x0010
            SWP_SHOWWINDOW = 0x0040
            user32.SetWindowPos(self.hwnd, HWND_TOPMOST, 0, 0, 0, 0,
                                SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_SHOWWINDOW)
        except Exception:
            pass

    def apply_config(self, config: OverlayConfig):
        """設定を適用"""
        self.config = config
        if self.window:
            self.window.attributes('-alpha', config.opacity)
            self.window.configure(bg=config.bg_color)
            self.label.configure(
                font=(config.font_family, config.font_size, "bold"),
                fg=config.text_color,
                bg=config.bg_color,
                wraplength=config.max_width,
                padx=config.padding_x,
                pady=config.padding_y
            )

    def update(self, blocks: List[TextBlock], capture_region=None):
        if not self.window:
            self.create()
        
        self.last_capture_region = capture_region
        
        # テキスト作成（タイプ別設定を適用）
        text = ""
        if blocks:
            lines = []
            for b in blocks:
                if b.translated.strip():
                    # タイプ別設定を取得
                    type_cfg = self.config.get_type_config(b.type)
                    
                    # このタイプが無効化されていればスキップ
                    if not type_cfg.enabled:
                        continue
                    
                    # プレフィックス・サフィックスを適用
                    prefix = type_cfg.prefix if type_cfg.use_custom else ""
                    suffix = type_cfg.suffix if type_cfg.use_custom else ""
                    
                    # 選択肢は常にプレフィックス（use_customがFalseでもデフォルトprefixがあれば適用）
                    if b.type == "choice" and not prefix:
                        prefix = self.config.type_configs.get("choice", {}).get("prefix", "")
                    
                    translated_text = f"{prefix}{b.translated}{suffix}"
                    
                    if self.config.show_original:
                        lines.append(f"{b.original}\n→ {translated_text}")
                    else:
                        lines.append(translated_text)
            text = "\n".join(lines)
        
        if not text:
            # テキストなし → 非表示
            if self.config.auto_hide:
                self.hide()
            return
        
        self.label.config(text=text)
        self.window.update_idletasks()
        
        # 位置計算
        req_w = self.label.winfo_reqwidth()
        req_h = self.label.winfo_reqheight()
        
        if self.config.position == "custom":
            new_x = self.config.custom_x
            new_y = self.config.custom_y
        elif capture_region:
            cx, cy, cw, ch = capture_region
            if self.config.position == "bottom":
                new_x = cx
                new_y = cy + ch + 5
            elif self.config.position == "top":
                new_x = cx
                new_y = cy - req_h - 5
            else:  # bottom default
                new_x = cx
                new_y = cy + ch + 5
        else:
            new_x = 50
            new_y = 50
        
        self.window.geometry(f"{req_w}x{req_h}+{new_x}+{new_y}")
        self.show()
        
    def show(self):
        if self.window and not self.is_visible:
            self.window.deiconify()
            self._force_topmost()
            self.is_visible = True
    
    def hide(self):
        if self.window and self.is_visible:
            self.window.withdraw()
            self.is_visible = False
            
    def destroy(self):
        if self.window:
            self.window.destroy()
            self.window = None
            self.is_visible = False


# ---------------------------------------------------------
# キャプチャ領域オーバーレイ（半透明の枠）
# ---------------------------------------------------------
class CaptureRegionOverlay:
    """キャプチャ領域を示す半透明の枠"""
    
    def __init__(self, parent):
        self.parent = parent
        self.window = None
        self.canvas = None
        self.is_visible = False
        self.border_color = "#00FF00"  # 緑色の枠
        self.border_width = 2
        self.opacity = 0.3
        
    def create(self):
        if self.window:
            return
        
        self.window = tk.Toplevel(self.parent)
        self.window.title("Capture Region")
        self.window.overrideredirect(True)
        self.window.attributes('-topmost', True)
        self.window.attributes('-alpha', self.opacity)
        
        # 透明な背景にするため、背景色を特殊な色に
        self.window.configure(bg='gray')
        
        # Canvasで枠線を描画
        self.canvas = tk.Canvas(
            self.window,
            highlightthickness=0,
            bg='gray'
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.window.withdraw()
        
        # クリックスルー設定
        self.window.update()
        try:
            hwnd = int(self.window.wm_frame(), 16)
            self._set_click_through(hwnd)
        except Exception as e:
            print(f"[CaptureOverlay] Setup error: {e}")
    
    def _set_click_through(self, hwnd):
        if not HAS_WIN32:
            return
        try:
            GWL_EXSTYLE = -20
            WS_EX_LAYERED = 0x00080000
            WS_EX_TRANSPARENT = 0x00000020
            WS_EX_NOACTIVATE = 0x08000000
            style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            user32.SetWindowLongW(hwnd, GWL_EXSTYLE, 
                                  style | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_NOACTIVATE)
        except Exception:
            pass

    def update(self, region):
        """領域を更新"""
        if not region:
            self.hide()
            return
            
        if not self.window:
            self.create()
        
        x, y, w, h = region
        
        # ウィンドウ位置とサイズを更新
        self.window.geometry(f"{w}x{h}+{x}+{y}")
        
        # 枠線を再描画
        self.canvas.delete("all")
        
        # 外枠を描画（内側は透明にしたいので枠線だけ）
        b = self.border_width
        # 上
        self.canvas.create_rectangle(0, 0, w, b, fill=self.border_color, outline="")
        # 下
        self.canvas.create_rectangle(0, h-b, w, h, fill=self.border_color, outline="")
        # 左
        self.canvas.create_rectangle(0, 0, b, h, fill=self.border_color, outline="")
        # 右
        self.canvas.create_rectangle(w-b, 0, w, h, fill=self.border_color, outline="")
        
        self.show()
    
    def show(self):
        if self.window and not self.is_visible:
            self.window.deiconify()
            self.is_visible = True
    
    def hide(self):
        if self.window and self.is_visible:
            self.window.withdraw()
            self.is_visible = False
    
    def destroy(self):
        if self.window:
            self.window.destroy()
            self.window = None
            self.is_visible = False


# ---------------------------------------------------------
# メインアプリ
# ---------------------------------------------------------
class VLMTranslator:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("RGT - Realtime GGUF Translate v8.1")
        self.root.geometry("580x850")
        self.root.attributes('-topmost', False)
        
        # 設定を読み込み
        self.config = AppConfig.load()
        
        # 状態
        self.is_running = False
        self.mouse_pos = (0, 0)
        self.vlm = VLMClient()
        self.cache = ImageCache()
        self.context = TranslationContext(max_entries=self.config.context_memory_size)
        self.term_dict = TermDictionary()
        self.current_capture_region = None

        # ユーザーの用語辞書をロード
        if self.config.term_dictionary:
            self.term_dict.terms = dict(self.config.term_dictionary)
            print(f"[Term] Loaded {len(self.config.term_dictionary)} terms")

        # ユーザーのOCR補正辞書をVLMClientにロード
        if self.config.ocr_corrections:
            self.vlm.ocr_corrections.update(self.config.ocr_corrections)
            print(f"[OCR] Loaded {len(self.config.ocr_corrections)} user corrections")
        
        # オーバーレイ
        self.overlay = None
        self.capture_region_overlay = None  # キャプチャ領域表示用
        
        # キャプチャモード
        self.target_hwnd = None
        self.fixed_region = None
        self.window_list = []

        # テキスト比較用
        self.last_original_text = ""
        self.last_translated_text = ""
        self.last_displayed_text = ""
        self.same_text_count = 0
        self.stable_threshold = 2

        # tk変数（設定から初期化）
        self.init_tk_vars()
        
        self.setup_ui()
        self.setup_inputs()
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(200, self.init_overlay)
    
    def init_tk_vars(self):
        """tk変数を設定から初期化"""
        # モデル設定
        self.model_path = tk.StringVar(value=self.config.model_path)
        self.mmproj_path = tk.StringVar(value=self.config.mmproj_path)
        self.n_gpu_layers = tk.IntVar(value=self.config.n_gpu_layers)
        self.n_ctx = tk.IntVar(value=self.config.n_ctx)
        
        # 画像リサイズ設定
        img_size = self.config.image_max_size
        self.image_max_size = tk.StringVar(value="なし" if img_size == 0 else str(img_size))
        
        # キャプチャ設定
        self.capture_mode = tk.StringVar(value=self.config.capture_mode)
        self.capture_width = tk.IntVar(value=self.config.capture_width)
        self.capture_height = tk.IntVar(value=self.config.capture_height)
        self.interval = tk.DoubleVar(value=self.config.interval)
        self.show_capture_region = tk.BooleanVar(value=True)  # キャプチャ領域表示

        # コンテキスト記憶
        self.context_memory_size_var = tk.IntVar(value=self.config.context_memory_size)

        # オーバーレイ
        self.show_overlay = tk.BooleanVar(value=self.config.show_overlay)
        
        # オーバーレイ詳細設定
        ov = self.config.overlay
        self.ov_font_size = tk.IntVar(value=ov.font_size)
        self.ov_font_family = tk.StringVar(value=ov.font_family)
        self.ov_text_color = tk.StringVar(value=ov.text_color)
        self.ov_bg_color = tk.StringVar(value=ov.bg_color)
        self.ov_opacity = tk.DoubleVar(value=ov.opacity)
        self.ov_position = tk.StringVar(value=ov.position)
        self.ov_custom_x = tk.IntVar(value=ov.custom_x)
        self.ov_custom_y = tk.IntVar(value=ov.custom_y)
        self.ov_max_width = tk.IntVar(value=ov.max_width)
        self.ov_show_original = tk.BooleanVar(value=ov.show_original)
        self.ov_auto_hide = tk.BooleanVar(value=ov.auto_hide)
    
    def save_config(self):
        """現在の設定を保存"""
        # モデル設定
        self.config.model_path = self.model_path.get()
        self.config.mmproj_path = self.mmproj_path.get()
        self.config.n_gpu_layers = self.n_gpu_layers.get()
        self.config.n_ctx = self.n_ctx.get()
        self.config.handler = "auto"  # 自動検出
        
        # 画像リサイズ設定
        img_size_str = self.image_max_size.get()
        self.config.image_max_size = 0 if img_size_str == "なし" else int(img_size_str)
        
        # キャプチャ設定
        self.config.capture_mode = self.capture_mode.get()
        self.config.capture_width = self.capture_width.get()
        self.config.capture_height = self.capture_height.get()
        self.config.interval = self.interval.get()
        
        # オーバーレイ
        self.config.show_overlay = self.show_overlay.get()
        
        # オーバーレイ詳細
        self.config.overlay.font_size = self.ov_font_size.get()
        self.config.overlay.font_family = self.ov_font_family.get()
        self.config.overlay.text_color = self.ov_text_color.get()
        self.config.overlay.bg_color = self.ov_bg_color.get()
        self.config.overlay.opacity = self.ov_opacity.get()
        self.config.overlay.position = self.ov_position.get()
        self.config.overlay.custom_x = self.ov_custom_x.get()
        self.config.overlay.custom_y = self.ov_custom_y.get()
        self.config.overlay.max_width = self.ov_max_width.get()
        self.config.overlay.show_original = self.ov_show_original.get()
        self.config.overlay.auto_hide = self.ov_auto_hide.get()

        # 用語辞書
        self.config.term_dictionary = dict(self.term_dict.terms)

        # コンテキスト記憶数
        self.config.context_memory_size = self.context_memory_size_var.get()
        self.context.max_entries = self.config.context_memory_size

        self.config.save()
    
    def setup_ui(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # === メインタブ ===
        main_tab = ttk.Frame(notebook, padding="10")
        notebook.add(main_tab, text="メイン")
        
        # モデル設定
        model_frame = ttk.LabelFrame(main_tab, text="モデル設定 (GGUF)", padding="5")
        model_frame.pack(fill=tk.X, pady=5)
        
        # モデルパス
        path_row1 = ttk.Frame(model_frame)
        path_row1.pack(fill=tk.X, pady=2)
        ttk.Label(path_row1, text="Model:", width=8).pack(side=tk.LEFT)
        ttk.Entry(path_row1, textvariable=self.model_path, width=35).pack(side=tk.LEFT, padx=5)
        ttk.Button(path_row1, text="参照", command=self.browse_model).pack(side=tk.LEFT)
        
        # mmprojパス
        path_row2 = ttk.Frame(model_frame)
        path_row2.pack(fill=tk.X, pady=2)
        ttk.Label(path_row2, text="MMProj:", width=8).pack(side=tk.LEFT)
        ttk.Entry(path_row2, textvariable=self.mmproj_path, width=35).pack(side=tk.LEFT, padx=5)
        ttk.Button(path_row2, text="参照", command=self.browse_mmproj).pack(side=tk.LEFT)
        
        # モデル説明
        ttk.Label(model_frame, text="※ Qwen3.5 / Qwen3-VL / Gemma-4(E2B/E4B)・Gemma-3 対応",
                  foreground="gray").pack(anchor=tk.W, pady=(2, 5))
        
        # GPU設定
        gpu_row = ttk.Frame(model_frame)
        gpu_row.pack(fill=tk.X, pady=2)
        ttk.Label(gpu_row, text="GPU Layers:").pack(side=tk.LEFT)
        ttk.Entry(gpu_row, textvariable=self.n_gpu_layers, width=5).pack(side=tk.LEFT, padx=5)
        ttk.Label(gpu_row, text="(-1=全部)").pack(side=tk.LEFT)
        ttk.Label(gpu_row, text="Context:").pack(side=tk.LEFT, padx=(20, 0))
        ttk.Entry(gpu_row, textvariable=self.n_ctx, width=6).pack(side=tk.LEFT, padx=5)
        
        # 画像リサイズ設定
        img_row = ttk.Frame(model_frame)
        img_row.pack(fill=tk.X, pady=2)
        ttk.Label(img_row, text="画像リサイズ:").pack(side=tk.LEFT)
        self.image_max_size_combo = ttk.Combobox(img_row, textvariable=self.image_max_size, width=8,
                     values=["なし", "512", "768", "1024", "1280"])
        self.image_max_size_combo.pack(side=tk.LEFT, padx=5)
        ttk.Label(img_row, text="(小さいほど高速、精度低下)", foreground="gray").pack(side=tk.LEFT)
        
        # ロードボタン
        btn_row = ttk.Frame(model_frame)
        btn_row.pack(fill=tk.X, pady=(5, 0))
        self.load_btn = ttk.Button(btn_row, text="モデルをロード", command=self.load_model)
        self.load_btn.pack(side=tk.LEFT)
        ttk.Button(btn_row, text="設定を保存", command=self.save_config).pack(side=tk.LEFT, padx=10)
        
        self.status_label = ttk.Label(model_frame, text="未ロード", foreground="gray")
        self.status_label.pack(anchor=tk.W, pady=(5, 0))
        
        # キャプチャモード
        capture_frame = ttk.LabelFrame(main_tab, text="キャプチャモード", padding="5")
        capture_frame.pack(fill=tk.X, pady=5)
        
        mode_row = ttk.Frame(capture_frame)
        mode_row.pack(fill=tk.X)
        ttk.Radiobutton(mode_row, text="カーソル", variable=self.capture_mode, 
                        value="cursor", command=self.on_mode_change).pack(side=tk.LEFT)
        ttk.Radiobutton(mode_row, text="ウィンドウ", variable=self.capture_mode,
                        value="window", command=self.on_mode_change).pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(mode_row, text="ゲーム", variable=self.capture_mode,
                        value="game", command=self.on_mode_change).pack(side=tk.LEFT)
        ttk.Radiobutton(mode_row, text="範囲指定", variable=self.capture_mode,
                        value="region", command=self.on_mode_change).pack(side=tk.LEFT, padx=10)
        
        # モード説明ラベル
        self.mode_desc_label = ttk.Label(capture_frame, text="", foreground="gray")
        self.mode_desc_label.pack(anchor=tk.W, pady=(2, 0))
        
        self.cursor_frame = ttk.Frame(capture_frame)
        self.cursor_frame.pack(fill=tk.X, pady=(5, 0))
        ttk.Label(self.cursor_frame, text="サイズ:").pack(side=tk.LEFT)
        ttk.Entry(self.cursor_frame, textvariable=self.capture_width, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Label(self.cursor_frame, text="x").pack(side=tk.LEFT)
        ttk.Entry(self.cursor_frame, textvariable=self.capture_height, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(self.cursor_frame, text="枠を表示", 
                        variable=self.show_capture_region).pack(side=tk.LEFT, padx=10)
        
        self.window_frame = ttk.Frame(capture_frame)
        ttk.Label(self.window_frame, text="対象:").pack(side=tk.LEFT)
        self.window_combo = ttk.Combobox(self.window_frame, width=35)
        self.window_combo.pack(side=tk.LEFT, padx=5)
        self.window_combo.bind("<<ComboboxSelected>>", self.on_window_selected)
        ttk.Button(self.window_frame, text="更新", command=self.refresh_windows).pack(side=tk.LEFT)
        ttk.Button(self.window_frame, text="全プロセス", command=self.refresh_all_processes).pack(side=tk.LEFT, padx=2)
        
        # ゲームモード用フレーム（ウィンドウと同じ選択UI）
        self.game_frame = ttk.Frame(capture_frame)
        ttk.Label(self.game_frame, text="対象:").pack(side=tk.LEFT)
        self.game_window_combo = ttk.Combobox(self.game_frame, width=35)
        self.game_window_combo.pack(side=tk.LEFT, padx=5)
        self.game_window_combo.bind("<<ComboboxSelected>>", self.on_game_window_selected)
        ttk.Button(self.game_frame, text="更新", command=self.refresh_windows).pack(side=tk.LEFT)
        ttk.Button(self.game_frame, text="全プロセス", command=self.refresh_all_processes).pack(side=tk.LEFT, padx=2)
        
        self.region_frame = ttk.Frame(capture_frame)
        ttk.Button(self.region_frame, text="範囲を選択", command=self.select_region).pack(side=tk.LEFT)
        self.region_label = ttk.Label(self.region_frame, text="未設定")
        self.region_label.pack(side=tk.LEFT, padx=10)
        
        # 初期表示
        self.on_mode_change()
        
        # 間隔設定
        interval_row = ttk.Frame(main_tab)
        interval_row.pack(fill=tk.X, pady=5)
        ttk.Label(interval_row, text="更新間隔(秒):").pack(side=tk.LEFT)
        ttk.Combobox(interval_row, textvariable=self.interval, width=5,
                     values=[0.3, 0.5, 1.0, 2.0, 3.0]).pack(side=tk.LEFT, padx=5)

        # コンテキスト記憶設定
        context_row = ttk.Frame(main_tab)
        context_row.pack(fill=tk.X, pady=2)
        ttk.Label(context_row, text="コンテキスト記憶数:").pack(side=tk.LEFT)
        ttk.Spinbox(context_row, textvariable=self.context_memory_size_var,
                     width=5, from_=0, to=20).pack(side=tk.LEFT, padx=5)
        ttk.Label(context_row, text="(0=無効, 翻訳の一貫性向上)",
                  foreground="gray").pack(side=tk.LEFT)

        # 開始/停止
        ctrl_row = ttk.Frame(main_tab)
        ctrl_row.pack(fill=tk.X, pady=10)
        self.start_btn = ttk.Button(ctrl_row, text="開始 (F9)", command=self.toggle, state=tk.DISABLED)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        ttk.Button(ctrl_row, text="クリア", command=self.clear_display).pack(side=tk.LEFT)
        
        # 結果表示はオーバーレイのみ使用
        
        # === オーバーレイ設定タブ ===
        overlay_tab = ttk.Frame(notebook, padding="10")
        notebook.add(overlay_tab, text="オーバーレイ")
        
        # オーバーレイ内のサブノートブック（共通/詳細）
        overlay_notebook = ttk.Notebook(overlay_tab)
        overlay_notebook.pack(fill=tk.BOTH, expand=True)
        
        # === 共通設定タブ ===
        common_tab = ttk.Frame(overlay_notebook, padding="5")
        overlay_notebook.add(common_tab, text="共通設定")
        
        # 表示設定
        display_frame = ttk.LabelFrame(common_tab, text="表示設定", padding="5")
        display_frame.pack(fill=tk.X, pady=5)
        
        ttk.Checkbutton(display_frame, text="オーバーレイを表示", 
                        variable=self.show_overlay,
                        command=self.toggle_overlay).pack(anchor=tk.W)
        ttk.Checkbutton(display_frame, text="テキストなしで自動非表示",
                        variable=self.ov_auto_hide).pack(anchor=tk.W)
        ttk.Checkbutton(display_frame, text="原文も表示",
                        variable=self.ov_show_original).pack(anchor=tk.W)
        
        # フォント設定
        font_frame = ttk.LabelFrame(common_tab, text="フォント（全タイプ共通）", padding="5")
        font_frame.pack(fill=tk.X, pady=5)
        
        font_row1 = ttk.Frame(font_frame)
        font_row1.pack(fill=tk.X, pady=2)
        ttk.Label(font_row1, text="フォント:").pack(side=tk.LEFT)
        ttk.Combobox(font_row1, textvariable=self.ov_font_family, width=15,
                     values=["Meiryo UI", "Yu Gothic UI", "MS Gothic", "Arial", "Consolas"]).pack(side=tk.LEFT, padx=5)
        ttk.Label(font_row1, text="サイズ:").pack(side=tk.LEFT, padx=(10, 0))
        ttk.Spinbox(font_row1, textvariable=self.ov_font_size, width=5, from_=8, to=48).pack(side=tk.LEFT, padx=5)
        
        # 色設定
        color_frame = ttk.LabelFrame(common_tab, text="色（全タイプ共通）", padding="5")
        color_frame.pack(fill=tk.X, pady=5)
        
        color_row = ttk.Frame(color_frame)
        color_row.pack(fill=tk.X, pady=2)
        
        ttk.Label(color_row, text="文字色:").pack(side=tk.LEFT)
        self.text_color_btn = tk.Button(color_row, width=3, bg=self.ov_text_color.get(),
                                         command=self.pick_text_color)
        self.text_color_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(color_row, text="背景色:").pack(side=tk.LEFT, padx=(10, 0))
        self.bg_color_btn = tk.Button(color_row, width=3, bg=self.ov_bg_color.get(),
                                       command=self.pick_bg_color)
        self.bg_color_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(color_row, text="透明度:").pack(side=tk.LEFT, padx=(10, 0))
        ttk.Scale(color_row, variable=self.ov_opacity, from_=0.1, to=1.0, 
                  orient=tk.HORIZONTAL, length=100).pack(side=tk.LEFT, padx=5)
        
        # 位置設定
        pos_frame = ttk.LabelFrame(common_tab, text="位置", padding="5")
        pos_frame.pack(fill=tk.X, pady=5)
        
        pos_row1 = ttk.Frame(pos_frame)
        pos_row1.pack(fill=tk.X, pady=2)
        ttk.Radiobutton(pos_row1, text="キャプチャ領域の下", variable=self.ov_position, 
                        value="bottom").pack(side=tk.LEFT)
        ttk.Radiobutton(pos_row1, text="キャプチャ領域の上", variable=self.ov_position,
                        value="top").pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(pos_row1, text="固定位置", variable=self.ov_position,
                        value="custom").pack(side=tk.LEFT)
        
        pos_row2 = ttk.Frame(pos_frame)
        pos_row2.pack(fill=tk.X, pady=2)
        ttk.Label(pos_row2, text="固定位置 X:").pack(side=tk.LEFT)
        ttk.Entry(pos_row2, textvariable=self.ov_custom_x, width=5).pack(side=tk.LEFT, padx=5)
        ttk.Label(pos_row2, text="Y:").pack(side=tk.LEFT)
        ttk.Entry(pos_row2, textvariable=self.ov_custom_y, width=5).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(pos_row2, text="最大幅:").pack(side=tk.LEFT, padx=(20, 0))
        ttk.Entry(pos_row2, textvariable=self.ov_max_width, width=5).pack(side=tk.LEFT, padx=5)
        
        # === 詳細設定タブ（タイプ別） ===
        detail_tab = ttk.Frame(overlay_notebook, padding="5")
        overlay_notebook.add(detail_tab, text="詳細設定（タイプ別）")
        
        # スクロール可能なフレーム
        detail_canvas = tk.Canvas(detail_tab, highlightthickness=0)
        detail_scrollbar = ttk.Scrollbar(detail_tab, orient="vertical", command=detail_canvas.yview)
        detail_scrollable = ttk.Frame(detail_canvas)
        
        detail_scrollable.bind(
            "<Configure>",
            lambda e: detail_canvas.configure(scrollregion=detail_canvas.bbox("all"))
        )
        
        detail_canvas.create_window((0, 0), window=detail_scrollable, anchor="nw")
        detail_canvas.configure(yscrollcommand=detail_scrollbar.set)
        
        detail_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        detail_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # タイプ別設定用の変数を初期化
        self.type_vars = {}
        type_labels = {
            "dialogue": ("セリフ", "キャラクターの台詞"),
            "choice": ("選択肢", "プレイヤーの選択オプション"),
            "narration": ("ナレーション", "状況説明・地の文")
        }
        
        for text_type, (label, desc) in type_labels.items():
            type_cfg = self.config.overlay.get_type_config(text_type)
            
            # タイプ別のフレーム
            type_frame = ttk.LabelFrame(detail_scrollable, text=f"{label} ({text_type})", padding="5")
            type_frame.pack(fill=tk.X, pady=5, padx=5)
            
            ttk.Label(type_frame, text=desc, foreground="gray").pack(anchor=tk.W)
            
            # 変数を作成
            vars_dict = {
                "enabled": tk.BooleanVar(value=type_cfg.enabled),
                "use_custom": tk.BooleanVar(value=type_cfg.use_custom),
                "font_size": tk.IntVar(value=type_cfg.font_size),
                "font_family": tk.StringVar(value=type_cfg.font_family),
                "text_color": tk.StringVar(value=type_cfg.text_color),
                "prefix": tk.StringVar(value=type_cfg.prefix),
                "suffix": tk.StringVar(value=type_cfg.suffix)
            }
            self.type_vars[text_type] = vars_dict
            
            # 基本設定行
            row1 = ttk.Frame(type_frame)
            row1.pack(fill=tk.X, pady=2)
            ttk.Checkbutton(row1, text="表示する", variable=vars_dict["enabled"]).pack(side=tk.LEFT)
            ttk.Checkbutton(row1, text="カスタム設定を使用", 
                           variable=vars_dict["use_custom"]).pack(side=tk.LEFT, padx=20)
            
            # プレフィックス・サフィックス行
            row2 = ttk.Frame(type_frame)
            row2.pack(fill=tk.X, pady=2)
            ttk.Label(row2, text="先頭記号:").pack(side=tk.LEFT)
            prefix_entry = ttk.Entry(row2, textvariable=vars_dict["prefix"], width=8)
            prefix_entry.pack(side=tk.LEFT, padx=5)
            ttk.Label(row2, text="末尾記号:").pack(side=tk.LEFT, padx=(10, 0))
            ttk.Entry(row2, textvariable=vars_dict["suffix"], width=8).pack(side=tk.LEFT, padx=5)
            
            # プリセットボタン（選択肢用）
            if text_type == "choice":
                preset_frame = ttk.Frame(row2)
                preset_frame.pack(side=tk.LEFT, padx=10)
                ttk.Button(preset_frame, text="→", width=3,
                          command=lambda v=vars_dict: v["prefix"].set("→ ")).pack(side=tk.LEFT, padx=2)
                ttk.Button(preset_frame, text="・", width=3,
                          command=lambda v=vars_dict: v["prefix"].set("・")).pack(side=tk.LEFT, padx=2)
                ttk.Button(preset_frame, text="▶", width=3,
                          command=lambda v=vars_dict: v["prefix"].set("▶ ")).pack(side=tk.LEFT, padx=2)
                ttk.Button(preset_frame, text="[ ]", width=3,
                          command=lambda v=vars_dict: (v["prefix"].set("[ "), v["suffix"].set(" ]"))).pack(side=tk.LEFT, padx=2)
            
            # カスタムフォント・色設定行
            row3 = ttk.Frame(type_frame)
            row3.pack(fill=tk.X, pady=2)
            ttk.Label(row3, text="文字色:").pack(side=tk.LEFT)
            
            # 色ボタンを作成
            color_btn = tk.Button(row3, width=3, bg=vars_dict["text_color"].get())
            color_btn.pack(side=tk.LEFT, padx=5)
            
            # 色選択コールバック
            def make_color_callback(var, btn):
                def callback():
                    color = colorchooser.askcolor(color=var.get(), title="文字色を選択")
                    if color[1]:
                        var.set(color[1])
                        btn.configure(bg=color[1])
                return callback
            
            color_btn.configure(command=make_color_callback(vars_dict["text_color"], color_btn))
            
            ttk.Label(row3, text="サイズ:").pack(side=tk.LEFT, padx=(10, 0))
            ttk.Spinbox(row3, textvariable=vars_dict["font_size"], width=5, from_=8, to=48).pack(side=tk.LEFT, padx=5)
        
        # 適用ボタン
        apply_row = ttk.Frame(overlay_tab)
        apply_row.pack(fill=tk.X, pady=10)
        ttk.Button(apply_row, text="オーバーレイに適用", command=self.apply_overlay_config).pack(side=tk.LEFT)
        ttk.Button(apply_row, text="プレビュー", command=self.preview_overlay).pack(side=tk.LEFT, padx=10)
        ttk.Button(apply_row, text="設定を保存", command=self.save_config).pack(side=tk.LEFT)
        
        # === OCR補正タブ ===
        ocr_tab = ttk.Frame(notebook, padding="10")
        notebook.add(ocr_tab, text="OCR補正")
        
        # 説明
        ttk.Label(ocr_tab, text="フォントの癖による誤認識を補正します", 
                  foreground="gray").pack(anchor=tk.W)
        ttk.Label(ocr_tab, text="例: coppins → coffins (FFがPPに見える場合)", 
                  foreground="gray").pack(anchor=tk.W, pady=(0, 10))
        
        # 補正辞書リスト
        list_frame = ttk.LabelFrame(ocr_tab, text="補正辞書", padding="5")
        list_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # リストボックス + スクロールバー
        list_container = ttk.Frame(list_frame)
        list_container.pack(fill=tk.BOTH, expand=True)
        
        self.ocr_listbox = tk.Listbox(list_container, height=10, font=("Consolas", 10))
        ocr_scrollbar = ttk.Scrollbar(list_container, orient="vertical", command=self.ocr_listbox.yview)
        self.ocr_listbox.configure(yscrollcommand=ocr_scrollbar.set)
        
        self.ocr_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ocr_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 既存の補正をリストに追加
        self.refresh_ocr_list()
        
        # 追加フォーム
        add_frame = ttk.LabelFrame(ocr_tab, text="補正を追加", padding="5")
        add_frame.pack(fill=tk.X, pady=5)
        
        add_row = ttk.Frame(add_frame)
        add_row.pack(fill=tk.X, pady=2)
        
        ttk.Label(add_row, text="誤認識:").pack(side=tk.LEFT)
        self.ocr_wrong_var = tk.StringVar()
        ttk.Entry(add_row, textvariable=self.ocr_wrong_var, width=15).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(add_row, text="→").pack(side=tk.LEFT, padx=5)
        
        ttk.Label(add_row, text="正しい:").pack(side=tk.LEFT)
        self.ocr_correct_var = tk.StringVar()
        ttk.Entry(add_row, textvariable=self.ocr_correct_var, width=15).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(add_row, text="追加", command=self.add_ocr_correction).pack(side=tk.LEFT, padx=10)
        
        # 削除ボタン
        btn_row = ttk.Frame(ocr_tab)
        btn_row.pack(fill=tk.X, pady=5)
        ttk.Button(btn_row, text="選択を削除", command=self.remove_ocr_correction).pack(side=tk.LEFT)
        ttk.Button(btn_row, text="設定を保存", command=self.save_config).pack(side=tk.LEFT, padx=10)

        # === 用語辞書タブ ===
        term_tab = ttk.Frame(notebook, padding="10")
        notebook.add(term_tab, text="用語辞書")

        ttk.Label(term_tab, text="ゲーム固有の用語を登録して翻訳の一貫性を保ちます",
                  foreground="gray").pack(anchor=tk.W)
        ttk.Label(term_tab, text="例: Dragonborn → ドラゴンボーン",
                  foreground="gray").pack(anchor=tk.W, pady=(0, 10))

        # 用語リスト
        term_list_frame = ttk.LabelFrame(term_tab, text="用語辞書", padding="5")
        term_list_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        term_container = ttk.Frame(term_list_frame)
        term_container.pack(fill=tk.BOTH, expand=True)

        self.term_listbox = tk.Listbox(term_container, height=10, font=("Consolas", 10))
        term_scrollbar = ttk.Scrollbar(term_container, orient="vertical", command=self.term_listbox.yview)
        self.term_listbox.configure(yscrollcommand=term_scrollbar.set)
        self.term_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        term_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.refresh_term_list()

        # 追加フォーム
        term_add_frame = ttk.LabelFrame(term_tab, text="用語を追加", padding="5")
        term_add_frame.pack(fill=tk.X, pady=5)

        term_add_row = ttk.Frame(term_add_frame)
        term_add_row.pack(fill=tk.X, pady=2)

        ttk.Label(term_add_row, text="英語:").pack(side=tk.LEFT)
        self.term_en_var = tk.StringVar()
        ttk.Entry(term_add_row, textvariable=self.term_en_var, width=15).pack(side=tk.LEFT, padx=5)
        ttk.Label(term_add_row, text="→").pack(side=tk.LEFT, padx=5)
        ttk.Label(term_add_row, text="日本語:").pack(side=tk.LEFT)
        self.term_ja_var = tk.StringVar()
        ttk.Entry(term_add_row, textvariable=self.term_ja_var, width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(term_add_row, text="追加", command=self.add_term).pack(side=tk.LEFT, padx=10)

        # 削除 + 保存
        term_btn_row = ttk.Frame(term_tab)
        term_btn_row.pack(fill=tk.X, pady=5)
        ttk.Button(term_btn_row, text="選択を削除", command=self.remove_term).pack(side=tk.LEFT)
        ttk.Button(term_btn_row, text="設定を保存", command=self.save_config).pack(side=tk.LEFT, padx=10)

        # ステータスバー
        self.status_var = tk.StringVar(value="モデルをロードしてください")
        ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN).pack(fill=tk.X, side=tk.BOTTOM)
    
    def pick_text_color(self):
        """文字色を選択"""
        color = colorchooser.askcolor(color=self.ov_text_color.get(), title="文字色を選択")
        if color[1]:
            self.ov_text_color.set(color[1])
            self.text_color_btn.configure(bg=color[1])
    
    def pick_bg_color(self):
        """背景色を選択"""
        color = colorchooser.askcolor(color=self.ov_bg_color.get(), title="背景色を選択")
        if color[1]:
            self.ov_bg_color.set(color[1])
            self.bg_color_btn.configure(bg=color[1])
    
    def apply_overlay_config(self):
        """オーバーレイ設定を適用"""
        # タイプ別設定を収集
        type_configs = {}
        if hasattr(self, 'type_vars'):
            for text_type, vars_dict in self.type_vars.items():
                type_configs[text_type] = {
                    "enabled": vars_dict["enabled"].get(),
                    "use_custom": vars_dict["use_custom"].get(),
                    "font_size": vars_dict["font_size"].get(),
                    "font_family": vars_dict["font_family"].get(),
                    "text_color": vars_dict["text_color"].get(),
                    "prefix": vars_dict["prefix"].get(),
                    "suffix": vars_dict["suffix"].get()
                }
        else:
            # デフォルト値
            type_configs = self.config.overlay.type_configs
        
        if self.overlay:
            config = OverlayConfig(
                font_size=self.ov_font_size.get(),
                font_family=self.ov_font_family.get(),
                text_color=self.ov_text_color.get(),
                bg_color=self.ov_bg_color.get(),
                opacity=self.ov_opacity.get(),
                position=self.ov_position.get(),
                custom_x=self.ov_custom_x.get(),
                custom_y=self.ov_custom_y.get(),
                max_width=self.ov_max_width.get(),
                show_original=self.ov_show_original.get(),
                auto_hide=self.ov_auto_hide.get(),
                type_configs=type_configs
            )
            self.overlay.apply_config(config)
            self.config.overlay = config
    
    def preview_overlay(self):
        """オーバーレイをプレビュー（全タイプのサンプル表示）"""
        self.apply_overlay_config()
        if self.overlay:
            # 各タイプのサンプルテキストで表示
            sample_blocks = [
                TextBlock(type="dialogue", original="Hello, how are you?", 
                         translated="こんにちは、お元気ですか？"),
                TextBlock(type="narration", original="The room fell silent.",
                         translated="部屋は静まり返った。"),
                TextBlock(type="choice", original="Yes, I will help.",
                         translated="はい、手伝います"),
                TextBlock(type="choice", original="No, I refuse.",
                         translated="いいえ、断ります"),
            ]
            self.overlay.update(sample_blocks, self.current_capture_region or (100, 100, 400, 200))
    
    def browse_model(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="GGUFモデルを選択",
            filetypes=[("GGUF files", "*.gguf"), ("All files", "*.*")]
        )
        if path:
            self.model_path.set(path)
    
    def browse_mmproj(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="MMProj (Vision) ファイルを選択",
            filetypes=[("GGUF files", "*.gguf"), ("All files", "*.*")]
        )
        if path:
            self.mmproj_path.set(path)
    
    def refresh_ocr_list(self):
        """OCR補正リストを更新"""
        self.ocr_listbox.delete(0, tk.END)
        # デフォルト辞書
        for wrong, correct in self.vlm.ocr_corrections.items():
            self.ocr_listbox.insert(tk.END, f"{wrong} → {correct}")
    
    def add_ocr_correction(self):
        """OCR補正を追加"""
        wrong = self.ocr_wrong_var.get().strip().lower()
        correct = self.ocr_correct_var.get().strip()
        
        if wrong and correct:
            # VLMClientに追加
            self.vlm.add_ocr_correction(wrong, correct)
            # 設定にも保存（ユーザー追加分）
            self.config.ocr_corrections[wrong] = correct
            # リスト更新
            self.refresh_ocr_list()
            # 入力クリア
            self.ocr_wrong_var.set("")
            self.ocr_correct_var.set("")
            self.status_var.set(f"補正追加: {wrong} → {correct}")
    
    def remove_ocr_correction(self):
        """選択したOCR補正を削除"""
        selection = self.ocr_listbox.curselection()
        if selection:
            item = self.ocr_listbox.get(selection[0])
            wrong = item.split(" → ")[0]
            
            # VLMClientから削除
            if wrong in self.vlm.ocr_corrections:
                del self.vlm.ocr_corrections[wrong]
            # 設定からも削除
            if wrong in self.config.ocr_corrections:
                del self.config.ocr_corrections[wrong]
            
            self.refresh_ocr_list()
            self.status_var.set(f"補正削除: {wrong}")

    def refresh_term_list(self):
        """用語辞書リストを更新"""
        self.term_listbox.delete(0, tk.END)
        for en, ja in self.term_dict.terms.items():
            self.term_listbox.insert(tk.END, f"{en} → {ja}")

    def add_term(self):
        """用語を追加"""
        en = self.term_en_var.get().strip()
        ja = self.term_ja_var.get().strip()
        if en and ja:
            self.term_dict.add(en, ja)
            self.refresh_term_list()
            self.term_en_var.set("")
            self.term_ja_var.set("")
            self.status_var.set(f"用語追加: {en} → {ja}")

    def remove_term(self):
        """選択した用語を削除"""
        selection = self.term_listbox.curselection()
        if selection:
            item = self.term_listbox.get(selection[0])
            en = item.split(" → ")[0]
            self.term_dict.remove(en)
            self.refresh_term_list()
            self.status_var.set(f"用語削除: {en}")

    def setup_inputs(self):
        def on_move(x, y):
            self.mouse_pos = (x, y)
        self.mouse_listener = mouse.Listener(on_move=on_move)
        self.mouse_listener.start()
        
        def on_press(key):
            if key == keyboard.Key.f9:
                self.root.after(0, self.toggle)
        self.key_listener = keyboard.Listener(on_press=on_press)
        self.key_listener.start()
    
    def on_mode_change(self):
        mode = self.capture_mode.get()
        self.cursor_frame.pack_forget()
        self.window_frame.pack_forget()
        self.game_frame.pack_forget()
        self.region_frame.pack_forget()
        
        # モード説明を更新
        mode_descriptions = {
            "cursor": "汎用モード: カーソル周辺のテキストを広く認識",
            "window": "汎用モード: ウィンドウ内のテキストを広く認識",
            "game": "ゲームモード: セリフ・選択肢・ナレーションを分類",
            "region": "汎用モード: 指定範囲のテキストを広く認識"
        }
        self.mode_desc_label.config(text=mode_descriptions.get(mode, ""))
        
        # プロンプトを切り替え
        is_game_mode = (mode == "game")
        self.vlm.set_game_mode(is_game_mode)
        
        if mode == "cursor":
            self.cursor_frame.pack(fill=tk.X, pady=(5, 0))
        elif mode == "window":
            self.window_frame.pack(fill=tk.X, pady=(5, 0))
            self.refresh_windows()
        elif mode == "game":
            self.game_frame.pack(fill=tk.X, pady=(5, 0))
            self.refresh_windows()
        elif mode == "region":
            self.region_frame.pack(fill=tk.X, pady=(5, 0))
    
    def refresh_windows(self):
        """通常のウィンドウ一覧を更新"""
        self.window_list = get_window_list()
        # (hwnd, title, process_name, is_visible) -> 表示用に変換
        titles = []
        for item in self.window_list:
            if len(item) >= 4:
                hwnd, title, pname, visible = item[:4]
                display = f"{title[:40]}" + (f" [{pname}]" if pname else "")
            else:
                hwnd, title = item[:2]
                display = title[:50]
            titles.append(display)
        
        self.window_combo['values'] = titles
        self.game_window_combo['values'] = titles
        if titles:
            self.window_combo.current(0)
            self.game_window_combo.current(0)
            self.on_window_selected(None)
    
    def refresh_all_processes(self):
        """全プロセスからウィンドウを取得（隠れウィンドウも含む）"""
        self.status_var.set("全プロセスを検索中...")
        self.root.update()
        
        processes = get_all_processes_with_windows()
        
        if processes:
            # (hwnd, title, process_name, pid)
            self.window_list = [(hwnd, title, pname, True) for hwnd, title, pname, pid in processes]
            
            titles = []
            for hwnd, title, pname, pid in processes:
                display = f"{title[:35]} [{pname}]"
                titles.append(display)
            
            self.window_combo['values'] = titles
            self.game_window_combo['values'] = titles
            if titles:
                self.window_combo.current(0)
                self.game_window_combo.current(0)
                self.on_window_selected(None)
            
            self.status_var.set(f"{len(processes)} 個のウィンドウを検出")
        else:
            self.status_var.set("プロセスが見つかりませんでした")
    
    def on_window_selected(self, event):
        idx = self.window_combo.current()
        if 0 <= idx < len(self.window_list):
            self.target_hwnd = self.window_list[idx][0]
    
    def on_game_window_selected(self, event):
        """ゲームモード用ウィンドウ選択"""
        idx = self.game_window_combo.current()
        if 0 <= idx < len(self.window_list):
            self.target_hwnd = self.window_list[idx][0]
    
    def select_region(self):
        self.root.withdraw()
        time.sleep(0.2)
        
        overlay = tk.Toplevel()
        overlay.attributes('-fullscreen', True)
        overlay.attributes('-topmost', True)
        overlay.attributes('-alpha', 0.3)
        overlay.configure(cursor="crosshair", bg="black")
        
        canvas = tk.Canvas(overlay, bg="black", highlightthickness=0)
        canvas.pack(fill=tk.BOTH, expand=True)
        
        start = [0, 0]
        rect_id = [None]
        
        def on_press(e):
            start[0], start[1] = e.x, e.y
            rect_id[0] = canvas.create_rectangle(e.x, e.y, e.x, e.y, outline="red", width=2)
        
        def on_drag(e):
            if rect_id[0]:
                canvas.coords(rect_id[0], start[0], start[1], e.x, e.y)
        
        def on_release(e):
            x1, y1 = min(start[0], e.x), min(start[1], e.y)
            w, h = abs(e.x - start[0]), abs(e.y - start[1])
            if w > 10 and h > 10:
                self.fixed_region = (x1, y1, w, h)
                self.region_label.config(text=f"{w}x{h} @ ({x1},{y1})")
            overlay.destroy()
            self.root.deiconify()
        
        def on_escape(e):
            overlay.destroy()
            self.root.deiconify()
        
        overlay.bind("<Button-1>", on_press)
        overlay.bind("<B1-Motion>", on_drag)
        overlay.bind("<ButtonRelease-1>", on_release)
        overlay.bind("<Escape>", on_escape)
    
    def load_model(self):
        """モデルをロード"""
        model_path = self.model_path.get()
        mmproj_path = self.mmproj_path.get()
        
        if not model_path or not mmproj_path:
            self.status_label.config(text="モデルとMMProjのパスを指定してください", foreground="red")
            return
        
        self.load_btn.config(state=tk.DISABLED)
        self.status_label.config(text="ロード中...", foreground="orange")
        self.status_var.set("モデルをロード中...")
        self.root.update()
        
        def do_load():
            ok, msg = self.vlm.load_model(
                model_path=model_path,
                mmproj_path=mmproj_path,
                n_gpu_layers=self.n_gpu_layers.get(),
                n_ctx=self.n_ctx.get(),
                handler_name="auto"
            )
            self.root.after(0, lambda: self.on_model_loaded(ok, msg))
        
        threading.Thread(target=do_load, daemon=True).start()
    
    def on_model_loaded(self, ok: bool, msg: str):
        self.load_btn.config(state=tk.NORMAL)
        if ok:
            # 画像リサイズ設定を反映
            img_size_str = self.image_max_size.get()
            img_size = 0 if img_size_str == "なし" else int(img_size_str)
            self.vlm.set_image_max_size(img_size)
            
            self.status_label.config(text=f"✓ {msg}", foreground="green")
            self.start_btn.config(state=tk.NORMAL)
            self.status_var.set("準備完了")
        else:
            self.status_label.config(text=f"✗ {msg}", foreground="red")
            self.start_btn.config(state=tk.DISABLED)
            self.status_var.set("モデルロード失敗")
    
    def toggle(self):
        if self.is_running:
            self.stop()
        else:
            self.start()
    
    def start(self):
        if not self.vlm.initialized:
            return
        self.is_running = True
        self.start_btn.config(text="停止 (F9)")
        self.status_var.set("実行中...")
        threading.Thread(target=self.main_loop, daemon=True).start()
    
    def stop(self):
        self.is_running = False
        self.start_btn.config(text="開始 (F9)")
        self.status_var.set("停止")
        # オーバーレイを非表示
        if self.overlay:
            self.overlay.hide()
        # キャプチャ領域オーバーレイも非表示
        if self.capture_region_overlay:
            self.capture_region_overlay.hide()
    
    def clear_display(self):
        self.cache = ImageCache()
        self.context.clear()
        self.last_original_text = ""
        self.last_translated_text = ""
        if self.overlay:
            self.overlay.hide()
    
    def get_capture_region(self) -> Optional[Tuple[int, int, int, int]]:
        mode = self.capture_mode.get()
        if mode == "cursor":
            cx, cy = self.mouse_pos
            w, h = self.capture_width.get(), self.capture_height.get()
            left = max(0, cx - w // 2)
            top = max(0, cy - h // 2)
            return (left, top, w, h)
        elif mode == "window" or mode == "game":
            if not self.target_hwnd:
                return None
            return get_client_rect(self.target_hwnd) or get_window_rect(self.target_hwnd)
        elif mode == "region":
            return self.fixed_region
        return None
    
    def main_loop(self):
        with mss.mss() as sct:
            while self.is_running:
                start_time = time.time()
                try:
                    self.process_frame(sct)
                except Exception as e:
                    print(f"Error: {e}")
                    traceback.print_exc()
                elapsed = time.time() - start_time
                wait = self.interval.get() - elapsed
                if wait > 0:
                    time.sleep(wait)
    
    def _do_translate(self, texts: List[str], img_bgr=None) -> Optional[List[TextBlock]]:
        """翻訳を実行（テキストキャッシュ確認 → VLM呼び出し → キャッシュ保存）"""
        # テキスト内容ベースのキャッシュを確認
        cached_by_text = self.cache.get_by_text(texts)
        if cached_by_text:
            print(f" [TEXT CACHE HIT]")
            return cached_by_text

        # VLM翻訳実行（コンテキスト・用語辞書付き）
        blocks = self.vlm.translate_text(texts, self.context, self.term_dict)
        if blocks:
            self.cache.set_by_text(texts, blocks)
            if img_bgr is not None:
                self.cache.set_translations(img_bgr, blocks)
            self.context.add(blocks)
        return blocks

    def process_frame(self, sct):
        region = self.get_capture_region()
        if not region: return

        left, top, w, h = region
        self.current_capture_region = (left, top, w, h)

        # カーソルモードの時はキャプチャ領域を表示
        if self.capture_mode.get() == "cursor" and self.show_capture_region.get():
            self.root.after(0, lambda r=(left, top, w, h): self.update_capture_region_overlay(r))

        monitor = {"left": left, "top": top, "width": w, "height": h}
        sct_img = sct.grab(monitor)
        img = np.array(sct_img)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # === キャッシュ層1: 画像→翻訳済みブロック（VLM呼び出し0回） ===
        cached_blocks = self.cache.get_translations(img_bgr)
        if cached_blocks is not None:
            self.root.after(0, lambda b=cached_blocks: self.display_results(b))
            return

        # === キャッシュ層2: 画像→抽出テキスト（抽出VLM呼び出し節約） ===
        texts = self.cache.get_extractions(img_bgr)
        if texts is not None:
            print(f" [EXTRACT CACHE HIT]", end="")
        else:
            texts = self.vlm.extract_text(img_bgr)
            if texts is None:
                print("!", end="", flush=True)
                return
            # 抽出結果をキャッシュ
            self.cache.set_extractions(img_bgr, texts)

        if not texts:
            self.context.on_empty()
            if self.last_original_text:
                print(f"[CLEAR] No text")
                self.last_original_text = ""
                self.same_text_count = 0
                if self.overlay and self.config.overlay.auto_hide:
                    self.root.after(0, self.overlay.hide)
            return

        current_text = " ".join(texts)

        # 完全一致 → カウント増加
        if current_text == self.last_original_text:
            self.same_text_count += 1

            # GROWING状態から安定した場合 → 翻訳実行
            if self.same_text_count == self.stable_threshold:
                if not self.last_translated_text or self.last_translated_text != current_text:
                    print(f" [STABLE -> TRANSLATE]")
                    self.last_translated_text = current_text
                    blocks = self._do_translate(texts, img_bgr)
                    if blocks:
                        self.root.after(0, lambda b=blocks: self.display_results(b))
                else:
                    print(" [STABLE]")
            elif self.same_text_count < self.stable_threshold:
                print(".", end="", flush=True)
            return

        # テキストが伸びてる途中チェック
        if self.last_original_text:
            prev_prefix = self.last_original_text[:min(20, len(self.last_original_text))]
            if current_text.startswith(prev_prefix) and len(current_text) > len(self.last_original_text):
                print(f"[GROWING] +{len(current_text) - len(self.last_original_text)} chars")
                self.last_original_text = current_text
                self.last_translated_text = ""
                self.same_text_count = 0
                return

        # 完全に新しいテキスト → 即翻訳
        print(f"\n[NEW] Text: '{current_text[:80]}...'")
        self.last_original_text = current_text
        self.last_translated_text = current_text
        self.same_text_count = 0

        blocks = self._do_translate(texts, img_bgr)
        if blocks:
            self.root.after(0, lambda b=blocks: self.display_results(b))
    
    def display_results(self, blocks: List[TextBlock]):
        filtered_blocks = []
        target_types = ["dialogue", "choice", "narration", "text"]  # 汎用モードのtextも含む
        
        for block in blocks:
            if block.type in target_types:
                filtered_blocks.append(block)
            elif block.type == "other" and len(block.original) > 5:
                filtered_blocks.append(block)
        
        new_text_lines = [b.translated for b in filtered_blocks if b.translated.strip()]
        new_text = "\n".join(new_text_lines)
        self.last_displayed_text = new_text
        self.update_overlay(filtered_blocks)
    
    def init_overlay(self):
        self.overlay = TranslationOverlay(self.root, self.config.overlay)
        if self.show_overlay.get():
            self.overlay.create()
        # キャプチャ領域オーバーレイも初期化
        self.capture_region_overlay = CaptureRegionOverlay(self.root)
    
    def toggle_overlay(self):
        if self.show_overlay.get():
            if self.overlay and not self.overlay.window:
                self.overlay.create()
            if self.overlay:
                self.overlay.show()
        else:
            if self.overlay:
                self.overlay.hide()
    
    def update_capture_region_overlay(self, region):
        """キャプチャ領域オーバーレイを更新"""
        if self.capture_region_overlay:
            self.capture_region_overlay.update(region)
    
    def update_overlay(self, blocks: List[TextBlock]):
        if not self.show_overlay.get() or not self.overlay:
            return
        if not self.overlay.window:
            self.overlay.create()
        self.overlay.update(blocks, self.current_capture_region)
    
    def on_close(self):
        self.is_running = False
        # 設定を自動保存
        self.save_config()
        if self.overlay:
            self.overlay.destroy()
        if self.capture_region_overlay:
            self.capture_region_overlay.destroy()
        try:
            self.mouse_listener.stop()
        except Exception:
            pass
        try:
            self.key_listener.stop()
        except Exception:
            pass
        self.root.destroy()
        os._exit(0)


# ---------------------------------------------------------
# エントリーポイント
# ---------------------------------------------------------
if __name__ == "__main__":
    if not HAS_LLAMA_CPP:
        print("=" * 50)
        print("エラー: llama-cpp-pythonがインストールされていません")
        print("=" * 50)
        print("")
        print("JamePengフォーク版が必要です（VLM対応）:")
        print("  https://github.com/JamePeng/llama-cpp-python/releases")
        print("")
        print("1. CUDAバージョンに合ったwheelをダウンロード")
        print("2. pip install llama_cpp_python-*.whl")
        print("=" * 50)
    elif not AVAILABLE_HANDLERS:
        print("=" * 50)
        print("エラー: 対応VLMハンドラが見つかりません")
        print("=" * 50)
        print("")
        print("JamePengフォーク版が必要です:")
        print("  https://github.com/JamePeng/llama-cpp-python/releases")
        print("")
        print("pip uninstall llama-cpp-python")
        print("pip install path/to/llama_cpp_python-*.whl")
        print("=" * 50)
    else:
        print("=" * 50)
        print("RGT - Realtime GGUF Translate v8.1")
        print("=" * 50)
        print("")
        print("対応VLM: Qwen3.5 / Qwen3-VL / Gemma-4 / Gemma-3")
        print("")
        handlers = list(AVAILABLE_HANDLERS.keys())
        print(f"検出済みハンドラ: {handlers}")
        print("")
        print("=" * 50)
        
        app = VLMTranslator()
        app.root.mainloop()