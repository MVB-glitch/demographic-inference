# -*- coding: utf-8 -*-
"""
LLM Evidence Extractor using Google Gemini.

Two extraction functions:
1. extract_location_evidence() — Returns DST mass functions for location (K/NK/Omega)
   with confidence scores for 4 independent evidence sources.
2. extract_demographics_evidence() — Returns mass functions for age and gender.
"""

import json
import time
from typing import Optional

from PIL import Image

from .config import PipelineConfig
from .data_loader import load_local_image
from .dst_engine import validate_mass


class LLMExtractor:
    """Wraps the Gemini API for evidence extraction."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._model = None

    @property
    def model(self):
        """Lazy-initialize the Gemini model."""
        if self._model is None:
            import google.generativeai as genai
            genai.configure(api_key=self.config.api_key)
            self._model = genai.GenerativeModel(
                self.config.model_name,
                generation_config={"response_mime_type": "application/json"},
            )
        return self._model

    def _load_user_images(self, user: dict) -> list[Image.Image]:
        """Load all available images for a user (profile + post images)."""
        if not self.config.image_dir:
            return []

        images = []
        # Profile image
        profile_file = user.get("profile_picture_url")
        if profile_file:
            img = load_local_image(profile_file, self.config.image_dir, self.config.image_max_size)
            if img:
                images.append(img)

        # Post images
        for post in user["posts"]:
            img_file = post.get("image_file")
            if img_file:
                img = load_local_image(img_file, self.config.image_dir, self.config.image_max_size)
                if img:
                    images.append(img)

        return images

    def _call_with_retry(self, payload: list, context_label: str = "") -> Optional[dict]:
        """Call Gemini API with retry logic for rate limits and transient errors."""
        for attempt in range(self.config.max_retries):
            try:
                response = self.model.generate_content(payload)
                return json.loads(response.text)
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "Quota" in error_msg:
                    wait = self.config.quota_wait_seconds
                    print(f"  [{context_label}] Rate limit reached. Waiting {wait}s... (attempt {attempt+1})")
                    time.sleep(wait)
                else:
                    wait = self.config.error_wait_seconds
                    print(f"  [{context_label}] Error: {error_msg}. Waiting {wait}s... (attempt {attempt+1})")
                    time.sleep(wait)
        return None  # All retries exhausted

    def extract_location_evidence(self, user: dict) -> dict:
        """
        Extract location evidence (Kanto vs Non-Kanto) from user data.

        Returns DST mass functions for 4 independent sources:
        A: Posts text, B: Friend network, C: Bio/Handle/UserID, D: Images
        Each with a confidence score for discounting.
        """
        prompt = f"""
あなたは日本語言語学、ネットワーク分析、画像分析の専門家です。

このユーザーが
K = 関東在住
NK = 非関東在住
である証拠を評価してください。

証拠は4つの独立した情報源から評価します：

A: 投稿テキスト
B: 友人・インタラクションネットワーク
C: Bio + Handle + User ID
D: 画像

重要：
各情報源は他の情報源に依存せず独立に評価してください。

---

データ

User ID: {user['user_id']}
Handle: {user['handle']}
Bio: {user['bio']}
Posts: {[post['text'] for post in user['posts']]}
Friends: {user['interactions']}

---

画像がある場合：
背景、看板、文字、建物、食べ物などから地域的手がかりを分析してください。

---

質量関数は Dempster-Shafer Theory に基づきます。

各質量関数は：

m(K) + m(NK) + m(Omega) = 1.00

Omega は「不確実性」を表します。

値は小数点2桁に丸めてください。

---

信頼度の目安

0.0 = 情報なし
0.2 = 弱い証拠
0.5 = 中程度の証拠
0.8 = 強い証拠
1.0 = 非常に明確

---

出力は必ず以下のJSONのみ：

{{
"features_A_posts": [],
"mass_A_posts": {{"K":0.00,"NK":0.00,"Omega":1.00}},
"conf_A":0.00,

"features_B_friends": [],
"mass_B_friends": {{"K":0.00,"NK":0.00,"Omega":1.00}},
"conf_B":0.00,

"features_C_bio": [],
"mass_C_bio": {{"K":0.00,"NK":0.00,"Omega":1.00}},
"conf_C":0.00,

"features_D_images": [],
"mass_D_images": {{"K":0.00,"NK":0.00,"Omega":1.00}},
"conf_D":0.00,
"reasoning": "必ず日本語で理由を説明してください"
}}

JSON以外のテキストは絶対に出力しないでください。

---

特別ルール：

画像が無い場合：
conf_D = 0.0
mass_D_images = {{"K":0.00,"NK":0.00,"Omega":1.00}}

Bio / Handle / User ID が無い場合：
conf_C = 0.0
mass_C_bio = {{"K":0.00,"NK":0.00,"Omega":1.00}}
"""

        images = self._load_user_images(user)
        payload = [prompt] + images

        result = self._call_with_retry(payload, "Location")

        if result is None:
            return self._default_location_evidence()

        # Validate all mass functions from LLM response
        for key in ["mass_A_posts", "mass_B_friends", "mass_C_bio", "mass_D_images"]:
            if key in result:
                result[key] = validate_mass(result[key])

        return result

    def extract_demographics_evidence(self, user: dict) -> dict:
        """
        Extract age and gender evidence from user data.

        Returns mass functions for age (6 categories + Omega) and gender (M/F + Omega).
        """
        display_name = user.get("handle", "")
        posts_text = [post["text"] for post in user["posts"]]

        prompt = f"""
あなたはX（旧Twitter）ユーザーの属性推定を行う、慎重かつ公平なアノテーターです。
あなたのタスクは、各ユーザーのプロフィールやコンテンツから得られる属性の手がかりについて段階的に推論し、Dempster-Shafer Theory (DST) に基づく構造化されたデータを出力することです。

各ユーザーについて、以下の情報が提供されます：
・ユーザー名 / ハンドルネーム (Handle)
・表示名 (Display Name)
・短い自己紹介 (Bio)
・15つのポスト (Tweets)
・プロフィール画像 (Profile Image)

これらの入力に基づき、ユーザーの【年齢層】と【性別】を推定してください。

分析の観点：
・名前の形態論と文字・言語形式（接尾辞、命名規則など）
・言語スタイル、文法、語彙の選択
・ポストの話題、興味・関心、トーン
・文化的、地域的、または言語的な手がかり
・テキスト内の絵文字、記号、または文体的な特徴
・プロフィール画像から得られる年齢、性別に関する視覚的な手がかり

DST（Dempster-Shafer Theory）のルール：
証拠が不十分または曖昧な場合は、無理に推測せず、不確実性（Omega）に質量（mass）を割り当ててください。
各属性の質量関数の合計は必ず 1.00 になるようにしてください。

【年齢層】
A = 18–24歳
B = 25–34歳
C = 35–44歳
D = 45–54歳
E = 55–64歳
F = 65歳以上
ルール: m(A) + m(B) + m(C) + m(D) + m(E) + m(F) + m(Omega) = 1.00

【性別】
M = 男性
F = 女性
ルール: m(M) + m(F) + m(Omega) = 1.00

---
データ
User ID: {user['user_id']}
Handle: {user['handle']}
Display Name: {display_name}
Bio: {user['bio']}
Posts: {posts_text}
---

出力は必ず以下のJSONフォーマットのみにしてください。JSON以外のテキストは絶対に出力しないでください：
{{
 "reasoning": "年齢と性別の推論過程、および特定した具体的な証拠を日本語で説明してください",
 "mass_age": {{
  "18_24": 0.00,
  "25_34": 0.00,
  "35_44": 0.00,
  "45_54": 0.00,
  "55_64": 0.00,
  "65_plus": 0.00,
  "Omega": 1.00
 }},
 "mass_gender": {{
  "M": 0.00,
  "F": 0.00,
  "Omega": 1.00
 }}
}}
"""

        images = self._load_user_images(user)
        payload = [prompt] + images

        result = self._call_with_retry(payload, "Demographics")

        if result is None:
            return self._default_demographics_evidence()

        # Validate mass functions
        if "mass_age" in result:
            result["mass_age"] = validate_mass(result["mass_age"])
        if "mass_gender" in result:
            result["mass_gender"] = validate_mass(result["mass_gender"])

        return result

    @staticmethod
    def _default_location_evidence() -> dict:
        """Fallback when API calls fail entirely."""
        return {
            "mass_A_posts": {"K": 0, "NK": 0, "Omega": 1}, "conf_A": 0,
            "mass_B_friends": {"K": 0, "NK": 0, "Omega": 1}, "conf_B": 0,
            "mass_C_bio": {"K": 0, "NK": 0, "Omega": 1}, "conf_C": 0,
            "mass_D_images": {"K": 0, "NK": 0, "Omega": 1}, "conf_D": 0,
            "reasoning": "APIエラーのため推論不可",
        }

    @staticmethod
    def _default_demographics_evidence() -> dict:
        """Fallback when API calls fail entirely."""
        return {
            "reasoning": "APIエラーのため推論不可",
            "mass_age": {"18_24": 0, "25_34": 0, "35_44": 0, "45_54": 0, "55_64": 0, "65_plus": 0, "Omega": 1},
            "mass_gender": {"M": 0, "F": 0, "Omega": 1},
        }
