# 属性推定パイプライン

**LLM + Dempster-Shafer理論のハイブリッドパイプライン** — X（旧Twitter）ユーザーデータから属性情報（年齢、性別、居住地域）を推定します。

## 概要

本パイプラインは、Google Gemini（LLM）の自然言語理解能力とDempster-Shafer理論（DST）の数学的厳密性を組み合わせ、ソーシャルメディアデータからユーザーの属性を推定します。

### アーキテクチャ

```
┌─────────────────────┐
│   データ取り込み      │  CSV + 画像
│   (data_loader.py)  │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐     ┌─────────────────────┐
│   地域推定の証拠     │     │   属性推定の証拠      │
│  (llm_extractor.py) │     │  (llm_extractor.py)  │
│                     │     │                      │
│  4つの独立した       │     │  単一のLLM呼び出し    │
│  質量関数:          │     │  → mass_age           │
│  A: 投稿テキスト     │     │  → mass_gender        │
│  B: 友人ネットワーク  │     │                      │
│  C: Bio/ハンドル     │     └──────────┬───────────┘
│  D: 画像            │                │
└─────────┬───────────┘                │
          │                            │
          ▼                            │
┌─────────────────────┐                │
│   DST融合           │                │
│   (dst_engine.py)   │                │
│                     │                │
│   割引 → 融合        │                │
│   → 判定            │                │
└─────────┬───────────┘                │
          │                            │
          ▼                            ▼
┌─────────────────────────────────────────┐
│          結果と評価指標                   │
│          (metrics.py)                   │
│                                         │
│  地域: 関東 / 非関東 / 不確実             │
│  年齢: 18-24 / 25-34 / ... / 65+        │
│  性別: 男性 / 女性 / 不確実               │
└─────────────────────────────────────────┘
```

### DST-LLMハイブリッドの仕組み

1. **証拠抽出**: LLMが4つの独立したデータソースを分析し、それぞれの質量関数 `m(K)`, `m(NK)`, `m(Ω)` と信頼度スコアを出力します。
2. **割引（ディスカウント）**: 各質量関数を信頼度スコアで割引し（Shaferの割引操作）、信頼性の低い証拠を不確実性の方向にシフトさせます。
3. **融合**: Dempsterの結合ルールにより、割引済みの4つの質量関数を順次融合し、単一の信念構造にまとめます。
4. **判定**: 最も高い質量を持つ仮説が採用されます。ただし、Omega（不確実性）が50%を超える場合は「不確実」と判定されます。

## インストール

### 前提条件
- Python 3.10以上
- Google Gemini APIキー（[こちらから取得](https://aistudio.google.com/app/apikey)）

### セットアップ

```bash
# リポジトリをクローン
git clone https://github.com/your-username/demographic-inference.git
cd demographic-inference

# 仮想環境を作成
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

# 依存関係をインストール
pip install -r requirements.txt

# パッケージとしてインストール（`demoinfer`コマンドが有効になります）
pip install -e .

# APIキーを設定
copy .env.example .env
# .envを編集し、GEMINI_API_KEYを追加してください
```

## 使い方

### コマンドライン

```bash
# 基本的な使い方
demoinfer --csv data/users.csv --images data/images/

# 最初の10ユーザーを5人ずつのバッチで処理
demoinfer --csv data/users.csv --images data/images/ --limit 10 --batch-size 5

# JSON形式でエクスポート
demoinfer --csv data/users.csv --format json --output results.json

# チェックポイントを無視して最初から実行
demoinfer --csv data/users.csv --no-resume

# python -m を使用
python -m demographic_inference --csv data/users.csv --images data/images/
```

### 全オプション一覧

```
オプション:
  --csv CSV             入力CSVファイルのパス（必須）
  --images PATH         画像ディレクトリのパス
  --output PATH         出力ファイルのパス（デフォルト: results.xlsx）
  --format FORMAT       xlsx, csv, json のいずれか（デフォルト: xlsx）
  --limit N             最初のN人のユーザーのみ処理
  --batch-size N        チェックポイントごとのユーザー数（デフォルト: 5）
  --max-posts N         ユーザーあたりの最大投稿数（デフォルト: 15）
  --api-key KEY         Gemini APIキー（または環境変数 GEMINI_API_KEY を設定）
  --model NAME          Geminiモデル名（デフォルト: gemini-2.5-flash）
  --checkpoint-dir DIR  チェックポイントディレクトリ（デフォルト: checkpoints/）
  --no-resume           チェックポイントを無視して最初から実行
  --verbose             詳細な出力を有効化
```

### Google Colab

本パイプラインはColabでも使用可能です。Colabのシークレットに `GEMINI_API_KEY` を設定すれば、configモジュールが自動的に読み込みます。

### CSV形式

入力CSVには以下のカラムが必要です：

| カラム名 | 説明 |
|--------|-------------|
| `user_id` | ユーザー固有ID |
| `handle` | X（Twitter）のハンドル名 |
| `bio_description` | ユーザーの自己紹介文 |
| `profile_picture_url` | プロフィール画像のファイル名 |
| `post_{1..15}_text` | 投稿のテキスト内容 |
| `post_{1..15}_image_url` | 投稿画像のファイル名 |
| `interaction_{1..5}_bio` | 交流したユーザーの自己紹介文 |
| `ground_truth` | 地域の正解ラベル |
| `ground_truth_age` | 年齢の正解ラベル（任意） |
| `ground_truth_gender` | 性別の正解ラベル（任意） |

### バッチ処理とチェックポイント

パイプラインは設定可能なバッチサイズ（デフォルト: 5人）でユーザーを処理します。各バッチの処理後、結果が `checkpoints/results_checkpoint.json` に保存されます。パイプラインが中断された場合、同じコマンドを再実行すると、最後のチェックポイントから自動的に再開されます。

```bash
# 100人のユーザーを5人ずつ処理（自動チェックポイント付き）
demoinfer --csv data/large_dataset.csv --limit 100 --batch-size 5
```

## プロジェクト構成

```
├── src/
│   └── demographic_inference/
│       ├── __init__.py          # パッケージ初期化
│       ├── __main__.py          # python -m エントリーポイント
│       ├── cli.py               # CLI引数解析
│       ├── config.py            # 設定とAPIキー解決
│       ├── data_loader.py       # CSV取り込みと画像読み込み
│       ├── llm_extractor.py     # Gemini API証拠抽出
│       ├── dst_engine.py        # DST数学処理（結合・割引・融合）
│       ├── pipeline.py          # バッチ/チェックポイント付きオーケストレータ
│       └── metrics.py           # 精度評価とレポート
├── tests/
│   ├── test_dst_engine.py       # DST数学ユニットテスト（29件）
│   └── test_data_loader.py      # データローダーテスト（14件）
├── data/
│   └── sample_users.csv         # サンプルテストデータ
├── pyproject.toml               # ビルド設定と依存関係
├── requirements.txt             # pip依存関係
├── .env.example                 # APIキーテンプレート
└── .gitignore
```

## テストの実行

```bash
python -m pytest tests/ -v
```

---

## コントリビューション

1. リポジトリをフォーク
2. 機能ブランチを作成（`git checkout -b feature/your-feature`）
3. テストを実行（`python -m pytest tests/ -v`）
4. コミットしてプッシュ
5. プルリクエストを作成
