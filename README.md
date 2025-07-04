# PyCaretMan: PyCaret Web UI

PyCaretManは、ローコード機械学習ライブラリである[PyCaret](https://pycaret.org/)を、Webブラウザから直感的に操作できるようにするためのGUIツールです。
このツールは3つの独立したFlaskアプリケーションで構成されており、モデルの**学習**、**推論**、そして**解釈（SHAP）**を簡単に行うことができます。

## 主な機能

### 1. モデル学習 (`app.py`)
- **CSVファイルのアップロード**: 学習用・テスト用のデータをアップロードします。
- **タスク選択**: 「分類」または「回帰」の機械学習タスクを選択できます。
- **モデル比較**: `compare_models()` を実行し、データセットに最適なモデルを自動で探索します。
- **個別モデル学習**: 特定のモデルを選択し、ハイパーパラメータを調整しながら学習を進めることも可能です。
- **交差検証**: k-foldやLOOCV（Leave-One-Out Cross-Validation）などの検証戦略を選択できます。
- **モデルの保存**: 学習済みのモデルは`.pkl`ファイルとして保存され、推論や分析で再利用できます。

### 2. モデル推論 (`inference_app.py`)
- **モデルの選択**: `app.py`で学習・保存したモデルを一覧から選択します。
- **データのアップロード**: 推論したい新しいデータをCSVファイルでアップロードします。
- **バッチ推論**: 大規模なデータにも対応できるよう、データをバッチ処理して推論を実行します。
- **結果の保存**: 推論結果をCSV、JSON、Excel形式でダウンロードできます。
- **履歴の確認**: 過去の推論結果の履歴を一覧で確認できます。

### 3. モデル解釈 (`shap_app.py`)
- **SHAP分析**: 学習済みモデルの予測根拠を[SHAP (SHapley Additive exPlanations)](https://shap.readthedocs.io/en/latest/index.html)を用いて分析します。
- **モデル評価**: データに対するモデルの精度（Accuracy, R2など）を評価します。
- **可視化**:
    - **Summary Plot**: 各特徴量がモデルの予測に与える影響の全体像を可視化します。
    - **Bar Plot**: 特徴量の重要度を平均SHAP値でランキング表示します。
    - **Dependence Plot**: 特定の特徴量とSHAP値の関係性を可視化します。
    - **Waterfall Plot**: 個々の予測に対する各特徴量の貢献度を可視化します。
- **レポートのダウンロード**: 分析結果のグラフを画像ファイルとしてダウンロードできます。

## 技術スタック
- **バックエンド**: Python, Flask, PyCaret, SHAP, Pandas, scikit-learn
- **フロントエンド**: HTML, CSS, JavaScript (テンプレートエンジンとしてJinja2を使用)

## ディレクトリ構成
```
/
├── app.py                    # 1. モデル学習アプリケーション
├── inference_app.py          # 2. モデル推論アプリケーション
├── shap_app.py               # 3. モデル解釈アプリケーション
├── requirements.txt          # Pythonの依存ライブラリ
├── templates/                # HTMLテンプレート
│   ├── index.html
│   ├── inference_index.html
│   └── shap_index.html
├── models/                   # (自動生成) 学習済みモデルが保存される
├── uploads/                  # (自動生成) アップロードされた一時ファイル
├── inference_predictions/    # (自動生成) 推論結果が保存される
└── shap_results/             # (自動生成) SHAP分析の画像が保存される
```

## セットアップと実行方法

### 1. 依存ライブラリのインストール
まず、`requirements.txt`に記載されているライブラリをインストールします。

```bash
pip install -r requirements.txt
```

### 2. 各アプリケーションの実行
本ツールは機能ごとに3つのアプリケーションに分かれています。それぞれを別のターミナルで実行してください。

**ターミナル1: モデル学習アプリ**
```bash
python app.py
```
ブラウザで `http://127.0.0.1:5000` を開きます。

**ターミナル2: モデル推論アプリ**
```bash
python inference_app.py
```
ブラウザで `http://127.0.0.1:5001` を開きます。

**ターミナル3: モデル解釈アプリ**
```bash
python shap_app.py
```
ブラウザで `http://127.0.0.1:5002` を開きます。

## 使用方法

1. **モデルの学習 (`:5000`)**
   - `app.py`を起動し、学習用のCSVデータをアップロードします。
   - 目標となる列やタスクの種類（分類/回帰）を選択します。
   - 「PyCaret実行」ボタンを押し、モデルの学習と評価を行います。
   - 結果を確認し、最適なモデルを保存します。

2. **推論の実行 (`:5001`)**
   - `inference_app.py`を起動します。
   - ステップ1で保存したモデルを選択し、推論したいデータをアップロードします。
   - 「推論実行」ボタンを押し、予測結果を生成・ダウンロードします。

3. **モデルの解釈 (`:5002`)**
   - `shap_app.py`を起動します。
   - 分析したいモデルと、分析に使用するデータセットを選択します。
   - 「SHAP分析実行」ボタンを押し、モデルの予測根拠を可視化・分析します。
