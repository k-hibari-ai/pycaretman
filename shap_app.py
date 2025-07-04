import os
import pandas as pd
import numpy as np
import shap
from flask import Flask, render_template, request, send_from_directory, jsonify
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
import logging
import json
from datetime import datetime
import matplotlib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
matplotlib.use('Agg')

# 日本語フォント設定
plt.rcParams['font.family'] = ['DejaVu Sans', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

# ロギングの設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# フォルダ設定（先ほどのアプリケーションと統合）
UPLOAD_FOLDER = 'uploads'
MODELS_FOLDER = 'models'  # 学習アプリと共有
SHAP_RESULTS_FOLDER = 'shap_results'
INFERENCE_UPLOADS_FOLDER = 'inference_uploads'
SHAP_UPLOADS_FOLDER = 'shap_uploads'  # 解析眼専用アップロードフォルダ

for folder in [UPLOAD_FOLDER, MODELS_FOLDER, SHAP_RESULTS_FOLDER, INFERENCE_UPLOADS_FOLDER, SHAP_UPLOADS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODELS_FOLDER'] = MODELS_FOLDER
app.config['SHAP_RESULTS_FOLDER'] = SHAP_RESULTS_FOLDER
app.config['INFERENCE_UPLOADS_FOLDER'] = INFERENCE_UPLOADS_FOLDER

def get_available_models():
    """利用可能なモデルファイルのリストを取得"""
    models = []
    if os.path.exists(MODELS_FOLDER):
        for file in os.listdir(MODELS_FOLDER):
            if file.endswith('.pkl'):
                filepath = os.path.join(MODELS_FOLDER, file)
                models.append({
                    'filename': file,
                    'display_name': file.replace('.pkl', '').replace('_', ' '),
                    'size': os.path.getsize(filepath),
                    'modified_time': os.path.getmtime(filepath)
                })
    # 変更時刻でソート（新しい順）
    models.sort(key=lambda x: x['modified_time'], reverse=True)
    return models

def get_available_datasets():
    """利用可能なデータセットのリストを取得"""
    datasets = []
    folders = [UPLOAD_FOLDER, INFERENCE_UPLOADS_FOLDER, SHAP_UPLOADS_FOLDER]
    
    for folder in folders:
        if os.path.exists(folder):
            for file in os.listdir(folder):
                if file.endswith('.csv'):
                    filepath = os.path.join(folder, file)
                    datasets.append({
                        'filename': file,
                        'filepath': filepath,
                        'folder': folder,
                        'folder_display': {
                            UPLOAD_FOLDER: '学習データ',
                            INFERENCE_UPLOADS_FOLDER: '推論データ', 
                            SHAP_UPLOADS_FOLDER: '解析眼専用'
                        }.get(folder, folder),
                        'size': os.path.getsize(filepath),
                        'modified_time': os.path.getmtime(filepath)
                    })
    # 変更時刻でソート（新しい順）
    datasets.sort(key=lambda x: x['modified_time'], reverse=True)
    return datasets

def load_pycaret_model(model_path):
    """PyCaretモデルを安全にロード"""
    try:
        # PyCaretモデルの場合、拡張子を除いたパスでロード
        if model_path.endswith('.pkl'):
            model_path_no_ext = model_path[:-4]
        else:
            model_path_no_ext = model_path
            
        # まずPyCaretのload_modelを試す
        try:
            # PyCaretがインストールされている場合の動的インポート
            try:
                from pycaret.classification import load_model as load_classification_model
                model = load_classification_model(model_path_no_ext)
                return model, 'classification'
            except:
                from pycaret.regression import load_model as load_regression_model
                model = load_regression_model(model_path_no_ext)
                return model, 'regression'
        except Exception as e:
            logging.warning(f"PyCaretでのロードに失敗: {e}")
            # 通常のpickleファイルとしてロード
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            return model, 'unknown'
            
    except Exception as e:
        logging.error(f"モデルロードエラー: {e}")
        raise

def extract_base_model(model):
    """PyCaretモデルから基本モデルを抽出"""
    if hasattr(model, 'steps'):
        # Pipeline形式の場合
        return model.steps[-1][1]
    elif hasattr(model, 'named_steps'):
        # sklearn Pipelineの場合
        step_names = list(model.named_steps.keys())
        return model.named_steps[step_names[-1]]
    else:
        return model

def create_shap_explainer(model, X_sample, model_type='unknown'):
    """適切なSHAP Explainerを作成"""
    base_model = extract_base_model(model)
    
    try:
        # TreeExplainerを最初に試す（高速）
        explainer = shap.TreeExplainer(base_model, feature_perturbation='interventional', model_output='raw')
        logging.info("TreeExplainerを使用します")
        return explainer, 'tree'
    except Exception as e:
        logging.warning(f"TreeExplainer失敗: {e}")
        
        try:
            # LinearExplainerを試す
            explainer = shap.LinearExplainer(base_model, X_sample)
            logging.info("LinearExplainerを使用します")
            return explainer, 'linear'
        except Exception as e:
            logging.warning(f"LinearExplainer失敗: {e}")
            
            try:
                # KernelExplainerにフォールバック（汎用だが低速）
                X_summary = shap.sample(X_sample, min(100, len(X_sample)))
                explainer = shap.KernelExplainer(model.predict, X_summary)
                logging.info("KernelExplainerを使用します")
                return explainer, 'kernel'
            except Exception as e:
                logging.error(f"全てのExplainer失敗: {e}")
                raise

def save_plot_as_base64():
    """現在のプロットをBase64エンコードした文字列として保存"""
    img = BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', dpi=300, facecolor='white')
    img.seek(0)
    plot_data = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_data

def save_plot_as_file(filename):
    """現在のプロットをファイルとして保存"""
    filepath = os.path.join(SHAP_RESULTS_FOLDER, filename)
    plt.savefig(filepath, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    return filepath

@app.route('/')
def index():
    """メインページ"""
    models = get_available_models()
    datasets = get_available_datasets()
    return render_template('shap_index.html', models=models, datasets=datasets)

@app.route('/upload_dataset', methods=['POST'])
def upload_dataset():
    """解析眼専用のデータセットアップロード"""
    try:
        if 'dataset_file' not in request.files:
            return jsonify({'error': 'ファイルが選択されていません'}), 400
            
        file = request.files['dataset_file']
        if file.filename == '':
            return jsonify({'error': 'ファイルが選択されていません'}), 400
            
        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'CSVファイルのみアップロード可能です'}), 400
            
        # タイムスタンプ付きのユニークなファイル名生成
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_base, ext = os.path.splitext(file.filename)
        safe_filename = f"shap_{timestamp}_{filename_base}{ext}"
        
        filepath = os.path.join(SHAP_UPLOADS_FOLDER, safe_filename)
        file.save(filepath)
        
        # ファイル情報を取得
        file_info = {
            'filename': safe_filename,
            'filepath': filepath,
            'folder': SHAP_UPLOADS_FOLDER,
            'folder_display': '解析眼専用',
            'size': os.path.getsize(filepath),
            'modified_time': os.path.getmtime(filepath)
        }
        
        logging.info(f"データセットアップロード完了: {safe_filename}")
        return jsonify({'success': True, 'file_info': file_info})
        
    except Exception as e:
        logging.error(f"データセットアップロードエラー: {e}")
        return jsonify({'error': f'アップロードエラー: {str(e)}'}), 500

@app.route('/get_columns', methods=['POST'])
def get_columns():
    """選択されたデータセットのカラム情報を取得"""
    try:
        data = request.json
        dataset_path = data.get('dataset_path')
        
        if not dataset_path or not os.path.exists(dataset_path):
            return jsonify({'error': 'データセットが見つかりません'}), 400
            
        df = pd.read_csv(dataset_path)
        columns = df.columns.tolist()
        
        # データ基本情報
        info = {
            'shape': df.shape,
            'columns': columns,
            'dtypes': df.dtypes.astype(str).to_dict(),
            'null_counts': df.isnull().sum().to_dict(),
            'sample_data': df.head(3).to_dict('records')
        }
        
        return jsonify({'success': True, 'info': info})
        
    except Exception as e:
        logging.error(f"カラム取得エラー: {e}")
        return jsonify({'error': f'カラム取得エラー: {str(e)}'}), 500



@app.route('/run_shap_analysis', methods=['POST'])
def run_shap_analysis():
    """SHAP解析を実行"""
    try:
        data = request.json
        model_filename = data.get('model_filename')
        dataset_path = data.get('dataset_path')
        target_column = data.get('target_column')
        analysis_type = data.get('analysis_type', 'summary')
        max_samples = data.get('max_samples', 1000)
        
        # バリデーション
        if not all([model_filename, dataset_path, target_column]):
            return jsonify({'error': '必要なパラメータが不足しています'}), 400
            
        model_path = os.path.join(MODELS_FOLDER, model_filename)
        if not os.path.exists(model_path):
            return jsonify({'error': 'モデルファイルが見つかりません'}), 400
            
        if not os.path.exists(dataset_path):
            return jsonify({'error': 'データセットが見つかりません'}), 400
            
        # データ読み込み
        df = pd.read_csv(dataset_path)
        if target_column not in df.columns:
            return jsonify({'error': f'目標列 "{target_column}" が見つかりません'}), 400
            
        # 特徴量とターゲットの分離
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # サンプリング（大きなデータセットの場合）
        if len(X) > max_samples:
            sample_indices = np.random.choice(len(X), max_samples, replace=False)
            X = X.iloc[sample_indices]
            y = y.iloc[sample_indices]
            logging.info(f"データを{max_samples}行にサンプリングしました")
        
        # モデル読み込み
        model, model_type = load_pycaret_model(model_path)
        logging.info(f"モデル読み込み完了: {model_type}")

        # 予測値の取得（変換前のデータを使用）
        try:
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X)[:, 1]
                y_pred = model.predict(X)
            else:
                y_pred = model.predict(X)
                y_pred_proba = None
        except Exception as e:
            logging.error(f"予測エラー: {e}")
            try:
                from pycaret.regression import predict_model as predict_regression
                from pycaret.classification import predict_model as predict_classification
                
                if model_type == 'regression':
                    pred_df = predict_regression(model, data=X)
                    y_pred = pred_df['prediction_label']
                else: # classification
                    pred_df = predict_classification(model, data=X)
                    y_pred = pred_df['prediction_label']
                    if 'prediction_score' in pred_df.columns:
                         y_pred_proba = pred_df['prediction_score']
                    else:
                        y_pred_proba = None

            except Exception as e_pycaret:
                logging.error(f"PyCaretでの予測エラー: {e_pycaret}")
                return jsonify({'error': f'モデル予測中にエラーが発生しました: {e_pycaret}'}), 500
        
        # パフォーマンスメトリクス計算
        metrics = {}
        if model_type == 'regression':
            metrics['mae'] = mean_absolute_error(y, y_pred)
            metrics['mse'] = mean_squared_error(y, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['r2'] = r2_score(y, y_pred)
        elif model_type == 'classification':
            try:
                metrics['accuracy'] = accuracy_score(y, y_pred)
                metrics['precision'] = precision_score(y, y_pred, average='weighted')
                metrics['recall'] = recall_score(y, y_pred, average='weighted')
                metrics['f1_score'] = f1_score(y, y_pred, average='weighted')
                if y_pred_proba is not None and len(np.unique(y)) == 2:
                     metrics['auc'] = roc_auc_score(y, y_pred_proba)
            except ValueError as e:
                logging.warning(f"分類メトリクスの計算に失敗しました: {e}。回帰メトリクスを試行します。")
                if 'continuous is not supported' in str(e):
                    model_type = 'regression'
                    metrics['mae'] = mean_absolute_error(y, y_pred)
                    metrics['mse'] = mean_squared_error(y, y_pred)
                    metrics['rmse'] = np.sqrt(metrics['mse'])
                    metrics['r2'] = r2_score(y, y_pred)
                else:
                    raise

        # SHAP解析用にデータを前処理
        X_shap = X
        if hasattr(model, 'steps'):
            try:
                from sklearn.pipeline import Pipeline
                preprocessor = Pipeline(model.steps[:-1])
                X_transformed = preprocessor.transform(X)
                
                feature_names = []
                try:
                    feature_names = preprocessor.get_feature_names_out()
                except Exception:
                    pass # Fallback to generic names
                
                if len(list(feature_names)) == 0:
                    feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]

                X_shap = pd.DataFrame(X_transformed, columns=feature_names, index=X.index)
                logging.info("SHAP解析用にデータを変換しました")
            except Exception as e:
                logging.error(f"データの前処理に失敗しました: {e}。元のデータを使用します。")
        
        # SHAP Explainer作成
        explainer, explainer_type = create_shap_explainer(model, X_shap, model_type)
        
        # SHAP値計算
        logging.info("SHAP値を計算中...")
        if explainer_type == 'tree':
            shap_values = explainer.shap_values(X_shap, check_additivity=False)
        elif explainer_type == 'linear':
            shap_values = explainer.shap_values(X_shap)
        else:  # kernel
            X_shap_small = X_shap.iloc[:min(100, len(X_shap))]
            shap_values = explainer.shap_values(X_shap_small)
            X_shap = X_shap_small
        
        if isinstance(shap_values, list) and len(shap_values) > 1:
            shap_values = shap_values[1]
        
        logging.info("SHAP値計算完了")
        
        results = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if analysis_type in ['summary', 'all']:
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_shap, show=False, max_display=20)
            results['summary_plot'] = save_plot_as_base64()
            save_plot_as_file(f'summary_plot_{timestamp}.png')
            
        if analysis_type in ['bar', 'all']:
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False, max_display=20)
            results['bar_plot'] = save_plot_as_base64()
            save_plot_as_file(f'bar_plot_{timestamp}.png')
            
        if analysis_type in ['dependence', 'all']:
            feature_importance_vals = np.abs(shap_values).mean(0)
            top_features_indices = np.argsort(feature_importance_vals)[-5:][::-1]
            top_features = X_shap.columns[top_features_indices]
            
            dependence_plots = {}
            for feature in top_features:
                plt.figure(figsize=(10, 6))
                shap.dependence_plot(feature, shap_values, X_shap, show=False)
                dependence_plots[feature] = save_plot_as_base64()
                save_plot_as_file(f'dependence_{feature.replace("/", "_")}_{timestamp}.png')
            results['dependence_plots'] = dependence_plots
            
        if analysis_type in ['waterfall', 'all'] and len(X_shap) > 0:
            plt.figure(figsize=(12, 8))
            current_expected_value = explainer.expected_value
            if isinstance(current_expected_value, (list, np.ndarray)) and len(current_expected_value) > 1:
                current_expected_value = current_expected_value[1]

            if hasattr(shap, 'plots') and hasattr(shap.plots, 'waterfall'):
                shap_explanation = shap.Explanation(
                    values=shap_values[0],
                    base_values=current_expected_value,
                    data=X_shap.iloc[0].values,
                    feature_names=X_shap.columns.tolist()
                )
                shap.plots.waterfall(shap_explanation, show=False)
            elif hasattr(shap, 'waterfall_plot'):
                shap.waterfall_plot(current_expected_value, shap_values[0], X_shap.iloc[0], show=False)
            results['waterfall_plot'] = save_plot_as_base64()
            save_plot_as_file(f'waterfall_{timestamp}.png')
            
        feature_importance = np.abs(shap_values).mean(0)
        feature_importance_df = pd.DataFrame({
            'feature': X_shap.columns,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        results['feature_importance'] = feature_importance_df.head(10).to_dict('records')
        results['explainer_type'] = explainer_type
        results['model_type'] = model_type
        results['sample_size'] = len(X_shap)
        results['metrics'] = metrics
        
        logging.info("SHAP解析完了")
        return jsonify({'success': True, 'results': results})
        
    except Exception as e:
        logging.error(f"SHAP解析エラー: {e}", exc_info=True)
        return jsonify({'error': f'SHAP解析エラー: {str(e)}'}), 500

@app.route('/download_results')
def download_results():
    """SHAP解析結果をダウンロード"""
    try:
        # 最新の結果ファイルを取得
        files = []
        if os.path.exists(SHAP_RESULTS_FOLDER):
            files = [f for f in os.listdir(SHAP_RESULTS_FOLDER) if f.endswith('.png')]
            files.sort(key=lambda x: os.path.getmtime(os.path.join(SHAP_RESULTS_FOLDER, x)), reverse=True)
        
        return jsonify({'success': True, 'files': files})
        
    except Exception as e:
        logging.error(f"ダウンロードリスト取得エラー: {e}")
        return jsonify({'error': f'ダウンロードリスト取得エラー: {str(e)}'}), 500

@app.route('/shap_results/<filename>')
def download_file(filename):
    """SHAP結果ファイルをダウンロード"""
    try:
        return send_from_directory(SHAP_RESULTS_FOLDER, filename, as_attachment=True)
    except FileNotFoundError:
        logging.error(f"ファイルが見つかりません: {filename}")
        return "ファイルが見つかりません", 404
    except Exception as e:
        logging.error(f"ファイルダウンロードエラー: {e}")
        return "ダウンロードエラーが発生しました", 500

if __name__ == '__main__':
    app.run(debug=True, port=5002)