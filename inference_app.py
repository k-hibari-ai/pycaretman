import os
from flask import Flask, request, render_template, jsonify, send_from_directory
import pandas as pd
import numpy as np
import logging
import glob
import json

# ロギングの設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# ファイル保存用フォルダ
UPLOAD_FOLDER = 'inference_uploads'
PREDICTION_RESULTS_FOLDER = 'inference_predictions'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PREDICTION_RESULTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PREDICTION_RESULTS_FOLDER'] = PREDICTION_RESULTS_FOLDER

# 学習アプリのRESULTS_FOLDERへのパス
# 環境に合わせて適切に設定してください
LEARNING_APP_RESULTS_FOLDER = '/Users/kotatsutakeda/Code/HIBARI/Pycaretman/models'
os.makedirs(LEARNING_APP_RESULTS_FOLDER, exist_ok=True)

# PyCaretの関数を動的にインポートするためのグローバル変数
pycaret_load_model = None
pycaret_predict_model = None

def load_pycaret_module_for_inference(task_type):
    global pycaret_load_model, pycaret_predict_model
    try:
        if task_type == 'classification':
            from pycaret.classification import load_model, predict_model
            logging.info("PyCaret Classification module loaded for inference.")
        elif task_type == 'regression':
            from pycaret.regression import load_model, predict_model
            logging.info("PyCaret Regression module loaded for inference.")
        else:
            raise ValueError(f"Invalid task type specified: {task_type}")

        pycaret_load_model = load_model
        pycaret_predict_model = predict_model
        return True
    except ImportError as e:
        logging.error(f"Error loading PyCaret module for {task_type}: {e}", exc_info=True)
        return False
    except Exception as e:
        logging.error(f"Unexpected error loading PyCaret module: {e}", exc_info=True)
        return False

@app.route('/')
def index():
    return render_template('inference_index.html')

@app.route('/get_available_models', methods=['GET'])
def get_available_models():
    """学習アプリのresultsフォルダから利用可能なモデルのリストを返します。"""
    try:
        if not os.path.exists(LEARNING_APP_RESULTS_FOLDER):
            logging.warning(f"LEARNING_APP_RESULTS_FOLDER not found: {LEARNING_APP_RESULTS_FOLDER}")
            return jsonify({'success': True, 'models': [], 'message': '学習アプリのresultsフォルダが見つかりません。'})

        model_files = [f for f in os.listdir(LEARNING_APP_RESULTS_FOLDER) if f.endswith('.pkl')]
        
        # モデル情報を拡張して返す
        model_info = []
        for model_file in model_files:
            # ファイル名からモデル情報を推定
            model_info_dict = {
                'filename': model_file,
                'display_name': model_file.replace('.pkl', '').replace('_', ' '),
                'size': os.path.getsize(os.path.join(LEARNING_APP_RESULTS_FOLDER, model_file)),
                'modified_time': os.path.getmtime(os.path.join(LEARNING_APP_RESULTS_FOLDER, model_file))
            }
            model_info.append(model_info_dict)
        
        # 変更時刻でソート（新しい順）
        model_info.sort(key=lambda x: x['modified_time'], reverse=True)
        
        logging.info(f"Available models in {LEARNING_APP_RESULTS_FOLDER}: {[m['filename'] for m in model_info]}")
        return jsonify({'success': True, 'models': model_info})
    except Exception as e:
        logging.error(f"利用可能なモデルの取得に失敗しました: {e}", exc_info=True)
        return jsonify({'error': f"利用可能なモデルの取得に失敗しました: {str(e)}"}), 500

@app.route('/upload_data_and_predict', methods=['POST'])
def upload_data_and_predict():
    """データをアップロードし、選択されたモデルで推論を実行します。"""
    selected_model_filename = request.form.get('modelFileName')
    data_file = request.files.get('dataFile')
    task_type = request.form.get('taskType')
    
    # 推論設定の取得
    batch_size = request.form.get('batchSize', type=int, default=1000)
    include_probability = request.form.get('includeProbability', 'false').lower() == 'true'
    probability_threshold = request.form.get('probabilityThreshold', type=float, default=0.5)
    output_format = request.form.get('outputFormat', 'csv')
    include_input_features = request.form.get('includeInputFeatures', 'false').lower() == 'true'

    if not selected_model_filename:
        return jsonify({'error': 'モデルが選択されていません。'}), 400
    if not data_file or data_file.filename == '':
        return jsonify({'error': '推論データファイルが選択されていません。'}), 400
    if not task_type:
        return jsonify({'error': 'モデルのタスクタイプ（分類または回帰）が選択されていません。'}), 400

    if not load_pycaret_module_for_inference(task_type):
        return jsonify({'error': f'選択されたタスクタイプ ({task_type}) 用のPyCaretモジュール読み込みに失敗しました。'}), 500

    if not data_file.filename.endswith('.csv'):
        return jsonify({'error': '無効なデータファイル形式です。(.csvファイルを選択してください)'}), 400

    timestamp = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")
    model_filepath = os.path.join(LEARNING_APP_RESULTS_FOLDER, selected_model_filename)
    data_filename_base, data_ext = os.path.splitext(data_file.filename)
    data_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f'inference_data_{timestamp}{data_ext}')

    try:
        data_file.save(data_filepath)
        logging.info(f"Selected model path: {model_filepath}")
        logging.info(f"Uploaded inference data to {data_filepath}")

        # モデルファイルの存在確認
        if not os.path.exists(model_filepath):
            return jsonify({'error': f"選択されたモデルファイル '{selected_model_filename}' が見つかりません。学習アプリのresultsフォルダを確認してください。"})

        # モデルのロード
        model_path_no_ext = model_filepath.replace('.pkl', '')
        loaded_model = pycaret_load_model(model_path_no_ext)
        logging.info("Model loaded successfully.")

        # データの読み込み
        input_df = pd.read_csv(data_filepath)
        logging.info(f"Data loaded successfully. Shape: {input_df.shape}")

        # バッチ処理での推論
        total_rows = len(input_df)
        all_predictions = []

        for i in range(0, total_rows, batch_size):
            batch_end = min(i + batch_size, total_rows)
            batch_df = input_df.iloc[i:batch_end].copy()
            
            logging.info(f"Processing batch {i//batch_size + 1}: rows {i} to {batch_end-1}")
            
            # バッチでの予測実行
            batch_predictions = pycaret_predict_model(loaded_model, data=batch_df)
            all_predictions.append(batch_predictions)

        # すべてのバッチの結果を結合
        predictions = pd.concat(all_predictions, ignore_index=True)
        logging.info(f"All predictions completed. Total predictions: {len(predictions)}")

        # 予測結果の処理
        result_df = pd.DataFrame()

        # 入力特徴量を含めるかどうか
        if include_input_features:
            result_df = input_df.copy()

        # 予測値の抽出とカラム追加
        if task_type == 'classification':
            if 'prediction_label' in predictions.columns:
                result_df['predicted_class'] = predictions['prediction_label']
                
                if include_probability and 'prediction_score' in predictions.columns:
                    result_df['prediction_probability'] = predictions['prediction_score']
                    
                    # 確率閾値の適用
                    if probability_threshold != 0.5:
                        # 閾値による予測の再調整
                        adjusted_predictions = (predictions['prediction_score'] >= probability_threshold).astype(int)
                        result_df['adjusted_prediction'] = adjusted_predictions
                        
        elif task_type == 'regression':
            if 'prediction' in predictions.columns:
                result_df['predicted_value'] = predictions['prediction']
            elif 'Label' in predictions.columns:
                result_df['predicted_value'] = predictions['Label']
            else:
                # 他の予測カラムを探す
                numeric_cols = predictions.select_dtypes(include=['number']).columns
                new_cols = [col for col in numeric_cols if col not in input_df.columns]
                if new_cols:
                    result_df['predicted_value'] = predictions[new_cols[0]]

        # 結果の統計情報を計算
        stats_info = {}
        if task_type == 'classification':
            if 'predicted_class' in result_df.columns:
                stats_info['class_distribution'] = result_df['predicted_class'].value_counts().to_dict()
                stats_info['total_predictions'] = len(result_df)
        elif task_type == 'regression':
            if 'predicted_value' in result_df.columns:
                stats_info['mean_prediction'] = float(result_df['predicted_value'].mean())
                stats_info['std_prediction'] = float(result_df['predicted_value'].std())
                stats_info['min_prediction'] = float(result_df['predicted_value'].min())
                stats_info['max_prediction'] = float(result_df['predicted_value'].max())

        # 結果ファイルの保存
        output_filename = f"predictions_{timestamp}"
        
        if output_format == 'csv':
            output_filepath = os.path.join(app.config['PREDICTION_RESULTS_FOLDER'], f"{output_filename}.csv")
            result_df.to_csv(output_filepath, index=False)
        elif output_format == 'json':
            output_filepath = os.path.join(app.config['PREDICTION_RESULTS_FOLDER'], f"{output_filename}.json")
            result_df.to_json(output_filepath, orient='records', indent=2)
        else:  # Excel
            output_filepath = os.path.join(app.config['PREDICTION_RESULTS_FOLDER'], f"{output_filename}.xlsx")
            result_df.to_excel(output_filepath, index=False)

        final_filename = os.path.basename(output_filepath)
        
        logging.info(f"Predictions saved to {output_filepath}")

        return jsonify({
            'success': True,
            'prediction_filename': final_filename,
            'stats': stats_info,
            'total_predictions': len(result_df),
            'batch_count': len(all_predictions),
            'output_format': output_format
        })

    except Exception as e:
        logging.error(f"推論中にエラーが発生しました: {str(e)}", exc_info=True)
        return jsonify({'error': f'推論中にエラーが発生しました: {str(e)}'}), 500
    finally:
        # 使用済みファイルのクリーンアップ
        if os.path.exists(data_filepath):
            try:
                os.remove(data_filepath)
                logging.info(f"Cleaned up inference data file: {data_filepath}")
            except Exception as e:
                logging.error(f"Error removing inference data file {data_filepath}: {e}")

@app.route('/download_predictions/<filename>', methods=['GET'])
def download_predictions(filename):
    """生成された予測結果ファイルをダウンロードさせます。"""
    logging.info(f"Download request for predictions: {filename}")
    try:
        return send_from_directory(app.config['PREDICTION_RESULTS_FOLDER'], filename, as_attachment=True)
    except FileNotFoundError:
        logging.error(f"Prediction file not found: {filename}")
        return "Prediction file not found.", 404
    except Exception as e:
        logging.error(f"Error serving prediction file {filename}: {e}")
        return "An error occurred during download.", 500

@app.route('/get_prediction_history', methods=['GET'])
def get_prediction_history():
    """過去の推論結果の履歴を返します。"""
    try:
        if not os.path.exists(app.config['PREDICTION_RESULTS_FOLDER']):
            return jsonify({'success': True, 'history': []})

        files = os.listdir(app.config['PREDICTION_RESULTS_FOLDER'])
        prediction_files = [f for f in files if f.startswith('predictions_')]
        
        history = []
        for filename in prediction_files:
            filepath = os.path.join(app.config['PREDICTION_RESULTS_FOLDER'], filename)
            file_stats = os.stat(filepath)
            
            history.append({
                'filename': filename,
                'size': file_stats.st_size,
                'created_time': file_stats.st_ctime,
                'format': filename.split('.')[-1] if '.' in filename else 'unknown'
            })
        
        # 作成時刻でソート（新しい順）
        history.sort(key=lambda x: x['created_time'], reverse=True)
        
        return jsonify({'success': True, 'history': history})
    except Exception as e:
        logging.error(f"推論履歴の取得に失敗しました: {e}", exc_info=True)
        return jsonify({'error': f"推論履歴の取得に失敗しました: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)