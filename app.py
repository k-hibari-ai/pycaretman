import os
from flask import Flask, request, render_template, jsonify, send_from_directory
import pandas as pd
from sklearn.model_selection import train_test_split, LeaveOneOut
import logging
import glob
import json

# ロギングの設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'models'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

# PyCaretの関数を動的にインポートするためのグローバル変数
pycaret_setup = None
pycaret_compare_models = None
pycaret_create_model = None
pycaret_tune_model = None
pycaret_evaluate_model = None
pycaret_finalize_model = None
pycaret_predict_model = None
pycaret_pull = None
pycaret_save_model = None
pycaret_load_model = None

def load_pycaret_module(task_type):
    global pycaret_setup, pycaret_compare_models, pycaret_create_model, pycaret_tune_model
    global pycaret_evaluate_model, pycaret_finalize_model, pycaret_predict_model, pycaret_pull, pycaret_save_model, pycaret_load_model
    try:
        if task_type == 'classification':
            from pycaret.classification import (setup, compare_models, create_model, tune_model, 
                                              evaluate_model, finalize_model, predict_model, 
                                              pull, save_model, load_model)
            logging.info("PyCaret Classification module loaded.")
        elif task_type == 'regression':
            from pycaret.regression import (setup, compare_models, create_model, tune_model,
                                          evaluate_model, finalize_model, predict_model, 
                                          pull, save_model, load_model)
            logging.info("PyCaret Regression module loaded.")
        else:
            raise ValueError(f"Invalid task type specified: {task_type}")

        pycaret_setup = setup
        pycaret_compare_models = compare_models
        pycaret_create_model = create_model
        pycaret_tune_model = tune_model
        pycaret_evaluate_model = evaluate_model
        pycaret_finalize_model = finalize_model
        pycaret_predict_model = predict_model
        pycaret_pull = pull
        pycaret_save_model = save_model
        pycaret_load_model = load_model
        return True
    except ImportError as e:
        logging.error(f"Error loading PyCaret module for {task_type}: {e}", exc_info=True)
        return False
    except Exception as e:
        logging.error(f"Unexpected error loading PyCaret module: {e}", exc_info=True)
        return False

@app.route('/')
def index():
    return render_template('index.html')

# ファイルアップロード共通処理
def handle_file_upload(files, file_type_prefix):
    """複数のCSVファイルを処理し、結合して一時ファイルとして保存します。"""
    dfs = []
    timestamp = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")

    for i, file in enumerate(files):
        if file and file.filename.endswith('.csv'):
            filename_base, ext = os.path.splitext(file.filename)
            unique_filename = f"{file_type_prefix}_{timestamp}_{i}_{filename_base}{ext}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)
            
            try:
                df = pd.read_csv(filepath)
                dfs.append(df)
            except Exception as e:
                logging.error(f"ファイル読み込みエラー ({file.filename}): {str(e)}")
                if os.path.exists(filepath):
                    os.remove(filepath)
                return None, f"ファイル読み込みエラー ({file.filename}): {str(e)}"
        else:
            return None, f"無効なファイル形式です: {file.filename} (CSVのみ許可)"

    if not dfs:
        return None, f"有効な{file_type_prefix}ファイルが見つかりませんでした。"

    try:
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f'combined_{file_type_prefix}_data_{timestamp}.csv')
        combined_df.to_csv(combined_filepath, index=False)
        return combined_filepath, None
    except Exception as e:
        logging.error(f"{file_type_prefix}データ結合エラー: {str(e)}")
        return None, f"{file_type_prefix}データ結合エラー: {str(e)}"

@app.route('/upload_files', methods=['POST'])
def upload_files():
    """データ入力方法に応じたファイルをアップロードし、結合済みファイルのパスとカラム情報を返します。"""
    data_split_method = request.form.get('dataSplitMethod')

    train_filepath = None
    test_filepath = None
    columns = []

    if data_split_method == 'separate':
        train_files = request.files.getlist('trainFiles[]')
        if not train_files or all(f.filename == '' for f in train_files):
            return jsonify({'error': '訓練ファイルが選択されていません。'}), 400
        train_filepath, train_error = handle_file_upload(train_files, 'train')
        if train_error:
            return jsonify({'error': train_error}), 500

        test_files = request.files.getlist('testFiles[]')
        if test_files and not all(f.filename == '' for f in test_files):
            test_filepath, test_error = handle_file_upload(test_files, 'test')
            if test_error:
                return jsonify({'error': test_error}), 500
        else:
            logging.info("テストファイルはアップロードされませんでした。")

        try:
            train_df = pd.read_csv(train_filepath)
            columns = train_df.columns.tolist()
        except Exception as e:
            if train_filepath and os.path.exists(train_filepath):
                os.remove(train_filepath)
            if test_filepath and os.path.exists(test_filepath):
                os.remove(test_filepath)
            return jsonify({'error': f'訓練データからカラム情報を取得中にエラーが発生しました: {str(e)}'}), 500

    elif data_split_method == 'combined':
        combined_files = request.files.getlist('combinedFiles[]')
        if not combined_files or all(f.filename == '' for f in combined_files):
            return jsonify({'error': '結合ファイルが選択されていません。'}), 400
        combined_filepath, combined_error = handle_file_upload(combined_files, 'combined')
        if combined_error:
            return jsonify({'error': combined_error}), 500

        train_filepath = combined_filepath
        try:
            combined_df = pd.read_csv(combined_filepath)
            columns = combined_df.columns.tolist()
        except Exception as e:
            if combined_filepath and os.path.exists(combined_filepath):
                os.remove(combined_filepath)
            return jsonify({'error': f'結合データからカラム情報を取得中にエラーが発生しました: {str(e)}'}), 500
    else:
        return jsonify({'error': '無効なデータ分割方法が指定されました。'}), 400

    return jsonify({
        'success': True,
        'train_filepath': train_filepath,
        'test_filepath': test_filepath,
        'columns': columns,
        'data_split_method': data_split_method
    })

@app.route('/run_pycaret', methods=['POST'])
def run_pycaret():
    """PyCaret解析を実行し、モデル比較結果とテストデータ予測結果、保存されたモデルリストを返します。"""
    data = request.json
    train_filepath = data.get('train_filepath')
    original_test_filepath = data.get('test_filepath')
    target_column = data.get('target_column')
    pycaret_split = data.get('pycaret_split')
    test_split_ratio = data.get('test_split_ratio')
    cv_method = data.get('cv_method', 'default')
    task_type = data.get('task_type')
    analysis_mode = data.get('analysis_mode', 'compare')  # 'compare' or 'individual'
    
    # 個別モデル設定（analysis_mode が 'individual' の場合）
    selected_model = data.get('selected_model')
    hyperparameter_tuning = data.get('hyperparameter_tuning', False)
    custom_hyperparameters = data.get('custom_hyperparameters', {})

    if not train_filepath or not target_column:
        return jsonify({'error': '訓練データパスまたは目標列が指定されていません。'}), 400
    if not task_type:
        return jsonify({'error': 'タスクタイプ（分類または回帰）が選択されていません。'}), 400
    
    if not load_pycaret_module(task_type):
        return jsonify({'error': f'PyCaretのモジュール読み込みに失敗しました。選択されたタスクタイプ: {task_type}'}), 500

    current_train_df = None
    current_test_df_for_predict = None
    analysis_timestamp = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")

    try:
        raw_train_df = pd.read_csv(train_filepath)
        if target_column not in raw_train_df.columns:
            return jsonify({'error': f'訓練データに指定された目標列 "{target_column}" が存在しません。'}), 400

        # setup関数に渡すパラメータを設定
        setup_params = {'data': None, 'target': target_column, 'session_id': 123}
        if cv_method == 'loocv':
            setup_params['fold_strategy'] = LeaveOneOut()
            logging.info("setup(): fold_strategy=LeaveOneOut()を設定します。")
        else:
            setup_params['fold_strategy'] = 'kfold'
            logging.info("setup(): fold_strategy='kfold'を設定します。")

        # データ分割ロジック
        if original_test_filepath:
            logging.info("訓練/テストデータが別々に装填されました。")
            current_train_df = raw_train_df
            current_test_df_for_predict = pd.read_csv(original_test_filepath)
            setup_params['data'] = current_train_df
            s = pycaret_setup(**setup_params)
            logging.info(f"Separate upload: train_df shape {current_train_df.shape}, test_df shape {current_test_df_for_predict.shape}")

        elif pycaret_split:
            logging.info("PyCaretにデータ分割を任せます。")
            current_train_df = raw_train_df
            setup_params['data'] = current_train_df
            s = pycaret_setup(**setup_params)
            
            X_test_from_pycaret = s.get_config('X_test')
            y_test_from_pycaret = s.get_config('y_test')
            current_test_df_for_predict = pd.concat([X_test_from_pycaret.reset_index(drop=True), y_test_from_pycaret.reset_index(drop=True)], axis=1)
            
            logging.info(f"PyCaret split: Initial data shape {current_train_df.shape}, PyCaret internal test_df shape {current_test_df_for_predict.shape}")

        else:
            logging.info(f"データを手動で分割します。検証比率: {test_split_ratio}")
            if not test_split_ratio or not (0 < float(test_split_ratio) < 1):
                return jsonify({'error': '有効な検証分割比率を指定してください (0から1の間)。'}), 400
            
            X = raw_train_df.drop(columns=[target_column])
            y = raw_train_df[target_column]

            stratify_param = None
            if task_type == 'classification':
                stratify_param = y if (y.dtype == 'object' or y.nunique() <= 10) else None
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=float(test_split_ratio), random_state=123, stratify=stratify_param
            )
            
            current_train_df = pd.concat([X_train, y_train], axis=1)
            current_test_df_for_predict = pd.concat([X_test, y_test], axis=1)

            setup_params['data'] = current_train_df
            s = pycaret_setup(**setup_params)
            logging.info(f"Manual split: train_df shape {current_train_df.shape}, test_df shape {current_test_df_for_predict.shape}")

        # モデル比較の準備
        compare_models_params = {}
        logging.info("compare_modelsを呼び出します。交差検証戦略はsetup()で設定済みです。")

        best_model = None
        model_results = None
        saved_model_filenames = []

        if analysis_mode == 'compare':
            # 従来のモデル比較
            try:
                best_model = pycaret_compare_models(**compare_models_params)
                model_results = pycaret_pull()
            except Exception as e:
                logging.error(f"モデル比較中にエラーが発生しました: {e}", exc_info=True)
                error_message = f"モデル比較中にエラーが発生しました。データセットまたは選択された検証方法を確認してください。詳細: {str(e)}"
                
                if "least populated class" in str(e).lower() and task_type == 'classification':
                    error_message = "データセットの問題です。目標列のいずれかのクラスのデータが少なすぎます（1つしかありません）。そのクラスを含む行を削除するか、他のクラスと統合してください。" + error_message
                
                return jsonify({'error': error_message})

            # ベストモデルの保存
            model_identifier = model_results.iloc[0]['Model']
            safe_model_name = model_identifier.replace(" ", "_").replace(".", "").replace("/", "_").replace("(", "").replace(")", "")
            if safe_model_name.lower().endswith(".pkl"):
                safe_model_name = safe_model_name[:-4]

            single_model_filename_no_ext = f"{safe_model_name}_{analysis_timestamp}"
            single_model_save_path_no_ext = os.path.join(RESULTS_FOLDER, single_model_filename_no_ext)
            
            pycaret_save_model(best_model, single_model_save_path_no_ext)
            actual_saved_filename = single_model_filename_no_ext + ".pkl"
            saved_model_filenames = [actual_saved_filename]
            
            logging.info(f"Best model '{model_identifier}' saved to {os.path.join(RESULTS_FOLDER, actual_saved_filename)}")

        else:
            # 個別モデルの学習とチューニング
            if not selected_model:
                return jsonify({'error': '個別モデル解析では、モデルを選択してください。'}), 400

            try:
                # モデルの作成
                if custom_hyperparameters:
                    # カスタムハイパーパラメータでモデル作成
                    created_model = pycaret_create_model(selected_model, **custom_hyperparameters)
                    logging.info(f"カスタムハイパーパラメータでモデル '{selected_model}' を作成しました。")
                else:
                    # デフォルトハイパーパラメータでモデル作成
                    created_model = pycaret_create_model(selected_model)
                    logging.info(f"デフォルトハイパーパラメータでモデル '{selected_model}' を作成しました。")

                # ハイパーパラメータチューニング
                if hyperparameter_tuning:
                    logging.info("ハイパーパラメータチューニングを実行します。")
                    tuned_model = pycaret_tune_model(created_model, n_iter=10, optimize='accuracy' if task_type == 'classification' else 'r2')
                    best_model = tuned_model
                    logging.info("ハイパーパラメータチューニングが完了しました。")
                else:
                    best_model = created_model

                # モデル評価結果の取得
                model_results = pycaret_pull()

                # モデルの保存
                safe_model_name = selected_model.replace(" ", "_").replace(".", "").replace("/", "_").replace("(", "").replace(")", "")
                tuning_suffix = "_tuned" if hyperparameter_tuning else "_default"
                model_filename_no_ext = f"{safe_model_name}{tuning_suffix}_{analysis_timestamp}"
                model_save_path_no_ext = os.path.join(RESULTS_FOLDER, model_filename_no_ext)
                
                pycaret_save_model(best_model, model_save_path_no_ext)
                actual_saved_filename = model_filename_no_ext + ".pkl"
                saved_model_filenames = [actual_saved_filename]
                
                logging.info(f"Individual model '{selected_model}' saved to {os.path.join(RESULTS_FOLDER, actual_saved_filename)}")

            except Exception as e:
                logging.error(f"個別モデル解析中にエラーが発生しました: {e}", exc_info=True)
                return jsonify({'error': f'個別モデル解析中にエラーが発生しました: {str(e)}'}), 500

        # 結果をHTMLに整形
        results_html = model_results.head(5).to_html(classes='table table-striped', index=False)
        prediction_results_html = ""

        # テストデータでの予測
        if current_test_df_for_predict is not None and not current_test_df_for_predict.empty:
            try:
                test_df_for_prediction_no_target = current_test_df_for_predict.drop(
                    columns=[target_column], errors='ignore'
                )

                # 最新のモデルファイルをロード（拡張子なしのパスでロード）
                loaded_model = pycaret_load_model(single_model_save_path_no_ext if analysis_mode == 'compare' else model_save_path_no_ext)
                predictions = pycaret_predict_model(loaded_model, data=test_df_for_prediction_no_target)

                # 予測結果の表示カラムを決定
                display_cols = []
                if task_type == 'classification':
                    display_cols = ['prediction_label', 'prediction_score']
                else:
                    if 'prediction' in predictions.columns:
                        display_cols = ['prediction']
                    elif 'Label' in predictions.columns:
                        display_cols = ['Label']
                    else:
                        potential_pred_cols = [col for col in predictions.columns if col not in raw_train_df.columns and predictions[col].dtype in ['int64', 'float64']]
                        if potential_pred_cols:
                            display_cols = potential_pred_cols
                        else:
                            logging.warning("No specific prediction column found for regression.")
                            prediction_results_html = predictions.head(10).to_html(classes='table table-striped', index=False)

                if display_cols:
                    existing_cols = [col for col in display_cols if col in predictions.columns]
                    if existing_cols:
                        prediction_results_html = predictions[existing_cols].head(10).to_html(classes='table table-striped', index=False)
                    else:
                        prediction_results_html = predictions.head(10).to_html(classes='table table-striped', index=False)

                logging.info("Prediction on test data completed.")

            except Exception as e:
                logging.error(f"テストデータでの予測エラーが発生しました: {e}", exc_info=True)
                prediction_results_html = f"<p class='text-danger'>テストデータでの予測中にエラーが発生しました: {str(e)}</p>"
        else:
            prediction_results_html = "<p class='text-muted'>テストデータが提供されなかったか、PyCaretの内部分割により評価されたため、別途予測結果は表示されません。</p>"

        return jsonify({
            'success': True,
            'results_html': results_html,
            'prediction_results_html': prediction_results_html,
            'saved_models': saved_model_filenames,
            'analysis_mode': analysis_mode,
            'message': f'PyCaret解析と予測が完了しました。解析モード: {analysis_mode}'
        })

    except Exception as e:
        logging.error(f"PyCaret解析中にエラーが発生しました: {e}", exc_info=True)
        return jsonify({'error': f'PyCaret解析中にエラーが発生しました: {str(e)}'}), 500
    finally:
        # Clean up uploaded files
        if train_filepath and os.path.exists(train_filepath):
            try:
                os.remove(train_filepath)
                logging.info(f"Cleaned up training file: {train_filepath}")
            except Exception as e:
                logging.error(f"Error removing training file {train_filepath}: {e}")
        if original_test_filepath and os.path.exists(original_test_filepath):
            try:
                os.remove(original_test_filepath)
                logging.info(f"Cleaned up test file: {original_test_filepath}")
            except Exception as e:
                logging.error(f"Error removing test file {original_test_filepath}: {e}")

@app.route('/get_available_models', methods=['GET'])
def get_available_models():
    """利用可能なPyCaretモデルのリストを返します。"""
    task_type = request.args.get('task_type', 'classification')
    
    # PyCaretで利用可能なモデルの一覧
    classification_models = {
        'lr': 'Logistic Regression',
        'knn': 'K Neighbors Classifier',
        'nb': 'Naive Bayes',
        'dt': 'Decision Tree Classifier',
        'svm': 'SVM - Linear Kernel',
        'rbfsvm': 'SVM - Radial Kernel',
        'gpc': 'Gaussian Process Classifier',
        'mlp': 'MLP Classifier',
        'ridge': 'Ridge Classifier',
        'rf': 'Random Forest Classifier',
        'qda': 'Quadratic Discriminant Analysis',
        'ada': 'Ada Boost Classifier',
        'gbc': 'Gradient Boosting Classifier',
        'lda': 'Linear Discriminant Analysis',
        'et': 'Extra Trees Classifier',
        'xgboost': 'Extreme Gradient Boosting',
        'lightgbm': 'Light Gradient Boosting Machine',
        'catboost': 'CatBoost Classifier'
    }
    
    regression_models = {
        'lr': 'Linear Regression',
        'lasso': 'Lasso Regression',
        'ridge': 'Ridge Regression',
        'en': 'Elastic Net',
        'lar': 'Least Angle Regression',
        'llar': 'Lasso Least Angle Regression',
        'omp': 'Orthogonal Matching Pursuit',
        'br': 'Bayesian Ridge',
        'ard': 'Automatic Relevance Determination',
        'par': 'Passive Aggressive Regressor',
        'ransac': 'Random Sample Consensus',
        'tr': 'TheilSen Regressor',
        'huber': 'Huber Regressor',
        'kr': 'Kernel Ridge',
        'svm': 'Support Vector Regression',
        'knn': 'K Neighbors Regressor',
        'dt': 'Decision Tree Regressor',
        'rf': 'Random Forest Regressor',
        'et': 'Extra Trees Regressor',
        'ada': 'AdaBoost Regressor',
        'gbr': 'Gradient Boosting Regressor',
        'mlp': 'MLP Regressor',
        'xgboost': 'Extreme Gradient Boosting',
        'lightgbm': 'Light Gradient Boosting Machine',
        'catboost': 'CatBoost Regressor'
    }
    
    if task_type == 'classification':
        return jsonify({'success': True, 'models': classification_models})
    else:
        return jsonify({'success': True, 'models': regression_models})

@app.route('/download_model/<filename>', methods=['GET'])
def download_model(filename):
    """指定されたモデルファイルをダウンロードさせます。"""
    logging.info(f"Download request for model: {filename}")
    try:
        return send_from_directory(app.config['RESULTS_FOLDER'], filename, as_attachment=True)
    except FileNotFoundError:
        logging.error(f"Model file not found: {filename}")
        return "Model not found.", 404
    except Exception as e:
        logging.error(f"Error serving model file {filename}: {e}")
        return "An error occurred during download.", 500

if __name__ == '__main__':
    app.run(debug=True)