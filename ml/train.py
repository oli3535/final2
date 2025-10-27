import os
import pandas as pd
import numpy as np
import pickle
import mlflow
import mlflow.sklearn
import pytest
import subprocess
import sys
from datetime import datetime
from dotenv import load_dotenv
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load environment variables first
load_dotenv()

class StudentGradePredictor:
    def __init__(self, model_path=None):
        # Load configuration from environment variables
        self.model_name = os.getenv('MODEL_NAME', 'student_grade_predictor')
        
        # 先初始化 debug 标志
        self.debug = os.getenv('DEBUG', 'False').lower() == 'true'
        
        # 然后使用修复路径的方法
        self.dataset_path = self._get_absolute_path(os.getenv('DATASET_PATH', 'data/student_data.csv'))
        
        self.model = None
        
        # Grade mappings
        self.grade_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        self.reverse_mapping = {v: k for k, v in self.grade_mapping.items()}
        
        # Set MLflow tracking URI - 使用正确的文件URI格式
        mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'file:./mlruns')
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        
        if self.debug:
            print(f"MLflow tracking URI: {mlflow_tracking_uri}")
        
        # Feature names
        self.feature_names = ['study_hours', 'attendance', 'participation', 'assignment_score']

    def _get_absolute_path(self, relative_path, debug=None):
        """Convert relative path to absolute path from project root"""
        # 获取当前文件的目录 (ml/)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 回到项目根目录
        project_root = os.path.dirname(current_dir)
        # 构建绝对路径
        absolute_path = os.path.join(project_root, relative_path)
        
        # 使用传入的debug参数或实例的debug属性
        debug_flag = debug if debug is not None else getattr(self, 'debug', False)
        if debug_flag:
            print(f"Resolved path: {relative_path} -> {absolute_path}")
            
        return absolute_path

    def get_git_commit_sha(self):
        """Get current git commit SHA for code versioning"""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "unknown"

    def get_dvc_dataset_hash(self):
        """Get DVC dataset version hash"""
        try:
            # Simulate getting DVC hash - in real scenario, use DVC API
            dataset_hash = f"dvc_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            return dataset_hash
        except Exception:
            return "dvc_not_configured"

    def create_confusion_matrix_plot(self, y_true, y_pred, run_id):
        """Create and save confusion matrix plot"""
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['A', 'B', 'C', 'D'],
                   yticklabels=['A', 'B', 'C', 'D'])
        plt.title('Confusion Matrix - Student Grade Prediction')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save plot with absolute path
        plot_path = self._get_absolute_path(f"ml/registry/confusion_matrix_{run_id}.png")
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        return plot_path

    def create_feature_importance_plot(self, feature_importances, run_id):
        """Create and save feature importance plot"""
        plt.figure(figsize=(10, 6))
        indices = np.argsort(feature_importances)[::-1]
        
        plt.bar(range(len(feature_importances)), feature_importances[indices])
        plt.xticks(range(len(feature_importances)), [self.feature_names[i] for i in indices], rotation=45)
        plt.title('Feature Importance - Student Grade Prediction')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.tight_layout()
        
        # Save plot with absolute path
        plot_path = self._get_absolute_path(f"ml/registry/feature_importance_{run_id}.png")
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        return plot_path

    def load_data(self):
        """Load data from path specified in environment variables"""
        try:
            if self.debug:
                print(f"Attempting to load data from: {self.dataset_path}")
                print(f"File exists: {os.path.exists(self.dataset_path)}")
            
            df = pd.read_csv(self.dataset_path)
            
            if self.debug:
                print(f"Data loaded successfully: {df.shape}")
                print(f"Columns: {df.columns.tolist()}")
                print(f"First few rows:\n{df.head()}")
                
            return df
        except Exception as e:
            print(f"Error loading data from {self.dataset_path}: {str(e)}")
            # 列出目录内容来调试
            data_dir = os.path.dirname(self.dataset_path)
            if os.path.exists(data_dir):
                print(f"Files in {data_dir}: {os.listdir(data_dir)}")
            raise Exception(f"Failed to load data from {self.dataset_path}: {str(e)}")

    def preprocess_data(self, df):
        """Preprocess data for training"""
        # 检查必需的列是否存在
        required_columns = ['study_hours', 'attendance', 'participation', 'assignment_score', 'grade']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        X = df[self.feature_names].values
        y = df['grade'].map(self.grade_mapping).values
        return X, y

    def train_baseline_model(self):
        """
        Baseline model experiment
        Simple Decision Tree with default parameters
        """
        print("Training Baseline Model...")
        
        df = self.load_data()
        X, y = self.preprocess_data(df)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # MLflow experiment tracking
        mlflow.set_experiment("student_grade_prediction")
        
        with mlflow.start_run(run_name="baseline_decision_tree") as run:
            run_id = run.info.run_id
            
            if self.debug:
                print(f"MLflow Run ID: {run_id}")
            
            # 1. CODE VERSION: Git commit SHA
            git_commit_sha = self.get_git_commit_sha()
            mlflow.set_tag("git_commit", git_commit_sha)
            mlflow.log_param("code_version", git_commit_sha)
            print(f"Logged code version: {git_commit_sha}")
            
            # 2. DATASET VERSION: DVC dataset hash
            dataset_hash = self.get_dvc_dataset_hash()
            mlflow.log_param("dataset_version", dataset_hash)
            mlflow.set_tag("dataset_hash", dataset_hash)
            print(f"Logged dataset version: {dataset_hash}")
            
            # 3. HYPERPARAMETERS - Baseline
            hyperparams = {
                "model_type": "DecisionTreeClassifier",
                "test_size": 0.2,
                "random_state": 42,
                "max_depth": 3,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "criterion": "gini",
                "experiment_type": "baseline"
            }
            
            for param, value in hyperparams.items():
                mlflow.log_param(param, value)
                print(f"Logged parameter: {param} = {value}")
            
            # 4. MODEL TRAINING - Baseline
            self.model = DecisionTreeClassifier(
                max_depth=hyperparams["max_depth"],
                random_state=hyperparams["random_state"],
                min_samples_split=hyperparams["min_samples_split"],
                min_samples_leaf=hyperparams["min_samples_leaf"]
            )
            
            self.model.fit(X_train, y_train)
            
            # 5. METRICS
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("train_samples", len(X_train))
            mlflow.log_metric("test_samples", len(X_test))
            mlflow.log_metric("baseline_score", accuracy)
            
            print(f"Logged metrics - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
            
            # 6. ARTIFACTS - 使用简单的日志记录，避免模型注册表
            # Log model using sklearn flavor (without registry)
            mlflow.sklearn.log_model(
                self.model, 
                "model",
                registered_model_name=None  # 不注册到模型注册表
            )
            print("Logged model artifact")
            
            # Log plots
            cm_plot_path = self.create_confusion_matrix_plot(y_test, y_pred, run_id)
            mlflow.log_artifact(cm_plot_path)
            print("Logged confusion matrix artifact")
            
            feature_importance_plot_path = self.create_feature_importance_plot(
                self.model.feature_importances_, run_id
            )
            mlflow.log_artifact(feature_importance_plot_path)
            print("Logged feature importance artifact")
            
            # Save model locally
            model_save_path = self._get_absolute_path(f"ml/registry/baseline_model_{run_id}.pkl")
            self.save_model(model_save_path)
            
            return accuracy

    def train_improved_model(self):
        """
        Improved model experiment
        Random Forest with hyperparameter tuning
        """
        print("Training Improved Model...")
        
        df = self.load_data()
        X, y = self.preprocess_data(df)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        with mlflow.start_run(run_name="improved_random_forest") as run:
            run_id = run.info.run_id
            
            if self.debug:
                print(f"MLflow Run ID: {run_id}")
            
            # 1. CODE VERSION: Git commit SHA
            git_commit_sha = self.get_git_commit_sha()
            mlflow.set_tag("git_commit", git_commit_sha)
            mlflow.log_param("code_version", git_commit_sha)
            print(f"Logged code version: {git_commit_sha}")
            
            # 2. DATASET VERSION: DVC dataset hash
            dataset_hash = self.get_dvc_dataset_hash()
            mlflow.log_param("dataset_version", dataset_hash)
            mlflow.set_tag("dataset_hash", dataset_hash)
            print(f"Logged dataset version: {dataset_hash}")
            
            # 3. HYPERPARAMETERS - Improved
            hyperparams = {
                "model_type": "RandomForestClassifier",
                "test_size": 0.2,
                "random_state": 42,
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "bootstrap": True,
                "criterion": "gini",
                "experiment_type": "improved"
            }
            
            for param, value in hyperparams.items():
                mlflow.log_param(param, value)
                print(f"Logged parameter: {param} = {value}")
            
            # 4. MODEL TRAINING - Improved
            self.model = RandomForestClassifier(
                n_estimators=hyperparams["n_estimators"],
                max_depth=hyperparams["max_depth"],
                random_state=hyperparams["random_state"],
                min_samples_split=hyperparams["min_samples_split"],
                min_samples_leaf=hyperparams["min_samples_leaf"],
                bootstrap=hyperparams["bootstrap"]
            )
            
            self.model.fit(X_train, y_train)
            
            # 5. METRICS - Improved
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("train_samples", len(X_train))
            mlflow.log_metric("test_samples", len(X_test))
            mlflow.log_metric("improved_score", accuracy)
            
            print(f"Logged metrics - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
            
            # 6. ARTIFACTS - 使用简单的日志记录，避免模型注册表
            # Log model using sklearn flavor (without registry)
            mlflow.sklearn.log_model(
                self.model, 
                "model",
                registered_model_name=None  # 不注册到模型注册表
            )
            print("Logged model artifact")
            
            # Log plots
            cm_plot_path = self.create_confusion_matrix_plot(y_test, y_pred, run_id)
            mlflow.log_artifact(cm_plot_path)
            print("Logged confusion matrix artifact")
            
            feature_importance_plot_path = self.create_feature_importance_plot(
                self.model.feature_importances_, run_id
            )
            mlflow.log_artifact(feature_importance_plot_path)
            print("Logged feature importance artifact")
            
            # Save model locally
            model_save_path = self._get_absolute_path(f"ml/registry/improved_model_{run_id}.pkl")
            self.save_model(model_save_path)
            
            return accuracy

    def save_model(self, model_path):
        """Save model to file"""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        if self.debug:
            print(f"Model saved to {model_path}")

    def load_model(self, model_path):
        """Load model from file"""
        absolute_path = self._get_absolute_path(model_path)
        if os.path.exists(absolute_path):
            with open(absolute_path, 'rb') as f:
                self.model = pickle.load(f)
            return True
        return False

    def predict(self, features):
        """Predict grade for given features"""
        if self.model is None:
            raise ValueError("Model not initialized")
        
        if len(features) != 4:
            raise ValueError(f"Expected 4 features, got {len(features)}")
        
        features_array = np.array(features).reshape(1, -1)
        prediction_encoded = self.model.predict(features_array)[0]
        return self.reverse_mapping[prediction_encoded]


# Training Execution Functions
def run_baseline_experiment():
    """Run baseline model experiment"""
    print("=" * 60)
    print("STARTING BASELINE MODEL EXPERIMENT")
    print("=" * 60)
    
    predictor = StudentGradePredictor()
    accuracy = predictor.train_baseline_model()
    
    print("=" * 60)
    print(f"BASELINE EXPERIMENT COMPLETED")
    print(f"Baseline Accuracy: {accuracy:.4f}")
    print("=" * 60)
    
    return accuracy

def run_improved_experiment():
    """Run improved model experiment"""
    print("=" * 60)
    print("STARTING IMPROVED MODEL EXPERIMENT")
    print("=" * 60)
    
    predictor = StudentGradePredictor()
    accuracy = predictor.train_improved_model()
    
    print("=" * 60)
    print(f"IMPROVED EXPERIMENT COMPLETED")
    print(f"Improved Accuracy: {accuracy:.4f}")
    print("=" * 60)
    
    return accuracy

def run_all_experiments():
    """Run both baseline and improved experiments"""
    print("STARTING ALL ML EXPERIMENTS")
    print("=" * 60)
    
    baseline_accuracy = run_baseline_experiment()
    improved_accuracy = run_improved_experiment()
    
    print("=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"Baseline Model Accuracy: {baseline_accuracy:.4f}")
    print(f"Improved Model Accuracy: {improved_accuracy:.4f}")
    print(f"Improvement: {improved_accuracy - baseline_accuracy:+.4f}")
    print("=" * 60)
    
    return baseline_accuracy, improved_accuracy

if __name__ == '__main__':
    # 运行实验
    try:
        baseline_acc, improved_acc = run_all_experiments()
        print("✅ All experiments completed successfully!")
        
        # 显示MLflow UI信息
        print("\n📊 MLflow Experiment Tracking:")
        print(f"To view experiments, run: mlflow ui --backend-store-uri file:./mlruns")
        print("Then open http://localhost:5000 in your browser")
        
    except Exception as e:
        print(f"❌ Experiment failed: {str(e)}")
        # 打印详细的错误信息
        import traceback
        traceback.print_exc()