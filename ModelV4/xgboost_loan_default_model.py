import numpy as np
import pandas as pd
import pickle
import joblib
from datetime import datetime
from typing import Tuple, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# XGBoost and ML libraries
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve, accuracy_score,
    f1_score, precision_score, recall_score
)

# Visualization and feature importance
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

class LoanDefaultPredictor:
    """
    XGBoost-based loan default prediction model with comprehensive preprocessing,
    training, evaluation, and interpretability features.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.is_trained = False
        
    def load_and_preprocess_data(self, filepath: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load and preprocess the loan default dataset.
        
        Args:
            filepath (str): Path to the CSV file
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Preprocessed features and target
        """
        try:
            df = pd.read_csv(filepath)
            print(f"Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset not found at {filepath}")
        
        # Basic data info
        print(f"Target distribution:")
        print(df['default_ind'].value_counts(normalize=True))
        print(f"Default rate: {df['default_ind'].mean():.2%}")
        
        # Handle missing values
        print(f"\nMissing values before preprocessing:")
        missing_counts = df.isnull().sum()
        print(missing_counts[missing_counts > 0])
        
        # Drop customer_id as it's just an identifier
        df = df.drop('customer_id', axis=1)
        
        # Handle months_since_last_delinquency - fill NaN with median
        df['months_since_last_delinquency'] = df['months_since_last_delinquency'].fillna(
            df['months_since_last_delinquency'].median()
        )
        
        # Feature engineering
        df = self._feature_engineering(df)
        
        # Separate features and target
        X = df.drop('default_ind', axis=1)
        y = df['default_ind']
        
        # Encode categorical variables
        X = self._encode_categorical_features(X)
        
        self.feature_names = X.columns.tolist()
        print(f"\nFinal feature set: {len(self.feature_names)} features")
        
        return X, y
    
    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional features from existing data.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with engineered features
        """
        df = df.copy()
        
        # Age groups
        df['age_group'] = pd.cut(df['age'], bins=[0, 30, 45, 60, 100], 
                                labels=['Young', 'Middle', 'Senior', 'Elder'])
        
        # Income to loan ratio
        df['income_to_loan_ratio'] = df['annual_income'] / df['loan_amount']
        
        # Credit utilization proxy
        df['credit_density'] = df['num_open_credit_lines'] / (df['credit_score'] / 100)
        
        # Payment burden
        df['monthly_payment_burden'] = (df['loan_amount'] / df['term_months']) / (df['annual_income'] / 12)
        
        # Risk score combination
        df['risk_score'] = (df['debt_to_income'] * 0.4 + 
                           (1 - df['credit_score']/850) * 0.3 + 
                           df['num_derogatory_marks']/10 * 0.3)
        
        # Loan to income category
        loan_to_income = df['loan_amount'] / df['annual_income']
        df['loan_size_category'] = pd.cut(loan_to_income, 
                                        bins=[0, 0.1, 0.3, 0.5, float('inf')],
                                        labels=['Small', 'Medium', 'Large', 'Very_Large'])
        
        # Convert issue_date to datetime and extract features
        df['issue_date'] = pd.to_datetime(df['issue_date'])
        df['issue_year'] = df['issue_date'].dt.year
        df['issue_month'] = df['issue_date'].dt.month
        df['issue_quarter'] = df['issue_date'].dt.quarter
        
        # Drop the original date column
        df = df.drop('issue_date', axis=1)
        
        return df
    
    def _encode_categorical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features using label encoding.
        
        Args:
            X (pd.DataFrame): Features dataframe
            
        Returns:
            pd.DataFrame: Encoded features
        """
        X = X.copy()
        categorical_columns = X.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
            else:
                X[col] = self.label_encoders[col].transform(X[col].astype(str))
        
        return X
    
    def train_model(self, X: pd.DataFrame, y: pd.Series, 
                   test_size: float = 0.2, val_size: float = 0.2,
                   hyperparameter_tuning: bool = True) -> Dict[str, Any]:
        """
        Train the XGBoost model with optional hyperparameter tuning.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target variable
            test_size (float): Test set proportion
            val_size (float): Validation set proportion
            hyperparameter_tuning (bool): Whether to perform hyperparameter tuning
            
        Returns:
            Dict[str, Any]: Training results and metrics
        """
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(test_size + val_size), 
            random_state=self.random_state, stratify=y
        )
        
        val_test_ratio = test_size / (test_size + val_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=val_test_ratio,
            random_state=self.random_state, stratify=y_temp
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples") 
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert to DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train_scaled, label=y_train, feature_names=self.feature_names)
        dval = xgb.DMatrix(X_val_scaled, label=y_val, feature_names=self.feature_names)
        dtest = xgb.DMatrix(X_test_scaled, label=y_test, feature_names=self.feature_names)
        
        if hyperparameter_tuning:
            print("Performing hyperparameter tuning...")
            best_params = self._hyperparameter_tuning(X_train_scaled, y_train)
        else:
            best_params = {
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0
            }
        
        # Configure XGBoost parameters
        params = {
            'objective': 'binary:logistic',
            'eval_metric': ['auc', 'logloss'],
            'max_depth': best_params['max_depth'],
            'learning_rate': best_params['learning_rate'],
            'subsample': best_params['subsample'],
            'colsample_bytree': best_params['colsample_bytree'],
            'reg_alpha': best_params['reg_alpha'],
            'reg_lambda': best_params['reg_lambda'],
            'random_state': self.random_state,
            'verbosity': 1
        }
        
        # Train model with early stopping
        print("Training XGBoost model...")
        evallist = [(dtrain, 'train'), (dval, 'validation')]
        
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=best_params.get('n_estimators', 1000),
            evals=evallist,
            early_stopping_rounds=50,
            verbose_eval=False
        )
        
        self.is_trained = True
        
        # Generate predictions
        train_preds = self.model.predict(dtrain)
        val_preds = self.model.predict(dval)  
        test_preds = self.model.predict(dtest)
        
        # Calculate metrics
        results = {
            'train_auc': roc_auc_score(y_train, train_preds),
            'val_auc': roc_auc_score(y_val, val_preds),
            'test_auc': roc_auc_score(y_test, test_preds),
            'test_accuracy': accuracy_score(y_test, (test_preds > 0.5).astype(int)),
            'test_precision': precision_score(y_test, (test_preds > 0.5).astype(int)),
            'test_recall': recall_score(y_test, (test_preds > 0.5).astype(int)),
            'test_f1': f1_score(y_test, (test_preds > 0.5).astype(int)),
            'X_test': X_test_scaled,
            'y_test': y_test,
            'test_predictions': test_preds,
            'best_params': best_params
        }
        
        print(f"\nModel Training Complete!")
        print(f"Train AUC: {results['train_auc']:.4f}")
        print(f"Validation AUC: {results['val_auc']:.4f}")
        print(f"Test AUC: {results['test_auc']:.4f}")
        print(f"Test Accuracy: {results['test_accuracy']:.4f}")
        
        return results
    
    def _hyperparameter_tuning(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Perform grid search for hyperparameter tuning.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training target
            
        Returns:
            Dict[str, Any]: Best parameters
        """
        # Define parameter grid
        param_grid = {
            'max_depth': [4, 6, 8],
            'learning_rate': [0.05, 0.1, 0.2],
            'n_estimators': [100, 200],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9],
            'reg_alpha': [0, 0.1],
            'reg_lambda': [1, 1.5]
        }
        
        # Create XGBoost classifier
        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            random_state=self.random_state,
            verbosity=0
        )
        
        # Perform grid search
        grid_search = GridSearchCV(
            xgb_model,
            param_grid,
            cv=3,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation AUC: {grid_search.best_score_:.4f}")
        
        return grid_search.best_params_
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray, 
                      test_predictions: np.ndarray) -> None:
        """
        Generate comprehensive model evaluation metrics and visualizations.
        
        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test target
            test_predictions (np.ndarray): Test predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        # Binary predictions
        y_pred_binary = (test_predictions > 0.5).astype(int)
        
        # Calculate metrics
        auc_score = roc_auc_score(y_test, test_predictions)
        accuracy = accuracy_score(y_test, y_pred_binary)
        precision = precision_score(y_test, y_pred_binary)
        recall = recall_score(y_test, y_pred_binary)
        f1 = f1_score(y_test, y_pred_binary)
        
        print("\n" + "="*60)
        print("MODEL EVALUATION RESULTS")
        print("="*60)
        print(f"ROC-AUC Score: {auc_score:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print("="*60)
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred_binary)
        print("\nConfusion Matrix:")
        print(f"{'':>15} {'Predicted 0':>12} {'Predicted 1':>12}")
        print(f"{'Actual 0':>15} {cm[0,0]:>12} {cm[0,1]:>12}")
        print(f"{'Actual 1':>15} {cm[1,0]:>12} {cm[1,1]:>12}")
        
        # Classification Report
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred_binary))
        
        # Create visualizations
        self._create_evaluation_plots(y_test, test_predictions, y_pred_binary)
    
    def _create_evaluation_plots(self, y_test: np.ndarray, y_pred_proba: np.ndarray, 
                               y_pred_binary: np.ndarray) -> None:
        """
        Create evaluation plots for model performance.
        
        Args:
            y_test (np.ndarray): True labels
            y_pred_proba (np.ndarray): Predicted probabilities
            y_pred_binary (np.ndarray): Binary predictions
        """
        plt.figure(figsize=(15, 12))
        
        # ROC Curve
        plt.subplot(2, 3, 1)
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        
        # Precision-Recall Curve
        plt.subplot(2, 3, 2)
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        plt.plot(recall, precision)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True)
        
        # Confusion Matrix Heatmap
        plt.subplot(2, 3, 3)
        cm = confusion_matrix(y_test, y_pred_binary)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        # Feature Importance
        plt.subplot(2, 3, 4)
        importance = self.model.get_score(importance_type='weight')
        features = list(importance.keys())
        scores = list(importance.values())
        
        # Get top 15 features
        top_indices = np.argsort(scores)[-15:]
        top_features = [features[i] for i in top_indices]
        top_scores = [scores[i] for i in top_indices]
        
        plt.barh(range(len(top_features)), top_scores)
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('Importance Score')
        plt.title('Top 15 Feature Importance')
        
        # Prediction Distribution
        plt.subplot(2, 3, 5)
        plt.hist(y_pred_proba[y_test == 0], bins=50, alpha=0.7, label='Non-Default', color='blue')
        plt.hist(y_pred_proba[y_test == 1], bins=50, alpha=0.7, label='Default', color='red')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Frequency')
        plt.title('Prediction Distribution')
        plt.legend()
        
        # Calibration Plot
        plt.subplot(2, 3, 6)
        from sklearn.calibration import calibration_curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_test, y_pred_proba, n_bins=10
        )
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="XGBoost")
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Plot')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('/Users/yonatanohayon/Desktop/Projects/Models/ModelV4/model_evaluation_plots.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def feature_importance_analysis(self) -> None:
        """
        Generate detailed feature importance analysis using multiple methods.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before feature importance analysis")
        
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*60)
        
        # Get feature importance from XGBoost
        importance_weight = self.model.get_score(importance_type='weight')
        importance_gain = self.model.get_score(importance_type='gain')
        importance_cover = self.model.get_score(importance_type='cover')
        
        # Create DataFrame for easier handling
        importance_df = pd.DataFrame({
            'feature': list(importance_weight.keys()),
            'weight': list(importance_weight.values()),
            'gain': [importance_gain.get(f, 0) for f in importance_weight.keys()],
            'cover': [importance_cover.get(f, 0) for f in importance_weight.keys()]
        })
        
        importance_df = importance_df.sort_values('gain', ascending=False)
        
        print("\nTop 20 Most Important Features (by Gain):")
        print("-" * 50)
        for i, (_, row) in enumerate(importance_df.head(20).iterrows(), 1):
            print(f"{i:2d}. {row['feature']:25} - Gain: {row['gain']:8.2f}")
        
        # Create feature importance visualization
        plt.figure(figsize=(12, 8))
        
        # Plot top 20 features by gain
        top_20 = importance_df.head(20)
        plt.barh(range(len(top_20)), top_20['gain'])
        plt.yticks(range(len(top_20)), top_20['feature'])
        plt.xlabel('Importance (Gain)')
        plt.title('Top 20 Feature Importance - XGBoost (Gain)')
        plt.gca().invert_yaxis()
        
        plt.tight_layout()
        plt.savefig('/Users/yonatanohayon/Desktop/Projects/Models/ModelV4/feature_importance.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return importance_df
    
    def shap_analysis(self, X_sample: np.ndarray, sample_size: int = 1000) -> None:
        """
        Perform SHAP analysis for model interpretability.
        
        Args:
            X_sample (np.ndarray): Sample of features for SHAP analysis
            sample_size (int): Number of samples to use for SHAP analysis
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before SHAP analysis")
        
        print("\n" + "="*60)
        print("SHAP ANALYSIS - MODEL INTERPRETABILITY")
        print("="*60)
        
        try:
            # Create SHAP explainer
            explainer = shap.TreeExplainer(self.model)
            
            # Use a subset for faster computation
            X_shap = X_sample[:sample_size] if len(X_sample) > sample_size else X_sample
            shap_values = explainer.shap_values(X_shap)
            
            print(f"SHAP analysis completed for {len(X_shap)} samples")
            
            # Create SHAP plots
            plt.figure(figsize=(15, 10))
            
            # Summary plot
            plt.subplot(2, 2, 1)
            shap.summary_plot(shap_values, X_shap, feature_names=self.feature_names, 
                            show=False, max_display=20)
            plt.title('SHAP Summary Plot')
            
            # Feature importance
            plt.subplot(2, 2, 2)
            shap.summary_plot(shap_values, X_shap, feature_names=self.feature_names,
                            plot_type="bar", show=False, max_display=20)
            plt.title('SHAP Feature Importance')
            
            plt.tight_layout()
            plt.savefig('/Users/yonatanohayon/Desktop/Projects/Models/ModelV4/shap_analysis.png', 
                       dpi=300, bbox_inches='tight')
            plt.show()
            
            return shap_values
            
        except Exception as e:
            print(f"SHAP analysis failed: {str(e)}")
            print("Continuing without SHAP analysis...")
            return None
    
    def save_model(self, filepath: str = None) -> str:
        """
        Save the trained model and preprocessing components.
        
        Args:
            filepath (str): Optional custom filepath
            
        Returns:
            str: Path where model was saved
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = f'/Users/yonatanohayon/Desktop/Projects/Models/ModelV4/xgboost_loan_model_{timestamp}.pkl'
        
        # Save model and preprocessing components
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'random_state': self.random_state
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to: {filepath}")
        return filepath
    
    def load_model(self, filepath: str) -> None:
        """
        Load a previously trained model.
        
        Args:
            filepath (str): Path to the saved model
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_names = model_data['feature_names']
        self.random_state = model_data['random_state']
        self.is_trained = True
        
        print(f"Model loaded from: {filepath}")
    
    def predict_single_loan(self, loan_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict default probability for a single loan application.
        
        Args:
            loan_data (Dict[str, Any]): Loan application data
            
        Returns:
            Dict[str, Any]: Prediction results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Convert to DataFrame
        df = pd.DataFrame([loan_data])
        
        # Apply the same preprocessing as training data
        df = self._feature_engineering(df)
        df = self._encode_categorical_features(df)
        
        # Ensure all required features are present
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0  # Default value for missing features
        
        # Reorder columns to match training data
        df = df[self.feature_names]
        
        # Scale features
        X_scaled = self.scaler.transform(df)
        
        # Create DMatrix and predict
        dmatrix = xgb.DMatrix(X_scaled, feature_names=self.feature_names)
        probability = self.model.predict(dmatrix)[0]
        
        # Determine risk level
        if probability < 0.3:
            risk_level = "Low Risk"
        elif probability < 0.6:
            risk_level = "Medium Risk"
        else:
            risk_level = "High Risk"
        
        results = {
            'default_probability': float(probability),
            'risk_level': risk_level,
            'recommendation': 'Approve' if probability < 0.5 else 'Reject'
        }
        
        return results


def main():
    """
    Main function to train and evaluate the XGBoost loan default prediction model.
    """
    print("XGBoost Loan Default Prediction Model")
    print("="*60)
    
    # Initialize the predictor
    predictor = LoanDefaultPredictor(random_state=RANDOM_STATE)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X, y = predictor.load_and_preprocess_data('synthetic_loan_data_1m.csv')
    
    # Train the model
    print("\nTraining XGBoost model...")
    results = predictor.train_model(X, y, hyperparameter_tuning=True)
    
    # Evaluate the model
    print("\nEvaluating model performance...")
    predictor.evaluate_model(results['X_test'], results['y_test'], results['test_predictions'])
    
    # Feature importance analysis
    importance_df = predictor.feature_importance_analysis()
    
    # SHAP analysis
    predictor.shap_analysis(results['X_test'])
    
    # Save the model
    model_path = predictor.save_model()
    
    # Print final summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Best Parameters: {results['best_params']}")
    print(f"Final Test AUC: {results['test_auc']:.4f}")
    print(f"Final Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"Model saved to: {model_path}")
    print("="*60)
    
    return predictor, results


if __name__ == "__main__":
    predictor, results = main()