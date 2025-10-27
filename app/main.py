import os
import sys
import json
import logging
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Add project root to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ml.train import StudentGradePredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StudentGradeAPI:
    """Simple command-line API for student grade prediction"""
    
    def __init__(self):
        self.predictor = StudentGradePredictor()
        self.model_loaded = False
        self.load_model()
    
    def load_model(self):
        """Load or train the model"""
        try:
            # Try to load existing model first
            if self.predictor.load_model():
                logger.info("Model loaded successfully")
                self.model_loaded = True
                return True
            else:
                # Train a new model if none exists
                logger.info("No trained model found. Training new model...")
                accuracy = self.predictor.train_baseline_model()
                logger.info(f"Model trained with accuracy: {accuracy:.3f}")
                self.model_loaded = True
                return True
        except Exception as e:
            logger.error(f"Failed to load/train model: {str(e)}")
            return False
    
    def health_check(self):
        """Health check response"""
        return {
            "status": "healthy" if self.model_loaded else "unhealthy",
            "model": os.getenv('MODEL_NAME', 'student_grade_predictor'),
            "environment": os.getenv('TEST_MODE', 'development'),
            "model_loaded": self.model_loaded
        }
    
    def predict_grade(self, features):
        """
        Predict student grade based on features
        
        Args:
            features: List of 4 features [study_hours, attendance, participation, assignment_score]
        
        Returns:
            dict: Prediction result with status_code field
        """
        try:
            # Input validation
            if not features or not isinstance(features, list):
                return {
                    "error": "Features must be a list",
                    "status_code": 400
                }
            
            if len(features) != 4:
                return {
                    "error": f"Need exactly 4 features, got {len(features)}",
                    "status_code": 400
                }
            
            # Feature range validation
            for i, feature in enumerate(features):
                if not isinstance(feature, (int, float)):
                    return {
                        "error": f"Feature {i+1} must be a number",
                        "status_code": 400
                    }
                
                if feature < 0 or feature > 2:
                    return {
                        "error": f"Feature {i+1} must be between 0-2",
                        "status_code": 400
                    }
            
            if not self.model_loaded:
                return {
                    "error": "Model not available. Please train the model first.",
                    "status_code": 500
                }
            
            # Make prediction
            prediction = self.predictor.predict(features)
            
            return {
                "prediction": prediction,
                "features": features,
                "model": os.getenv('MODEL_NAME'),
                "status": "success",
                "status_code": 200
            }
        
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return {
                "error": f"Prediction failed: {str(e)}",
                "status_code": 500
            }
    
    def batch_predict(self, samples):
        """
        Batch prediction for multiple samples
        
        Args:
            samples: List of feature lists
        
        Returns:
            dict: Batch prediction results with status_code field
        """
        try:
            if not samples or not isinstance(samples, list):
                return {
                    "error": "Samples must be a non-empty list",
                    "status_code": 400
                }
            
            if len(samples) == 0:
                return {
                    "error": "Samples list cannot be empty",
                    "status_code": 400
                }
            
            if not self.model_loaded:
                return {
                    "error": "Model not available",
                    "status_code": 500
                }
            
            predictions = []
            for i, sample in enumerate(samples):
                if len(sample) != 4:
                    return {
                        "error": f"Sample {i} must have 4 features",
                        "status_code": 400
                    }
                
                for j, feature in enumerate(sample):
                    if not isinstance(feature, (int, float)) or feature < 0 or feature > 2:
                        return {
                            "error": f"Sample {i}, feature {j+1} must be a number between 0-2",
                            "status_code": 400
                        }
                
                prediction = self.predictor.predict(sample)
                predictions.append({
                    "sample_index": i,
                    "prediction": prediction,
                    "features": sample
                })
            
            return {
                "predictions": predictions,
                "total_samples": len(samples),
                "status": "success",
                "status_code": 200
            }
        
        except Exception as e:
            logger.error(f"Batch prediction error: {str(e)}")
            return {
                "error": f"Batch prediction failed: {str(e)}",
                "status_code": 500
            }
    
    def model_info(self):
        """Get model information"""
        return {
            "model_name": os.getenv('MODEL_NAME'),
            "dataset_path": os.getenv('DATASET_PATH'),
            "environment": os.getenv('TEST_MODE'),
            "model_loaded": self.model_loaded
        }

# Global API instance
api = StudentGradeAPI()

# ============================================================================
# Command Line Interface
# ============================================================================

def print_menu():
    """Print command line menu"""
    print("\n" + "="*50)
    print("STUDENT GRADE PREDICTION API")
    print("="*50)
    print("1. Health Check")
    print("2. Predict Grade")
    print("3. Batch Predict")
    print("4. Model Info")
    print("5. Run Tests")
    print("6. Exit")
    print("="*50)

def get_features_input():
    """Get features input from user"""
    try:
        print("\nEnter 4 features (0-2):")
        features = []
        feature_names = ['study_hours', 'attendance', 'participation', 'assignment_score']
        
        for i, name in enumerate(feature_names, 1):
            while True:
                try:
                    value = float(input(f"{i}. {name}: "))
                    if 0 <= value <= 2:
                        features.append(value)
                        break
                    else:
                        print("Value must be between 0 and 2")
                except ValueError:
                    print("Please enter a valid number")
        
        return features
    except KeyboardInterrupt:
        return None

def get_batch_samples_input():
    """Get batch samples input from user"""
    try:
        samples = []
        print("\nEnter number of samples to predict:")
        num_samples = int(input("Number of samples: "))
        
        for i in range(num_samples):
            print(f"\nSample {i+1}:")
            features = get_features_input()
            if features is None:
                return None
            samples.append(features)
        
        return samples
    except (ValueError, KeyboardInterrupt):
        return None

def main():
    """Main command line interface"""
    print("Starting Student Grade Prediction API...")
    
    if not api.model_loaded:
        print("❌ Failed to initialize model. Please check the logs.")
        return
    
    print("✅ Model loaded successfully!")
    
    while True:
        print_menu()
        
        try:
            choice = input("\nSelect option (1-6): ").strip()
            
            if choice == '1':
                # Health Check
                result = api.health_check()
                print(f"\nHealth Check: {json.dumps(result, indent=2)}")
            
            elif choice == '2':
                # Predict Grade
                features = get_features_input()
                if features is None:
                    continue
                
                result = api.predict_grade(features)
                print(f"\nPrediction Result: {json.dumps(result, indent=2)}")
            
            elif choice == '3':
                # Batch Predict
                samples = get_batch_samples_input()
                if samples is None:
                    continue
                
                result = api.batch_predict(samples)
                print(f"\nBatch Prediction Result: {json.dumps(result, indent=2)}")
            
            elif choice == '4':
                # Model Info
                result = api.model_info()
                print(f"\nModel Info: {json.dumps(result, indent=2)}")
            
            elif choice == '5':
                # Run Tests
                run_api_tests()
            
            elif choice == '6':
                print("Goodbye!")
                break
            
            else:
                print("Invalid option. Please select 1-6.")
        
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {str(e)}")

# ============================================================================
# API Tests (Integrated in the same file)
# ============================================================================

import pytest
from unittest.mock import patch, MagicMock

class TestAPI:
    """Test Suite 2: API Endpoint Tests"""
    
    def setup_method(self):
        """Setup before each test"""
        # Create a fresh API instance for each test
        self.api = StudentGradeAPI()
        
        # Mock the predictor to ensure it's always initialized
        self.mock_predictor = MagicMock()
        self.mock_predictor.predict.return_value = 'A'
        self.mock_predictor.load_model.return_value = True
        self.api.predictor = self.mock_predictor
        self.api.model_loaded = True
    
    def test_health_check(self):
        """Test health check endpoint"""
        result = self.api.health_check()
        
        assert result['status'] == 'healthy'
        assert 'model' in result
        assert 'environment' in result
        assert result['model_loaded'] == True
    
    def test_predict_valid_input(self):
        """Test prediction with valid input"""
        features = [2, 2, 1, 2]
        response = self.api.predict_grade(features)
        
        assert response['status_code'] == 200
        assert response['prediction'] == 'A'
        assert response['features'] == features
        assert response['status'] == 'success'
    
    def test_predict_invalid_feature_count(self):
        """Test prediction with wrong feature count"""
        # Too few features
        response = self.api.predict_grade([2, 2, 1])
        assert response['status_code'] == 400
        assert 'error' in response
        
        # Too many features
        response = self.api.predict_grade([2, 2, 1, 2, 1])
        assert response['status_code'] == 400
        assert 'error' in response
    
    def test_predict_invalid_feature_types(self):
        """Test prediction with invalid feature types"""
        # String values
        response = self.api.predict_grade(['a', 'b', 'c', 'd'])
        assert response['status_code'] == 400
        assert 'error' in response
        
        # Out of range values
        response = self.api.predict_grade([-1, 2, 1, 2])
        assert response['status_code'] == 400
        assert 'error' in response
        
        response = self.api.predict_grade([3, 2, 1, 2])
        assert response['status_code'] == 400
        assert 'error' in response
    
    def test_predict_model_not_loaded(self):
        """Test prediction when model is not loaded"""
        self.api.model_loaded = False
        features = [2, 2, 1, 2]
        response = self.api.predict_grade(features)
        
        assert response['status_code'] == 500
        assert 'error' in response
        assert 'Model not available' in response['error']
    
    def test_batch_predict_valid_input(self):
        """Test batch prediction with valid input"""
        samples = [
            [2, 2, 1, 2],
            [1, 1, 0, 1],
            [0, 0, 0, 0]
        ]
        response = self.api.batch_predict(samples)
        
        assert response['status_code'] == 200
        assert 'predictions' in response
        assert len(response['predictions']) == 3
        assert response['total_samples'] == 3
        assert response['status'] == 'success'
    
    def test_batch_predict_empty_samples(self):
        """Test batch prediction with empty samples"""
        response = self.api.batch_predict([])
        assert response['status_code'] == 400
        assert 'error' in response
    
    def test_batch_predict_invalid_samples(self):
        """Test batch prediction with invalid samples"""
        # Invalid sample in batch
        samples = [
            [2, 2, 1, 2],
            [1, 1, 0],  # Wrong feature count
            [0, 0, 0, 0]
        ]
        response = self.api.batch_predict(samples)
        assert response['status_code'] == 400
        assert 'error' in response

def run_api_tests():
    """Run all API tests"""
    test_instance = TestAPI()
    
    print("Running API Tests...")
    print("=" * 50)
    
    tests = [
        ('Health Check', test_instance.test_health_check),
        ('Valid Input Prediction', test_instance.test_predict_valid_input),
        ('Invalid Feature Count', test_instance.test_predict_invalid_feature_count),
        ('Invalid Feature Types', test_instance.test_predict_invalid_feature_types),
        ('Model Not Loaded', test_instance.test_predict_model_not_loaded),
        ('Batch Prediction', test_instance.test_batch_predict_valid_input),
        ('Empty Batch', test_instance.test_batch_predict_empty_samples),
        ('Invalid Batch', test_instance.test_batch_predict_invalid_samples),
    ]
    
    all_passed = True
    
    for test_name, test_method in tests:
        try:
            test_instance.setup_method()
            test_method()
            print(f"✓ {test_name} test passed")
        except Exception as e:
            print(f"✗ {test_name} test failed: {str(e)}")
            all_passed = False
    
    print("=" * 50)
    if all_passed:
        print("All API tests passed! ✓")
    else:
        print("Some API tests failed! ✗")
    
    return all_passed

# ============================================================================
# Pytest Entry Point
# ============================================================================

if __name__ == '__main__':
    # Run the command line interface
    main()