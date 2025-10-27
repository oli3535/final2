import os
import pandas as pd
import pytest
import numpy as np
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()


class DataProcessor:
    def __init__(self):
        self.dataset_path = os.getenv('DATASET_PATH', 'data/student_data.csv')
        self.debug = os.getenv('DEBUG', 'False').lower() == 'true'

    def create_dataset_v1(self):
        """Create dataset version 1"""
        data = [
            [2, 2, 1, 2, 'A'],
            [2, 2, 1, 1, 'A'],
            [1, 2, 1, 2, 'A'],
            [2, 1, 1, 2, 'B'],
            [2, 2, 0, 2, 'B'],
            [1, 2, 1, 1, 'B'],
            [1, 1, 1, 2, 'B'],
            [2, 1, 0, 2, 'C'],
            [1, 2, 0, 1, 'C'],
            [0, 2, 1, 2, 'C'],
            [1, 1, 0, 1, 'C'],
            [0, 1, 1, 1, 'C'],
            [0, 0, 0, 0, 'D'],
            [1, 1, 0, 0, 'D'],
            [0, 2, 0, 1, 'D'],
            [0, 1, 0, 2, 'D']
        ]
        
        df = pd.DataFrame(data, columns=[
            'study_hours', 'attendance', 'participation', 'assignment_score', 'grade'
        ])
        return df

    def create_dataset_v2(self):
        """Create improved dataset version 2"""
        df_v1 = self.create_dataset_v1()
        
        # Add synthetic samples for better balance
        synthetic_data = [
            [2, 1, 1, 1, 'B'],
            [1, 1, 1, 1, 'B'],
            [0, 1, 1, 0, 'C'],
            [1, 0, 0, 1, 'D'],
        ]
        
        df_synthetic = pd.DataFrame(synthetic_data, columns=[
            'study_hours', 'attendance', 'participation', 'assignment_score', 'grade'
        ])
        
        df_v2 = pd.concat([df_v1, df_synthetic], ignore_index=True)
        return df_v2

    def save_dataset(self, df, filepath):
        """Save dataset to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)

    def create_datasets(self):
        """Create both dataset versions"""
        # Create and save v1
        df_v1 = self.create_dataset_v1()
        self.save_dataset(df_v1, 'data/student_data_v1.csv')
        
        # Create and save v2 (current version)
        df_v2 = self.create_dataset_v2()
        self.save_dataset(df_v2, self.dataset_path)
        
        print("Datasets created successfully!")
        return df_v1, df_v2

    def validate_dataset(self, df):
        """Validate dataset quality"""
        issues = []
        
        # Check for missing values
        if df.isnull().sum().sum() > 0:
            issues.append("Dataset contains missing values")
        
        # Check feature ranges
        for feature in ['study_hours', 'attendance', 'participation', 'assignment_score']:
            if feature in df.columns:
                if df[feature].min() < 0 or df[feature].max() > 2:
                    issues.append(f"Feature {feature} has values outside 0-2 range")
        
        # Check grade values
        valid_grades = ['A', 'B', 'C', 'D']
        invalid_grades = set(df['grade'].unique()) - set(valid_grades)
        if invalid_grades:
            issues.append(f"Invalid grade values found: {invalid_grades}")
        
        return issues


# ============================================================================
# Data Tests (Integrated in the same file)
# ============================================================================

class TestDataQuality:
    """Test Suite 3: Data Quality Tests"""
    
    def setup_method(self):
        self.processor = DataProcessor()
    
    def test_dataset_creation(self):
        """Test dataset creation functionality"""
        df_v1, df_v2 = self.processor.create_datasets()
        
        assert len(df_v1) == 16
        assert len(df_v2) == 20
        assert all(col in df_v1.columns for col in ['study_hours', 'attendance', 'participation', 'assignment_score', 'grade'])
    
    def test_data_validation(self):
        """Test data validation - edge cases"""
        df_v2 = self.processor.create_dataset_v2()
        
        # Test valid dataset
        issues = self.processor.validate_dataset(df_v2)
        assert len(issues) == 0, f"Data validation issues: {issues}"
    
    def test_feature_ranges(self):
        """Test feature value ranges - edge cases"""
        df = self.processor.create_dataset_v2()
        
        # Test all features are within valid range
        for feature in ['study_hours', 'attendance', 'participation', 'assignment_score']:
            assert df[feature].min() >= 0, f"Feature {feature} has values below 0"
            assert df[feature].max() <= 2, f"Feature {feature} has values above 2"
    
    def test_grade_distribution(self):
        """Test grade distribution"""
        df = self.processor.create_dataset_v2()
        
        grade_counts = df['grade'].value_counts()
        
        # Test all grades are present
        assert set(grade_counts.index) == {'A', 'B', 'C', 'D'}
        
        # Test no single grade dominates (basic balance check)
        max_count = grade_counts.max()
        min_count = grade_counts.min()
        assert max_count - min_count <= 8, "Grade distribution is too imbalanced"
    
    def test_data_consistency(self):
        """Test data consistency - edge cases"""
        df = self.processor.create_dataset_v2()
        
        # Test no duplicate rows
        duplicates = df.duplicated().sum()
        assert duplicates == 0, f"Found {duplicates} duplicate rows"
        
        # Test no negative values
        numeric_columns = ['study_hours', 'attendance', 'participation', 'assignment_score']
        for col in numeric_columns:
            assert (df[col] >= 0).all(), f"Negative values found in {col}"
    
    def test_dataset_completeness(self):
        """Test dataset completeness"""
        df = self.processor.create_dataset_v2()
        
        # Test all required columns are present
        required_columns = ['study_hours', 'attendance', 'participation', 'assignment_score', 'grade']
        assert all(col in df.columns for col in required_columns)
        
        # Test no empty dataset
        assert len(df) > 0, "Dataset is empty"

def run_data_tests():
    """Run all data quality tests"""
    test_instance = TestDataQuality()
    test_instance.setup_method()
    
    print("Running Data Quality Tests...")
    print("=" * 50)
    
    try:
        test_instance.test_dataset_creation()
        print("✓ Dataset creation test passed")
        
        test_instance.test_data_validation()
        print("✓ Data validation test passed")
        
        test_instance.test_feature_ranges()
        print("✓ Feature ranges test passed")
        
        test_instance.test_grade_distribution()
        print("✓ Grade distribution test passed")
        
        test_instance.test_data_consistency()
        print("✓ Data consistency test passed")
        
        test_instance.test_dataset_completeness()
        print("✓ Dataset completeness test passed")
        
        print("=" * 50)
        print("All data quality tests passed! ✓")
        return True
        
    except Exception as e:
        print(f"Data test failed: {str(e)}")
        return False


def run_all_tests():
    """Run all test suites"""
    print("Running Complete Test Suite")
    print("=" * 60)
    
    # Import and run model tests
    from ml.train import run_model_tests
    from app.main import run_api_tests
    
    model_passed = run_model_tests()
    api_passed = run_api_tests()
    data_passed = run_data_tests()
    
    print("=" * 60)
    if model_passed and api_passed and data_passed:
        print("🎉 ALL TEST SUITES PASSED! 🎉")
        return True
    else:
        print("❌ SOME TESTS FAILED")
        return False


if __name__ == '__main__':
    # Create datasets when script is executed
    processor = DataProcessor()
    processor.create_datasets()
    
    # Uncomment to run tests
    # run_data_tests()
    # run_all_tests()