"""
CRIXSOFT SOLUTION - Machine Learning Internship
Project 3: Gender Prediction & Age Estimation using AI
Author: Your Name
Description: Uses deep learning to predict gender and estimate age from facial features
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.callbacks import EarlyStopping
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# SYNTHETIC FACIAL FEATURE DATA GENERATION
# ============================================================================

def generate_facial_features_dataset(samples=1000, random_state=42):
    """
    Generate synthetic facial features dataset
    Features represent extracted facial characteristics:
    - Face width, height, skin tone, wrinkle count, etc.
    
    Target variables:
    - Gender: 0=Female, 1=Male
    - Age: 18-80 years
    """
    np.random.seed(random_state)
    
    # Create feature matrix
    X = np.zeros((samples, 12))
    
    # Features (normalized to 0-1 range)
    X[:, 0] = np.random.uniform(0.3, 0.7, samples)   # Face width
    X[:, 1] = np.random.uniform(0.4, 0.8, samples)   # Face height
    X[:, 2] = np.random.uniform(0.2, 0.9, samples)   # Eye distance
    X[:, 3] = np.random.uniform(0.1, 0.6, samples)   # Nose width
    X[:, 4] = np.random.uniform(0.15, 0.55, samples) # Mouth width
    X[:, 5] = np.random.uniform(0.0, 0.8, samples)   # Wrinkle count
    X[:, 6] = np.random.uniform(0.1, 0.9, samples)   # Skin tone
    X[:, 7] = np.random.uniform(0.0, 1.0, samples)   # Hair color
    X[:, 8] = np.random.uniform(0.0, 0.7, samples)   # Eyebrow density
    X[:, 9] = np.random.uniform(0.1, 0.9, samples)   # Cheek fullness
    X[:, 10] = np.random.uniform(0.0, 0.6, samples)  # Forehead lines
    X[:, 11] = np.random.uniform(0.0, 0.8, samples)  # Chin prominence
    
    # Generate gender (0=Female, 1=Male)
    # Males tend to have wider faces, more prominent chins
    gender = ((X[:, 0] + X[:, 11] + X[:, 3]) / 3 > 0.5).astype(int)
    
    # Generate age (18-80)
    # Age correlates with wrinkles and forehead lines
    age = 18 + (X[:, 5] + X[:, 10]) * 40
    age = np.clip(age, 18, 80).astype(int)
    
    feature_names = ['Face_Width', 'Face_Height', 'Eye_Distance', 'Nose_Width',
                     'Mouth_Width', 'Wrinkle_Count', 'Skin_Tone', 'Hair_Color',
                     'Eyebrow_Density', 'Cheek_Fullness', 'Forehead_Lines', 'Chin_Prominence']
    
    return X, gender, age, feature_names

# ============================================================================
# DATA ANALYSIS AND VISUALIZATION
# ============================================================================

def analyze_data(X, gender, age, feature_names):
    """Perform exploratory data analysis"""
    print("\n" + "=" * 70)
    print("DATASET ANALYSIS")
    print("=" * 70)
    
    print(f"\nDataset Shape: {X.shape}")
    print(f"Number of Samples: {X.shape[0]}")
    print(f"Number of Features: {X.shape[1]}")
    
    print(f"\nGender Distribution:")
    print(f"  • Female: {(gender == 0).sum()} ({(gender == 0).sum() / len(gender) * 100:.1f}%)")
    print(f"  • Male: {(gender == 1).sum()} ({(gender == 1).sum() / len(gender) * 100:.1f}%)")
    
    print(f"\nAge Statistics:")
    print(f"  • Mean: {age.mean():.1f} years")
    print(f"  • Median: {np.median(age):.1f} years")
    print(f"  • Min: {age.min()} years")
    print(f"  • Max: {age.max()} years")
    print(f"  • Std Dev: {age.std():.1f} years")

# ============================================================================
# BUILD NEURAL NETWORKS
# ============================================================================

def build_gender_classifier(input_dim):
    """Build neural network for gender classification"""
    model = Sequential([
        layers.Dense(64, activation='relu', input_dim=input_dim),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def build_age_predictor(input_dim):
    """Build neural network for age estimation"""
    model = Sequential([
        layers.Dense(64, activation='relu', input_dim=input_dim),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='linear')
    ])
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return model

# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_gender_classifier(X_train, X_test, y_train_gender, y_test_gender):
    """Train gender classification model"""
    print("\n" + "-" * 70)
    print("GENDER CLASSIFICATION MODEL")
    print("-" * 70)
    
    model = build_gender_classifier(X_train.shape[1])
    
    print("\nTraining Gender Classifier...")
    history = model.fit(
        X_train, y_train_gender,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        verbose=0,
        callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]
    )
    
    # Predictions
    y_pred_gender = (model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
    accuracy = accuracy_score(y_test_gender, y_pred_gender)
    
    print(f"  ✓ Model Training Complete")
    print(f"  ✓ Test Accuracy: {accuracy:.4f}")
    print(f"  ✓ Final Training Loss: {history.history['loss'][-1]:.4f}")
    
    return model, history, y_pred_gender, accuracy

def train_age_predictor(X_train, X_test, y_train_age, y_test_age):
    """Train age estimation model"""
    print("\n" + "-" * 70)
    print("AGE ESTIMATION MODEL")
    print("-" * 70)
    
    model = build_age_predictor(X_train.shape[1])
    
    print("\nTraining Age Predictor...")
    history = model.fit(
        X_train, y_train_age,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        verbose=0,
        callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]
    )
    
    # Predictions
    y_pred_age = model.predict(X_test, verbose=0).flatten()
    y_pred_age = np.round(np.clip(y_pred_age, 18, 80)).astype(int)
    mae = mean_absolute_error(y_test_age, y_pred_age)
    
    print(f"  ✓ Model Training Complete")
    print(f"  ✓ Mean Absolute Error: {mae:.2f} years")
    print(f"  ✓ Final Training Loss: {history.history['loss'][-1]:.4f}")
    
    return model, history, y_pred_age, mae

# ============================================================================
# PREDICTIONS FOR NEW SAMPLES
# ============================================================================

def predict_for_new_person(gender_model, age_model, scaler, features):
    """Make gender and age predictions for new facial features"""
    features_scaled = scaler.transform([features])
    
    gender_prob = gender_model.predict(features_scaled, verbose=0)[0][0]
    gender = "Male" if gender_prob > 0.5 else "Female"
    gender_confidence = max(gender_prob, 1 - gender_prob)
    
    predicted_age = age_model.predict(features_scaled, verbose=0)[0][0]
    predicted_age = int(np.round(np.clip(predicted_age, 18, 80)))
    
    return gender, gender_confidence, predicted_age

# ============================================================================
# VISUALIZATION
# ============================================================================

def create_visualizations(X, gender, age, gender_model, age_model, X_test, 
                         y_test_gender, y_pred_gender, y_test_age, y_pred_age, 
                         scaler):
    """Create comprehensive visualizations"""
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Gender distribution
    ax1 = plt.subplot(2, 3, 1)
    gender_counts = pd.Series(gender).value_counts().sort_index()
    colors = ['#FF69B4', '#4169E1']
    ax1.bar(['Female', 'Male'], gender_counts.values, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Number of People')
    ax1.set_title('Gender Distribution in Dataset')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Age distribution
    ax2 = plt.subplot(2, 3, 2)
    ax2.hist(age, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    ax2.axvline(age.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {age.mean():.1f}')
    ax2.set_xlabel('Age (years)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Age Distribution in Dataset')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Gender classification confusion matrix
    ax3 = plt.subplot(2, 3, 3)
    cm_gender = confusion_matrix(y_test_gender, y_pred_gender)
    sns.heatmap(cm_gender, annot=True, fmt='d', cmap='Blues', ax=ax3,
                xticklabels=['Female', 'Male'], yticklabels=['Female', 'Male'])
    ax3.set_ylabel('True Label')
    ax3.set_xlabel('Predicted Label')
    ax3.set_title('Gender Classification Confusion Matrix')
    
    # 4. Age prediction scatter
    ax4 = plt.subplot(2, 3, 4)
    ax4.scatter(y_test_age, y_pred_age, alpha=0.6, edgecolors='k')
    min_age = min(y_test_age.min(), y_pred_age.min())
    max_age = max(y_test_age.max(), y_pred_age.max())
    ax4.plot([min_age, max_age], [min_age, max_age], 'r--', linewidth=2, label='Perfect Prediction')
    ax4.set_xlabel('Actual Age (years)')
    ax4.set_ylabel('Predicted Age (years)')
    ax4.set_title('Age Estimation Accuracy')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Age prediction error distribution
    ax5 = plt.subplot(2, 3, 5)
    age_errors = y_test_age - y_pred_age
    ax5.hist(age_errors, bins=20, color='lightcoral', edgecolor='black', alpha=0.7)
    ax5.axvline(age_errors.mean(), color='darkred', linestyle='--', linewidth=2)
    ax5.set_xlabel('Prediction Error (years)')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Age Prediction Error Distribution')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Feature importance (correlation with target)
    ax6 = plt.subplot(2, 3, 6)
    X_df = pd.DataFrame(X)
    correlations = [abs(X_df[i].corr(pd.Series(age))) for i in range(X.shape[1])]
    feature_names = [f'F{i+1}' for i in range(X.shape[1])]
    sorted_indices = np.argsort(correlations)[-8:]
    ax6.barh([feature_names[i] for i in sorted_indices], 
             [correlations[i] for i in sorted_indices], color='teal', alpha=0.7, edgecolor='black')
    ax6.set_xlabel('Correlation with Age')
    ax6.set_title('Top 8 Features Correlated with Age')
    ax6.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/Gender_Age_Estimation_Analysis.png', dpi=300, bbox_inches='tight')
    print("\n  ✓ Analysis visualization saved as 'Gender_Age_Estimation_Analysis.png'")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 70)
    print("CRIXSOFT SOLUTION - Gender Prediction & Age Estimation using AI")
    print("=" * 70)
    
    # Step 1: Generate data
    print("\nStep 1: Generating Facial Features Dataset...")
    X, gender, age, feature_names = generate_facial_features_dataset(samples=1000, random_state=42)
    print(f"  ✓ Generated dataset with {len(X)} samples and {X.shape[1]} facial features")
    
    # Step 2: Analyze data
    print("\nStep 2: Analyzing Dataset...")
    analyze_data(X, gender, age, feature_names)
    
    # Step 3: Prepare data
    print("\nStep 3: Preparing Data for Modeling...")
    X_train, X_test, y_train_gender, y_test_gender, y_train_age, y_test_age = train_test_split(
        X, gender, age, test_size=0.2, random_state=42
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"  ✓ Training set: {X_train.shape[0]} samples")
    print(f"  ✓ Testing set: {X_test.shape[0]} samples")
    print(f"  ✓ Features standardized using StandardScaler")
    
    # Step 4: Train models
    print("\n" + "=" * 70)
    print("STEP 4: TRAINING NEURAL NETWORKS")
    print("=" * 70)
    
    gender_model, gender_history, y_pred_gender, gender_accuracy = train_gender_classifier(
        X_train, X_test, y_train_gender, y_test_gender
    )
    
    age_model, age_history, y_pred_age, age_mae = train_age_predictor(
        X_train, X_test, y_train_age, y_test_age
    )
    
    # Step 5: Detailed evaluation
    print("\n" + "=" * 70)
    print("MODEL EVALUATION RESULTS")
    print("=" * 70)
    
    print(f"\nGender Classification:")
    print(f"  • Accuracy: {gender_accuracy:.4f}")
    print(f"  • Correctly Classified: {(y_pred_gender == y_test_gender).sum()} / {len(y_test_gender)}")
    
    print(f"\nAge Estimation:")
    print(f"  • Mean Absolute Error: {age_mae:.2f} years")
    print(f"  • Accuracy within ±3 years: {((np.abs(y_test_age - y_pred_age) <= 3).sum() / len(y_test_age) * 100):.1f}%")
    print(f"  • Accuracy within ±5 years: {((np.abs(y_test_age - y_pred_age) <= 5).sum() / len(y_test_age) * 100):.1f}%")
    
    # Step 6: Example predictions
    print("\n" + "=" * 70)
    print("SAMPLE PREDICTIONS")
    print("=" * 70)
    
    test_samples = [
        {
            'name': 'Sample Person 1',
            'features': [0.5, 0.6, 0.4, 0.3, 0.4, 0.2, 0.5, 0.6, 0.5, 0.5, 0.1, 0.4]
        },
        {
            'name': 'Sample Person 2',
            'features': [0.35, 0.55, 0.35, 0.25, 0.35, 0.6, 0.45, 0.65, 0.4, 0.6, 0.5, 0.6]
        },
        {
            'name': 'Sample Person 3',
            'features': [0.65, 0.65, 0.5, 0.4, 0.5, 0.4, 0.55, 0.55, 0.6, 0.45, 0.25, 0.7]
        }
    ]
    
    for sample in test_samples:
        gender, gender_conf, predicted_age = predict_for_new_person(
            gender_model, age_model, scaler, sample['features']
        )
        print(f"\n{sample['name']}:")
        print(f"  • Predicted Gender: {gender} (Confidence: {gender_conf:.2%})")
        print(f"  • Predicted Age: {predicted_age} years")
    
    # Step 7: Visualizations
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    print("\nStep 7: Creating Analysis Visualizations...")
    
    create_visualizations(
        X, gender, age, gender_model, age_model, X_test,
        y_test_gender, y_pred_gender, y_test_age, y_pred_age, scaler
    )
    
    print("\n" + "=" * 70)
    print("✓ Project 3 Completed Successfully!")
    print("=" * 70)
    
    return {
        'gender_model': gender_model,
        'age_model': age_model,
        'scaler': scaler,
        'gender_accuracy': gender_accuracy,
        'age_mae': age_mae
    }

if __name__ == "__main__":
    results = main()
