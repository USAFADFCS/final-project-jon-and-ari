#!/usr/bin/env python3
"""
Random Forest Enhancement for Combat Triage System
Adds ML-based decision support to SALT protocol
"""

import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ============================================================
# FEATURE ENGINEERING
# ============================================================
def entities_to_features(entities, sensors=None):
    """
    Convert triage entities to RF feature vector
    Handles missing data gracefully
    """
    s = sensors or {}
    
    features = {
        # Binary features (0=No, 1=Yes, -1=Unknown)
        'can_walk': 1 if entities.get('can_walk') else (0 if entities.get('can_walk') is False else -1),
        'bleeding_severe': 1 if entities.get('bleeding_severe') else 0,
        'obeys_commands': 1 if entities.get('obeys_commands') else (0 if entities.get('obeys_commands') is False else -1),
        'radial_pulse': 1 if entities.get('radial_pulse') else (0 if entities.get('radial_pulse') is False else -1),
        'responds_to_voice': 1 if entities.get('responds_to_voice') else (0 if entities.get('responds_to_voice') is False else -1),
        
        # Numerical features
        'resp_rate': entities.get('resp_rate') or -1,  # -1 = unknown
        
        # Categorical encoded as ordinal
        'mental_status': {
            'alert': 4,
            'verbal': 3,
            'pain': 2,
            'unresponsive': 1,
            None: -1
        }.get(entities.get('mental_status'), -1),
        
        # Sensor fusion
        'sensor_movement': 1 if s.get('movement_detected') else 0,
        'sensor_bleeding': 1 if s.get('thermal_bleeding_detected') else 0,
        'sensor_heart_rate': s.get('heart_rate', -1)
    }
    
    # Convert to numpy array in consistent order
    feature_vector = np.array([
        features['can_walk'],
        features['bleeding_severe'],
        features['obeys_commands'],
        features['radial_pulse'],
        features['responds_to_voice'],
        features['resp_rate'],
        features['mental_status'],
        features['sensor_movement'],
        features['sensor_bleeding'],
        features['sensor_heart_rate']
    ]).reshape(1, -1)
    
    return feature_vector


# ============================================================
# SYNTHETIC TRAINING DATA GENERATOR
# ============================================================
def generate_training_data(n_samples=1000):
    """
    Generate synthetic triage training data based on SALT rules
    This would ideally be replaced with real combat casualty data
    """
    np.random.seed(42)
    
    X = []
    y = []
    
    # Label encoding
    label_map = {
        'Minimal': 0,
        'Delayed': 1,
        'Immediate': 2,
        'Expectant': 3
    }
    
    for _ in range(n_samples):
        # Generate random patient
        can_walk = np.random.choice([0, 1, -1], p=[0.3, 0.5, 0.2])
        bleeding_severe = np.random.choice([0, 1], p=[0.7, 0.3])
        obeys_commands = np.random.choice([0, 1, -1], p=[0.2, 0.6, 0.2])
        radial_pulse = np.random.choice([0, 1, -1], p=[0.2, 0.6, 0.2])
        responds_to_voice = np.random.choice([0, 1, -1], p=[0.15, 0.7, 0.15])
        
        # Respiratory rate
        if responds_to_voice == 0:  # Unresponsive
            resp_rate = np.random.choice([0, 5, 10, 20], p=[0.3, 0.3, 0.2, 0.2])
        else:
            resp_rate = np.random.choice([12, 16, 20, 25, 35], p=[0.3, 0.3, 0.2, 0.15, 0.05])
        
        mental_status = np.random.choice([1, 2, 3, 4, -1], p=[0.1, 0.15, 0.25, 0.4, 0.1])
        
        # Sensor data
        sensor_movement = can_walk if can_walk != -1 else np.random.choice([0, 1])
        sensor_bleeding = bleeding_severe
        sensor_heart_rate = np.random.randint(40, 160) if responds_to_voice == 1 else np.random.randint(0, 180)
        
        features = [
            can_walk, bleeding_severe, obeys_commands, radial_pulse,
            responds_to_voice, resp_rate, mental_status,
            sensor_movement, sensor_bleeding, sensor_heart_rate
        ]
        
        # Apply SALT rules to label
        if can_walk == 1:
            label = 'Minimal'
        elif resp_rate == 0:
            label = 'Expectant'
        elif bleeding_severe == 1 or resp_rate >= 30 or obeys_commands == 0 or radial_pulse == 0:
            label = 'Immediate'
        elif responds_to_voice == 0:
            label = 'Immediate' if resp_rate > 0 else 'Expectant'
        else:
            label = 'Delayed'
        
        X.append(features)
        y.append(label_map[label])
    
    return np.array(X), np.array(y)


# ============================================================
# RANDOM FOREST TRIAGE MODEL
# ============================================================
class RandomForestTriageClassifier:
    """Random Forest model for combat triage"""
    
    def __init__(self, model_path='models/rf_triage.pkl'):
        self.model = None
        self.model_path = model_path
        self.label_map = {
            0: 'Minimal',
            1: 'Delayed',
            2: 'Immediate',
            3: 'Expectant'
        }
        self.feature_names = [
            'can_walk', 'bleeding_severe', 'obeys_commands', 'radial_pulse',
            'responds_to_voice', 'resp_rate', 'mental_status',
            'sensor_movement', 'sensor_bleeding', 'sensor_heart_rate'
        ]
    
    def train(self, X, y, save=True):
        """Train the Random Forest model"""
        print("🌲 Training Random Forest...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Initialize RF with optimized parameters
        self.model = RandomForestClassifier(
            n_estimators=200,           # More trees = better performance
            max_depth=10,                # Prevent overfitting
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',     # Handle imbalanced classes
            random_state=42,
            n_jobs=-1                    # Use all CPU cores
        )
        
        # Train
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        print("\n📊 Model Performance:")
        print(classification_report(y_test, y_pred, 
                                   target_names=list(self.label_map.values())))
        
        print("\n🎯 Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        # Feature importance
        print("\n🔍 Feature Importance:")
        importances = self.model.feature_importances_
        for name, importance in sorted(zip(self.feature_names, importances), 
                                      key=lambda x: x[1], reverse=True):
            print(f"  {name}: {importance:.3f}")
        
        # Save model
        if save:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
            print(f"\n✓ Model saved to {self.model_path}")
        
        return self.model
    
    def load(self):
        """Load pre-trained model"""
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"✓ Model loaded from {self.model_path}")
            return True
        else:
            print(f"⚠️  No model found at {self.model_path}")
            return False
    
    def predict(self, entities, sensors=None):
        """Make triage prediction"""
        if self.model is None:
            raise ValueError("Model not trained or loaded. Call train() or load() first.")
        
        # Convert entities to features
        X = entities_to_features(entities, sensors)
        
        # Predict
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        
        # Get category and confidence
        category = self.label_map[prediction]
        confidence = probabilities[prediction]
        
        # Get all class probabilities
        prob_dict = {self.label_map[i]: prob for i, prob in enumerate(probabilities)}
        
        return category, confidence, prob_dict


# ============================================================
# HYBRID TRIAGE SYSTEM
# ============================================================
def hybrid_triage_decision(entities, sensors=None, rf_model=None, use_rf=True):
    """
    Hybrid triage: RF + SALT rule-based validation
    """
    from main import salt_rules_direct_patient, calculate_confidence
    
    # Get SALT rule-based result (always compute as safety)
    salt_category = salt_rules_direct_patient(entities, sensors)
    salt_confidence = calculate_confidence(entities)
    
    # If RF not available or disabled, use SALT
    if not use_rf or rf_model is None or rf_model.model is None:
        return {
            'category': salt_category,
            'confidence': salt_confidence,
            'method': 'SALT_ONLY',
            'rf_category': None,
            'rf_confidence': None,
            'agreement': None
        }
    
    try:
        # Get RF prediction
        rf_category, rf_confidence, rf_probs = rf_model.predict(entities, sensors)
        
        # Check agreement
        agreement = (rf_category == salt_category)
        
        # Decision logic
        if rf_confidence >= 0.85:
            # High confidence RF
            if agreement:
                # Both agree - highest confidence
                final_category = rf_category
                final_confidence = rf_confidence
                method = 'RF+SALT_AGREE'
            else:
                # Disagreement - use RF but flag
                final_category = rf_category
                final_confidence = rf_confidence * 0.9  # Slight penalty for disagreement
                method = 'RF_OVERRIDE'
                print(f"⚠️  DISAGREEMENT: RF says {rf_category}, SALT says {salt_category}")
        
        elif rf_confidence >= 0.60:
            # Medium confidence - prefer SALT if it's more severe
            severity_order = ['Minimal', 'Delayed', 'Immediate', 'Expectant']
            rf_severity = severity_order.index(rf_category)
            salt_severity = severity_order.index(salt_category)
            
            if salt_severity >= rf_severity:
                # SALT is same or more severe - use SALT
                final_category = salt_category
                final_confidence = salt_confidence
                method = 'SALT_SAFETY'
            else:
                # RF is more severe - use RF
                final_category = rf_category
                final_confidence = rf_confidence
                method = 'RF_MEDIUM'
        
        else:
            # Low confidence - use proven SALT rules
            final_category = salt_category
            final_confidence = salt_confidence
            method = 'SALT_FALLBACK'
        
        return {
            'category': final_category,
            'confidence': final_confidence,
            'method': method,
            'rf_category': rf_category,
            'rf_confidence': rf_confidence,
            'rf_probabilities': rf_probs,
            'salt_category': salt_category,
            'salt_confidence': salt_confidence,
            'agreement': agreement
        }
    
    except Exception as e:
        print(f"⚠️  RF prediction failed: {e}")
        # Fallback to SALT
        return {
            'category': salt_category,
            'confidence': salt_confidence,
            'method': 'SALT_ERROR_FALLBACK',
            'rf_category': None,
            'rf_confidence': None,
            'agreement': None
        }


# ============================================================
# TRAINING SCRIPT
# ============================================================
def train_triage_model():
    """Train and save the RF triage model"""
    print("\n" + "="*80)
    print("TRAINING RANDOM FOREST TRIAGE MODEL")
    print("="*80)
    
    # Generate training data
    print("\n📊 Generating synthetic training data...")
    X, y = generate_training_data(n_samples=5000)
    print(f"✓ Generated {len(X)} training samples")
    
    # Class distribution
    unique, counts = np.unique(y, return_counts=True)
    print("\n📈 Class distribution:")
    label_names = ['Minimal', 'Delayed', 'Immediate', 'Expectant']
    for label_id, count in zip(unique, counts):
        print(f"  {label_names[label_id]}: {count} ({count/len(y)*100:.1f}%)")
    
    # Train model
    rf_model = RandomForestTriageClassifier()
    rf_model.train(X, y, save=True)
    
    print("\n" + "="*80)
    print("✓ Training complete!")
    print("="*80)
    
    return rf_model


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("RANDOM FOREST TRIAGE ENHANCEMENT")
    print("="*80)
    print("\nOptions:")
    print("1. Train new model")
    print("2. Test existing model")
    print("3. Load and use model")
    print("="*80)
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == "1":
        # Train new model
        model = train_triage_model()
        
        # Test on a sample patient
        print("\n\n🧪 Testing on sample patient...")
        test_entities = {
            'can_walk': False,
            'bleeding_severe': True,
            'obeys_commands': True,
            'radial_pulse': False,
            'responds_to_voice': True,
            'resp_rate': 28,
            'mental_status': 'verbal'
        }
        
        result = hybrid_triage_decision(test_entities, rf_model=model)
        print(f"\n📊 Test Result:")
        print(f"  Category: {result['category']}")
        print(f"  Confidence: {result['confidence']:.2f}")
        print(f"  Method: {result['method']}")
        if result['rf_probabilities']:
            print(f"  RF Probabilities: {result['rf_probabilities']}")
    
    elif choice == "2":
        # Test existing model
        model = RandomForestTriageClassifier()
        if model.load():
            print("\nModel loaded. Enter test patient data...")
        else:
            print("No model found. Run option 1 to train first.")
    
    elif choice == "3":
        print("\n✓ Import this module and use hybrid_triage_decision()")
    
    print("\n✓ Complete")