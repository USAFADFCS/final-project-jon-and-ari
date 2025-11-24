#!/usr/bin/env python3
"""
RATS AI TRIAGE CLASSIFIER - MARCH Protocol Implementation
==========================================================
Raspberry Pi-optimized combat triage system using:
- MARCH protocol (Massive hemorrhage, Airway, Respiration, Circulation, Hypothermia)
- Random Forest classifier trained on synthetic data
- Real-time audio transcription with contextual understanding
- GPS coordinate transmission to first responders

Author: Jonathon Watson
Version: 3.0 - MARCH Protocol Focus
"""

import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')
import re

# Audio processing
import sounddevice as sd
from scipy.io.wavfile import write

# ML Models 
from transformers import pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# ============================================================
# CONFIGURATION - Optimized for Raspberry Pi
# ============================================================
CONFIG = {
    # Whisper model (use tiny for Pi, small for better accuracy)
    "whisper_model": "openai/whisper-tiny.en",  # Faster on Pi
    "device": "cpu",
    
    # Zero-shot classifier
    "classifier_model": "typeform/distilbert-base-uncased-mnli",
    
    # Audio settings
    "sample_rate": 16000,
    "enable_audio": True,
    "recording_timeout": 5,
    
    # Triage thresholds
    "confidence_threshold": 0.35,
    
    # File paths
    "model_save_path": "march_triage_rf.pkl",
    "output_dir": "triage_assessments",
    "synthetic_data_path": "synthetic_training_data.pkl"
}

print("="*80)
print("RATS AI TRIAGE CLASSIFIER - MARCH PROTOCOL")
print("="*80)

# Load Whisper ASR
print("\n[1/3] Loading Whisper speech recognition...")
asr = pipeline(
    "automatic-speech-recognition",
    model=CONFIG["whisper_model"],
    device=CONFIG["device"]
)
print("✓ Whisper model loaded")

# Load Zero-Shot Classifier
print("\n[2/3] Loading contextual classifier...")
classifier = pipeline(
    "zero-shot-classification",
    model=CONFIG["classifier_model"],
    device=-1
)
print("✓ Classifier loaded")

print("\n[3/3] System ready")
print("="*80 + "\n")


# ============================================================
# MODULE 1: MARCH PROTOCOL ASSESSMENT ENGINE
# ============================================================
class MARCHProtocolAssessment:
    """
    MARCH Protocol-based assessment system.
    
    MARCH Components:
    - M: Massive hemorrhage (life-threatening bleeding)
    - A: Airway (obstruction, ability to speak)
    - R: Respiration (breathing rate, difficulty)
    - C: Circulation (heart rate, pulse, shock signs)
    - H: Hypothermia (body temperature, exposure)
    """
    
    def __init__(self, classifier_pipeline):
        self.classifier = classifier_pipeline
        
        # MARCH assessment questions
        self.march_questions = [
            {
                "category": "MASSIVE_HEMORRHAGE",
                "priority": 1,
                "questions": [
                    {
                        "id": "bleeding_check",
                        "text": "Are you bleeding? Tell me where and how much.",
                        "target": "massive_hemorrhage",
                        "timeout": 10
                    }
                ]
            },
            {
                "category": "AIRWAY",
                "priority": 2,
                "questions": [
                    {
                        "id": "airway_check",
                        "text": "Can you speak clearly? Say your name for me.",
                        "target": "airway_patent",
                        "timeout": 5
                    }
                ]
            },
            {
                "category": "RESPIRATION",
                "priority": 3,
                "questions": [
                    {
                        "id": "breathing_check",
                        "text": "Are you having trouble breathing? Take a deep breath.",
                        "target": "respiration_adequate",
                        "timeout": 10
                    },
                    {
                        "id": "resp_rate_count",
                        "text": "I'm counting your breaths for 15 seconds. Breathe normally.",
                        "target": "resp_rate",
                        "timeout": 20,
                        "observational": True
                    }
                ]
            },
            {
                "category": "CIRCULATION",
                "priority": 4,
                "questions": [
                    {
                        "id": "consciousness_check",
                        "text": "Can you hear me? Squeeze my hand if you understand.",
                        "target": "circulation_adequate",
                        "timeout": 5
                    }
                ]
            },
            {
                "category": "HYPOTHERMIA",
                "priority": 5,
                "questions": [
                    {
                        "id": "temperature_check",
                        "text": "Are you feeling cold or shivering?",
                        "target": "hypothermia_risk",
                        "timeout": 5
                    }
                ]
            }
        ]
    
    # def analyze_march_response(self, 
    #                            category: str,
    #                            question_id: str,
    #                            patient_response: str,
    #                            target: str) -> Dict:
    #     """
    #     Analyze patient response for specific MARCH component.
        
    #     Returns:
    #         {
    #             'value': extracted value,
    #             'severity': 0-4 (0=normal, 4=critical),
    #             'confidence': 0.0-1.0,
    #             'evidence': explanation
    #         }
    #     """
    #     text_lower = patient_response.lower()
        
    #     # Route to appropriate analyzer
    #     if category == "MASSIVE_HEMORRHAGE":
    #         return self._analyze_hemorrhage(patient_response, text_lower)
        
    #     elif category == "AIRWAY":
    #         return self._analyze_airway(patient_response, text_lower)
        
    #     elif category == "RESPIRATION":
    #         if target == "resp_rate":
    #             return self._analyze_resp_rate(patient_response, text_lower)
    #         else:
    #             return self._analyze_breathing(patient_response, text_lower)
        
    #     elif category == "CIRCULATION":
    #         return self._analyze_circulation(patient_response, text_lower)
        
    #     elif category == "HYPOTHERMIA":
    #         return self._analyze_hypothermia(patient_response, text_lower)
        
    #     return {
    #         'value': None,
    #         'severity': 0,
    #         'confidence': 0.0,
    #         'evidence': 'Unknown category'
    #     }
    
    def analyze_march_response(self, 
                        category: str,
                        question_id: str,
                        patient_response: str,
                        target: str) -> Dict:
        """
        Analyze patient response for specific MARCH component.
        
        Returns:
            {
                'value': extracted value,
                'severity': 0-4 (0=normal, 4=critical),
                'confidence': 0.0-1.0,
                'evidence': explanation
            }
        """
        text_lower = patient_response.lower()
        
        # Route to appropriate analyzer
        if category == "MASSIVE_HEMORRHAGE":
            return self._analyze_hemorrhage(patient_response, text_lower)
        
        elif category == "AIRWAY":
            return self._analyze_airway(patient_response, text_lower)
        
        elif category == "RESPIRATION":
            if target == "resp_rate":
                return self._analyze_resp_rate(patient_response, text_lower)
            else:
                return self._analyze_breathing(patient_response, text_lower)
        
        elif category == "CIRCULATION":
            return self._analyze_circulation(patient_response, text_lower)
        
        elif category == "HYPOTHERMIA":
            return self._analyze_hypothermia(patient_response, text_lower)
        
        return {
            'value': None,
            'severity': 0,
            'confidence': 0.0,
            'evidence': 'Unknown category'
    }

    def _analyze_hemorrhage(self, response: str, text_lower: str) -> Dict:
        """
        Analyze for massive hemorrhage (M in MARCH).
        Severity: 0=none, 1=minor, 2=moderate, 3=severe, 4=life-threatening
        
        Strategy:
        1. Check if bleeding is controlled/resolved (early exit)
        2. Check for very specific severe patterns (high confidence regex)
        3. Use AI classification for everything else (nuanced assessment)
        """
        
        # Step 1: Check for controlled/resolved bleeding
        controlled_patterns = [
            r'\b(stopped|controlled|under control)\b',
            r'\b(tourniquet).*\b(stopped|working|applied)\b',
            r'\b(pressure).*\b(stopped|working|helping)\b',
            r'\b(bleeding).*\b(stopped|slowed|better)\b',
            r'\b(no longer|not anymore).*\b(bleeding)\b'
        ]
        
        has_bleeding_mention = re.search(r'\b(bleeding|blood|hemorrhage)\b', text_lower)
        is_controlled = False
        
        if has_bleeding_mention:
            for pattern in controlled_patterns:
                if re.search(pattern, text_lower):
                    is_controlled = True
                    break
        
        if is_controlled:
            return {
                'value': True,
                'severity': 1,  # Minor - intervention successful
                'confidence': 0.90,
                'evidence': 'Bleeding mentioned but successfully controlled with intervention'
            }
        
        # Step 2: Check for VERY SPECIFIC severe patterns only
        # These are unambiguous indicators of life-threatening hemorrhage
        life_threatening_patterns = [
            r'\b(spurting|gushing|pouring)\b.*\b(blood)\b',
            r'\b(blood).*\b(spurting|gushing|pouring)\b',
            r'\b(can\'t stop|won\'t stop).*\b(bleeding)\b',
            r'\b(bleeding).*\b(can\'t stop|won\'t stop)\b',
            r'\b(arterial).*\b(bleeding)\b',
            r'\b(bleeding out)\b',
            r'\b(massive hemorrhage)\b',
            r'\b(exsanguinating)\b'
        ]
        
        for pattern in life_threatening_patterns:
            if re.search(pattern, text_lower):
                return {
                    'value': True,
                    'severity': 4,
                    'confidence': 0.95,
                    'evidence': 'Life-threatening hemorrhage detected - unambiguous pattern'
                }
        
        # Step 3: If bleeding mentioned but not severe regex, use AI for nuanced classification
        if has_bleeding_mention:
            try:
                result = self.classifier(
                    response,
                    candidate_labels=[
                        "life-threatening massive uncontrolled bleeding",
                        "severe heavy bleeding requiring immediate attention",
                        "moderate bleeding that needs treatment soon",
                        "minor bleeding or small wound"
                    ]
                )
                
                # Map AI classification to severity
                top_label = result['labels'][0]
                confidence = result['scores'][0]
                
                if "life-threatening" in top_label:
                    severity = 4
                elif "severe" in top_label:
                    severity = 3
                elif "moderate" in top_label:
                    severity = 2
                else:
                    severity = 1
                
                return {
                    'value': True,
                    'severity': severity,
                    'confidence': confidence,
                    'evidence': f'AI classified: {top_label}'
                }
            except Exception as e:
                # Fallback if AI fails
                return {
                    'value': True,
                    'severity': 2,
                    'confidence': 0.60,
                    'evidence': 'Bleeding detected, AI classification failed - defaulting to moderate'
                }
        
        # No bleeding mentioned
        return {
            'value': False,
            'severity': 0,
            'confidence': 0.85,
            'evidence': 'No bleeding indicators found'
        }

    def _analyze_airway(self, response: str, text_lower: str) -> Dict:
        """
        Analyze airway patency (A in MARCH).
        If patient can speak clearly, airway is patent (open).
        """
        if len(response.strip()) > 5:
            # Patient spoke - airway is patent
            return {
                'value': True,
                'severity': 0,
                'confidence': 0.95,
                'evidence': 'Patient speaking clearly - airway patent'
            }
        else:
            # No clear speech - potential airway issue
            return {
                'value': False,
                'severity': 4,
                'confidence': 0.90,
                'evidence': 'Unable to speak - potential airway obstruction'
            }

    def _analyze_breathing(self, response: str, text_lower: str) -> Dict:
        """
        Analyze breathing adequacy (R in MARCH).
        """
        # Respiratory distress patterns
        distress_patterns = [
            r'\b(can\'t breathe|can\'t breath|gasping|choking)\b',
            r'\b(hard to breathe|difficult|struggling)\b',
            r'\b(chest.*tight|chest.*pain)\b'
        ]
        
        # Normal breathing patterns
        normal_patterns = [
            r'\b(breathing fine|breathing okay|no problem)\b',
            r'\b(okay|fine|good)\b'
        ]
        
        # Check distress
        for pattern in distress_patterns:
            if re.search(pattern, text_lower):
                return {
                    'value': False,
                    'severity': 3,
                    'confidence': 0.90,
                    'evidence': 'Respiratory distress detected'
                }
        
        # Check normal
        for pattern in normal_patterns:
            if re.search(pattern, text_lower):
                return {
                    'value': True,
                    'severity': 0,
                    'confidence': 0.85,
                    'evidence': 'Breathing appears adequate'
                }
        
        # AI classification
        try:
            result = self.classifier(
                response,
                candidate_labels=[
                    "severe respiratory distress",
                    "mild breathing difficulty",
                    "breathing normally"
                ]
            )
            
            if "severe" in result['labels'][0]:
                severity = 3
                adequate = False
            elif "mild" in result['labels'][0]:
                severity = 2
                adequate = False
            else:
                severity = 0
                adequate = True
            
            return {
                'value': adequate,
                'severity': severity,
                'confidence': result['scores'][0],
                'evidence': f'AI assessment: {result["labels"][0]}'
            }
        except:
            pass
        
        return {
            'value': None,
            'severity': 2,
            'confidence': 0.50,
            'evidence': 'Unable to assess breathing adequacy'
        }

    def _analyze_resp_rate(self, response: str, text_lower: str) -> Dict:
        """
        Extract respiratory rate (R in MARCH).
        Normal: 12-20 breaths/min
        Concerning: <10 or >30
        """        
        patterns = [
            r'(\d+)\s*breaths?',
            r'(\d+)\s*per\s*minute',
            r'counted\s*(\d+)',
            r'rate\s*(?:of\s*)?(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                rate = int(match.group(1))
                
                if 12 <= rate <= 20:
                    severity = 0
                    evidence = f'Normal respiratory rate: {rate}/min'
                elif 10 <= rate < 12 or 20 < rate <= 30:
                    severity = 2
                    evidence = f'Abnormal respiratory rate: {rate}/min'
                else:
                    severity = 3
                    evidence = f'Critical respiratory rate: {rate}/min'
                
                return {
                    'value': rate,
                    'severity': severity,
                    'confidence': 0.95,
                    'evidence': evidence
                }
        
        return {
            'value': None,
            'severity': 2,
            'confidence': 0.0,
            'evidence': 'Could not extract respiratory rate'
        }

    def _analyze_circulation(self, response: str, text_lower: str) -> Dict:
        """
        Analyze circulation adequacy (C in MARCH).
        Based on consciousness and response to commands.
        """
        if len(response.strip()) > 0:
            # Patient responsive - circulation likely adequate
            return {
                'value': True,
                'severity': 0,
                'confidence': 0.85,
                'evidence': 'Patient responsive - circulation adequate'
            }
        else:
            # Unresponsive - circulation compromised
            return {
                'value': False,
                'severity': 4,
                'confidence': 0.90,
                'evidence': 'Patient unresponsive - circulation compromised'
            }

    def _analyze_hypothermia(self, response: str, text_lower: str) -> Dict:
        """
        Analyze hypothermia risk (H in MARCH).
        """
        
        # Cold/hypothermia patterns
        cold_patterns = [
            r'\b(freezing|very cold|shivering|shaking)\b',
            r'\b(can\'t feel|numb|frozen)\b'
        ]
        
        # Warm patterns
        warm_patterns = [
            r'\b(warm|hot|fine|okay)\b',
            r'\b(not cold|no)\b'
        ]
        
        for pattern in cold_patterns:
            if re.search(pattern, text_lower):
                return {
                    'value': True,
                    'severity': 2,
                    'confidence': 0.85,
                    'evidence': 'Hypothermia risk detected'
                }
        
        for pattern in warm_patterns:
            if re.search(pattern, text_lower):
                return {
                    'value': False,
                    'severity': 0,
                    'confidence': 0.80,
                    'evidence': 'No hypothermia risk'
                }
        
        return {
            'value': None,
            'severity': 1,
            'confidence': 0.50,
            'evidence': 'Hypothermia status unclear'
        }


# ============================================================
# MODULE 2: RANDOM FOREST MARCH CLASSIFIER
# ============================================================
class MARCHRandomForestClassifier:
    """
    Random Forest classifier for MARCH-based triage.
    
    Features:
    - MARCH component scores (from patient responses)
    - Sensor data (heart rate, respiration, temperature)
    - Combined criticality prediction
    
    Trained on synthetic data until real sensors available.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Feature names aligned with MARCH
        self.feature_names = [
            # MARCH verbal assessment scores (0-4 severity)
            'hemorrhage_severity',      # M: 0=none, 4=life-threatening
            'airway_severity',          # A: 0=patent, 4=obstructed
            'respiration_severity',     # R: 0=normal, 4=critical
            'circulation_severity',     # C: 0=adequate, 4=shock
            'hypothermia_severity',     # H: 0=normal, 4=severe
            
            # Sensor data (when available)
            'heart_rate',               # [SENSOR] bpm
            'resp_rate',                # [SENSOR] breaths/min
            'body_temp',                # [SENSOR] °C from thermal camera
            
            # Derived features
            'total_march_score',        # Sum of MARCH severities
            'critical_components'       # Count of severity >= 3
        ]
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            )
    
    def encode_march_features(self,
                             march_results: Dict,
                             sensor_data: Dict) -> np.ndarray:
        """
        Convert MARCH assessment + sensor data into feature vector.
        
        Args:
            march_results: Dict with MARCH component analyses
            sensor_data: Dict with sensor readings
        
        Returns:
            Feature vector for classification
        """
        features = []
        
        # Extract MARCH severities
        hemorrhage_sev = march_results.get('MASSIVE_HEMORRHAGE', {}).get('severity', 0)
        airway_sev = march_results.get('AIRWAY', {}).get('severity', 0)
        respiration_sev = march_results.get('RESPIRATION', {}).get('severity', 0)
        circulation_sev = march_results.get('CIRCULATION', {}).get('severity', 0)
        hypothermia_sev = march_results.get('HYPOTHERMIA', {}).get('severity', 0)
        
        features.extend([
            hemorrhage_sev,
            airway_sev,
            respiration_sev,
            circulation_sev,
            hypothermia_sev
        ])
        
        # Sensor data (with defaults if not available)
        heart_rate = sensor_data.get('heart_rate', 80)
        resp_rate = sensor_data.get('resp_rate', 16)
        body_temp = sensor_data.get('body_temp', 36.5)
        
        features.extend([
            heart_rate,
            resp_rate,
            body_temp
        ])
        
        # Derived features
        total_score = sum([hemorrhage_sev, airway_sev, respiration_sev, 
                          circulation_sev, hypothermia_sev])
        critical_count = sum(1 for sev in [hemorrhage_sev, airway_sev, 
                                           respiration_sev, circulation_sev, 
                                           hypothermia_sev] if sev >= 3)
        
        features.extend([
            total_score,
            critical_count
        ])
        
        return np.array(features).reshape(1, -1)
    
    def predict_criticality(self,
                           march_results: Dict,
                           sensor_data: Dict = None) -> Dict:
        """
        Predict patient criticality using Random Forest.
        
        Returns:
            {
                'category': 'IMMEDIATE'/'DELAYED'/'MINIMAL'/'EXPECTANT',
                'criticality_score': 0-100,
                'confidence': 0.0-1.0,
                'march_summary': MARCH component summary,
                'gps_coordinates': location data,
                'first_responder_package': transmission data
            }
        """
        if sensor_data is None:
            sensor_data = {}
        
        # Encode features
        X = self.encode_march_features(march_results, sensor_data)
        
        # If not trained, use rule-based MARCH triage
        if not self.is_trained:
            return self._rule_based_march_triage(march_results, sensor_data, X[0])
        
        # ML prediction
        X_scaled = self.scaler.transform(X)
        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]
        confidence = max(probabilities)
        
        categories = ['MINIMAL', 'DELAYED', 'IMMEDIATE', 'EXPECTANT']
        category = categories[prediction]
        
        # Calculate criticality score
        criticality = self._calculate_march_criticality(march_results, X[0])
        
        # Generate MARCH summary
        march_summary = self._generate_march_summary(march_results)
        
        # Create first responder package
        first_responder_package = self._create_transmission_package(
            category,
            criticality,
            march_summary,
            sensor_data
        )
        
        return {
            'category': category,
            'criticality_score': criticality,
            'confidence': confidence,
            'march_summary': march_summary,
            'gps_coordinates': {
                'latitude': sensor_data.get('gps_lat'),
                'longitude': sensor_data.get('gps_lon')
            },
            'first_responder_package': first_responder_package
        }
    
    def _rule_based_march_triage(self,
                                 march_results: Dict,
                                 sensor_data: Dict,
                                 features: np.ndarray) -> Dict:
        """
        Rule-based triage using MARCH priorities.
        Used when ML model not yet trained.
        """
        # Extract severities
        hemorrhage_sev = march_results.get('MASSIVE_HEMORRHAGE', {}).get('severity', 0)
        airway_sev = march_results.get('AIRWAY', {}).get('severity', 0)
        respiration_sev = march_results.get('RESPIRATION', {}).get('severity', 0)
        circulation_sev = march_results.get('CIRCULATION', {}).get('severity', 0)
        
        # MARCH priority-based decision
        if hemorrhage_sev >= 3:
            category = 'IMMEDIATE'
            criticality = 90
        elif airway_sev >= 3:
            category = 'IMMEDIATE'
            criticality = 95
        elif respiration_sev >= 3:
            category = 'IMMEDIATE'
            criticality = 85
        elif circulation_sev >= 4:
            category = 'EXPECTANT'
            criticality = 98
        elif circulation_sev >= 3:
            category = 'IMMEDIATE'
            criticality = 80
        elif any(sev >= 2 for sev in [hemorrhage_sev, respiration_sev]):
            category = 'DELAYED'
            criticality = 50
        else:
            category = 'MINIMAL'
            criticality = 20
        
        march_summary = self._generate_march_summary(march_results)
        
        first_responder_package = self._create_transmission_package(
            category,
            criticality,
            march_summary,
            sensor_data
        )
        
        return {
            'category': category,
            'criticality_score': criticality,
            'confidence': 0.75,
            'march_summary': march_summary,
            'gps_coordinates': {
                'latitude': sensor_data.get('gps_lat'),
                'longitude': sensor_data.get('gps_lon')
            },
            'first_responder_package': first_responder_package,
            'method': 'rule_based_MARCH'
        }
    
    def _calculate_march_criticality(self,
                                    march_results: Dict,
                                    features: np.ndarray) -> float:
        """
        Calculate 0-100 criticality score based on MARCH.
        """
        total_score = features[8]  # total_march_score
        critical_count = features[9]  # critical_components
        
        # Base score from total MARCH severity
        base_score = (total_score / 20) * 100  # Max possible is 20 (5 components * 4)
        
        # Boost for critical components
        critical_boost = critical_count * 15
        
        # Cap at 100
        return min(base_score + critical_boost, 100)
    
    def _generate_march_summary(self, march_results: Dict) -> str:
        """
        Generate concise MARCH summary for first responders.
        
        Format: "M:SEVERE A:PATENT R:DISTRESS C:ADEQUATE H:RISK"
        """
        severity_map = {
            0: 'NORMAL',
            1: 'MINOR',
            2: 'MODERATE',
            3: 'SEVERE',
            4: 'CRITICAL'
        }
        
        components = []
        
        for category in ['MASSIVE_HEMORRHAGE', 'AIRWAY', 'RESPIRATION', 
                        'CIRCULATION', 'HYPOTHERMIA']:
            letter = category[0]
            severity = march_results.get(category, {}).get('severity', 0)
            status = severity_map.get(severity, 'UNKNOWN')
            components.append(f"{letter}:{status}")
        
        return " ".join(components)
    
    def _create_transmission_package(self,
                                    category: str,
                                    criticality: float,
                                    march_summary: str,
                                    sensor_data: Dict) -> Dict:
        """
        Create standardized package for first responder transmission.
        """
        return {
            'timestamp': datetime.now().isoformat(),
            'triage_category': category,
            'criticality_score': criticality,
            'march_summary': march_summary,
            'location': {
                'latitude': sensor_data.get('gps_lat'),
                'longitude': sensor_data.get('gps_lon'),
                'accuracy_meters': sensor_data.get('gps_accuracy', 10)
            },
            'vitals': {
                'heart_rate': sensor_data.get('heart_rate'),
                'respiratory_rate': sensor_data.get('resp_rate'),
                'body_temperature': sensor_data.get('body_temp')
            },
            'recommended_actions': self._recommend_march_interventions(march_summary),
            'transport_priority': 1 if criticality >= 80 else (2 if criticality >= 50 else 3)
        }
    
    def _recommend_march_interventions(self, march_summary: str) -> List[str]:
        # Could be used to provide patient self help instructions through drone speaker
        """Recommend interventions based on MARCH summary"""
        interventions = []
        
        if 'M:SEVERE' in march_summary or 'M:CRITICAL' in march_summary:
            interventions.append('APPLY_TOURNIQUET')
            interventions.append('DIRECT_PRESSURE')
        
        if 'A:SEVERE' in march_summary or 'A:CRITICAL' in march_summary:
            interventions.append('SECURE_AIRWAY')
            interventions.append('HEAD_TILT_CHIN_LIFT')
        
        if 'R:SEVERE' in march_summary or 'R:CRITICAL' in march_summary:
            interventions.append('OXYGEN_THERAPY')
            interventions.append('CHEST_DECOMPRESSION_IF_NEEDED')
        
        if 'C:SEVERE' in march_summary or 'C:CRITICAL' in march_summary:
            interventions.append('IV_ACCESS')
            interventions.append('FLUID_RESUSCITATION')
        
        if 'H:MODERATE' in march_summary or 'H:SEVERE' in march_summary:
            interventions.append('PREVENT_HEAT_LOSS')
            interventions.append('WARM_FLUIDS')
        
        return interventions
    
    def train_on_synthetic_data(self, n_samples: int = 1000):
        """
        Generate synthetic training data and train the model.
        
        Creates realistic MARCH scenarios with corresponding triage categories.
        """
        print(f"\n🔧 Generating {n_samples} synthetic training samples...")
        
        X_train = []
        y_train = []
        
        for _ in range(n_samples):
            # Generate random MARCH severities
            hemorrhage = np.random.choice([0, 0, 0, 1, 2, 3, 4], p=[0.4, 0.2, 0.1, 0.1, 0.1, 0.05, 0.05])
            airway = np.random.choice([0, 0, 0, 1, 3, 4], p=[0.7, 0.15, 0.05, 0.05, 0.03, 0.02])
            respiration = np.random.choice([0, 0, 1, 2, 3], p=[0.6, 0.2, 0.1, 0.07, 0.03])
            circulation = np.random.choice([0, 0, 1, 2, 3, 4], p=[0.6, 0.2, 0.1, 0.05, 0.03, 0.02])
            hypothermia = np.random.choice([0, 0, 1, 2], p=[0.7, 0.2, 0.08, 0.02])
            
            # Generate corresponding vitals
            if hemorrhage >= 3 or circulation >= 3:
                heart_rate = np.random.randint(120, 160)
            elif hemorrhage >= 2 or circulation >= 2:
                heart_rate = np.random.randint(100, 120)
            else:
                heart_rate = np.random.randint(60, 100)
            
            if respiration >= 3:
                resp_rate = np.random.choice([np.random.randint(5, 10), np.random.randint(30, 40)])
            elif respiration >= 2:
                resp_rate = np.random.randint(22, 30)
            else:
                resp_rate = np.random.randint(12, 20)
            
            if hypothermia >= 2:
                body_temp = np.random.uniform(32, 35)
            elif hypothermia >= 1:
                body_temp = np.random.uniform(35, 36)
            else:
                body_temp = np.random.uniform(36, 37.5)
            
            # Calculate derived features
            total_score = hemorrhage + airway + respiration + circulation + hypothermia
            critical_count = sum(1 for sev in [hemorrhage, airway, respiration, circulation, hypothermia] if sev >= 3)
            
            # Create feature vector
            features = [
                hemorrhage, airway, respiration, circulation, hypothermia,
                heart_rate, resp_rate, body_temp,
                total_score, critical_count
            ]
            
            # Determine triage category
            if airway >= 4 or circulation >= 4:
                category = 3  # EXPECTANT
            elif hemorrhage >= 3 or airway >= 3 or respiration >= 3 or circulation >= 3:
                category = 2  # IMMEDIATE
            elif total_score >= 5 or critical_count >= 1:
                category = 1  # DELAYED
            else:
                category = 0  # MINIMAL
            
            X_train.append(features)
            y_train.append(category)
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Train model
        print("🎯 Training Random Forest classifier...")
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        self.is_trained = True
        
        # Show feature importance
        print("\n📊 Feature Importance:")
        for name, importance in sorted(
            zip(self.feature_names, self.model.feature_importances_),
            key=lambda x: x[1],
            reverse=True
        ):
            print(f"   {name}: {importance:.3f}")
        
        print(f"\n✓ Model trained successfully on {n_samples} samples")
    
    def save_model(self, path: str):
        """Save trained model"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'is_trained': self.is_trained,
            'feature_names': self.feature_names
        }, path)
        print(f"💾 Model saved to {path}")
    
    def load_model(self, path: str):
        """Load trained model"""
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.is_trained = data['is_trained']
        self.feature_names = data['feature_names']
        print(f"✓ Model loaded from {path}")


# ============================================================
# MODULE 3: AUDIO INTERFACE
# ============================================================
def record_audio(duration: int = 5) -> np.ndarray:
    """Record audio from microphone"""
    if not CONFIG['enable_audio']:
        return None
    
    print(f"  🎤 Recording for {duration} seconds...")
    audio = sd.rec(
        int(duration * CONFIG['sample_rate']),
        samplerate=CONFIG['sample_rate'],
        channels=1,
        dtype='float32'
    )
    sd.wait()
    print("  ✓ Recording complete")
    return audio.flatten()


def transcribe_audio(audio_data: np.ndarray) -> str:
    """Transcribe audio using Whisper"""
    if audio_data is None:
        return ""
    
    try:
        result = asr(audio_data) #asr is the whisper model
        transcription = result['text'].strip()
        print(f"  📝 Transcription: '{transcription}'")
        return transcription
    except Exception as e:
        print(f"  ⚠️ Transcription error: {e}")
        return ""


# ============================================================
# MODULE 4: MAIN TRIAGE WORKFLOW
# ============================================================
def conduct_march_triage(
    patient_id: Optional[str] = None,
    sensor_data: Optional[Dict] = None,
    use_audio: bool = True
) -> Dict:
    """
    Main MARCH protocol triage workflow.
    
    Returns complete assessment with GPS and MARCH summary.
    """
    print("\n" + "="*80)
    print("RATS AI TRIAGE - MARCH PROTOCOL ASSESSMENT")
    print("="*80 + "\n")
    
    if patient_id is None:
        patient_id = f"PAT_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    if sensor_data is None:
        sensor_data = {}
    
    # Initialize components
    march_assessor = MARCHProtocolAssessment(classifier)
    march_classifier = MARCHRandomForestClassifier(CONFIG['model_save_path'])
    
    # Train on synthetic data if not already trained
    if not march_classifier.is_trained:
        march_classifier.train_on_synthetic_data(n_samples=1000)
        march_classifier.save_model(CONFIG['model_save_path'])
    
    # Store MARCH results
    march_results = {}
    
    # Conduct MARCH assessment
    for march_category in march_assessor.march_questions:
        category_name = march_category['category']
        
        print(f"\n{'─'*80}")
        print(f"MARCH COMPONENT: {category_name}")
        print(f"{'─'*80}")
        
        for question in march_category['questions']:
            print(f"\n🗣️  {question['text']}\n")
            
            # Get patient response
            if use_audio and CONFIG['enable_audio']:
                audio = record_audio(duration=question.get('timeout', 5))
                patient_response = transcribe_audio(audio)
            else:
                patient_response = input("  💬 Enter response: ")
            
            if not patient_response:
                patient_response = "[NO RESPONSE]"
            
            # Analyze response
            print(f"\n  🔍 Analyzing...")
            analysis = march_assessor.analyze_march_response(
                category_name,
                question['id'],
                patient_response,
                question['target']
            )
            
            # Display results
            print(f"\n  📊 ANALYSIS:")
            print(f"     Value: {analysis['value']}")
            print(f"     Severity: {analysis['severity']}/4")
            print(f"     Confidence: {analysis['confidence']:.2f}")
            print(f"     Evidence: {analysis['evidence']}")
            
            # Store result
            march_results[category_name] = analysis
    
    # Generate triage recommendation
    print(f"\n{'='*80}")
    print("GENERATING TRIAGE RECOMMENDATION")
    print(f"{'='*80}\n")
    
    triage_result = march_classifier.predict_criticality(
        march_results,
        sensor_data
    )
    
    # Display results
    print(f"\n🏥 TRIAGE CATEGORY: {triage_result['category']}")
    print(f"⚠️  CRITICALITY: {triage_result['criticality_score']:.0f}/100")
    print(f"📊 CONFIDENCE: {triage_result['confidence']:.2%}")
    print(f"\n📋 MARCH SUMMARY: {triage_result['march_summary']}")
    
    if triage_result['gps_coordinates']['latitude']:
        print(f"\n📍 GPS: {triage_result['gps_coordinates']['latitude']:.6f}, "
              f"{triage_result['gps_coordinates']['longitude']:.6f}")
    
    # Display first responder package
    print(f"\n{'='*80}")
    print("FIRST RESPONDER TRANSMISSION PACKAGE")
    print(f"{'='*80}\n")
    print(json.dumps(triage_result['first_responder_package'], indent=2))
    
    # Save assessment
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    output_file = os.path.join(
        CONFIG['output_dir'],
        f"{patient_id}_march_assessment.json"
    )
    
    assessment_data = {
        'patient_id': patient_id,
        'timestamp': datetime.now().isoformat(),
        'march_results': march_results,
        'triage_result': triage_result,
        'sensor_data': sensor_data
    }
    
    with open(output_file, 'w') as f:
        json.dump(assessment_data, f, indent=2)
    
    print(f"\n💾 Assessment saved to: {output_file}")
    
    return assessment_data


# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("RATS AI TRIAGE CLASSIFIER v3.0")
    print("MARCH Protocol Implementation")
    print("="*80 + "\n")
    
    # Mock sensor data (replace with actual GPS and sensors)
    mock_sensor_data = {
        'patient_id': 'COMBAT_001',
        'gps_lat': 34.0522,
        'gps_lon': -118.2437,
        'gps_accuracy': 5,
        'heart_rate': 115,
        'resp_rate': 24,
        'body_temp': 36.1
    }
    
    # Conduct assessment
    results = conduct_march_triage(
        patient_id="COMBAT_001",
        sensor_data=mock_sensor_data,
        use_audio=CONFIG['enable_audio']
    )
    
    print("\n" + "="*80)
    print("ASSESSMENT COMPLETE")
    print("="*80 + "\n")