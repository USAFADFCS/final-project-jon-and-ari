#!/usr/bin/env python3
"""
ENHANCED COMBAT TRIAGE AI SYSTEM
==================================
Complete implementation with:
- Contextual translation error correction
- Zero-shot classification for ambiguous responses
- Random Forest criticality ranking
- Sensor data integration interfaces
- First responder transmission system

Author: Combat Triage AI Team
Version: 2.0
"""

import os
import re
import time
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Audio processing
import sounddevice as sd
from scipy.io.wavfile import write

# ML Models
from transformers import pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# ============================================================
# CONFIGURATION
# ============================================================
CONFIG = {
    # Whisper model for speech recognition (use multilingual for translation)
    "whisper_model": "openai/whisper-small.en",  # Change to "openai/whisper-medium" for multilingual
    "device": "cpu",
    
    # Zero-shot classifier for contextual understanding
    "classifier_model": "typeform/distilbert-base-uncased-mnli",
    
    # Audio settings
    "sample_rate": 16000,
    "enable_audio": True,
    
    # Triage thresholds
    "confidence_threshold": 0.35,  # Lowered for better sensitivity
    "max_unknown_ratio": 0.5,  # Flag for manual review if >50% data missing
    
    # File paths
    "model_save_path": "combat_triage_rf.pkl",
    "output_dir": "patient_assessments"
}

print("="*80)
print("COMBAT TRIAGE AI SYSTEM - INITIALIZING")
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
print("\n[2/3] Loading DistilBERT contextual classifier...")
classifier = pipeline(
    "zero-shot-classification",
    model=CONFIG["classifier_model"],
    device=-1  # CPU
)
print("✓ DistilBERT classifier loaded")

print("\n[3/3] System ready for deployment")
print("="*80 + "\n")


# ============================================================
# MODULE 1: CONTEXTUAL TRANSLATION & ERROR CORRECTION
# ============================================================
class ContextualTranslationCorrector:
    """
    Handles translation errors through contextual understanding.
    
    Key Features:
    - Medical context awareness (knows we're asking about walking, breathing, etc.)
    - Phonetic similarity matching (walk vs watch, bleeding vs breeding)
    - Zero-shot classification as fallback
    - Confidence scoring for corrections
    """
    
    def __init__(self, classifier_pipeline):
        self.classifier = classifier_pipeline
        
        # Medical context dictionary: question type -> common mistranslations
        self.medical_context_corrections = {
            "walking_ability": {
                "mistranslations": ["watch", "wok", "work", "wake"],
                "correct_term": "walk",
                "phonetic_threshold": 0.7
            },
            "bleeding_assessment": {
                "mistranslations": ["breeding", "reading", "leading", "beading"],
                "correct_term": "bleeding",
                "phonetic_threshold": 0.7
            },
            "breathing_assessment": {
                "mistranslations": ["breading", "breaking", "briefing"],
                "correct_term": "breathing",
                "phonetic_threshold": 0.75
            },
            "pain_assessment": {
                "mistranslations": ["pane", "pen", "pan"],
                "correct_term": "pain",
                "phonetic_threshold": 0.8
            }
        }
    
    def correct_with_context(self, 
                            transcribed_text: str, 
                            question_context: str,
                            expected_context: str) -> Tuple[str, float, str]:
        """
        Apply contextual correction to transcribed text.
        
        Args:
            transcribed_text: Raw output from Whisper
            question_context: The question that was asked
            expected_context: What medical context we're in (e.g., "walking_ability")
        
        Returns:
            (corrected_text, confidence, correction_method)
        """
        original_text = transcribed_text
        text_lower = transcribed_text.lower()
        
        # Step 1: Check for known mistranslations
        if expected_context in self.medical_context_corrections:
            context_data = self.medical_context_corrections[expected_context]
            
            for mistranslation in context_data["mistranslations"]:
                if mistranslation in text_lower:
                    # Found a likely mistranslation - correct it
                    corrected = transcribed_text.replace(
                        mistranslation, 
                        context_data["correct_term"]
                    )
                    corrected = corrected.replace(
                        mistranslation.capitalize(),
                        context_data["correct_term"].capitalize()
                    )
                    
                    confidence = context_data["phonetic_threshold"]
                    
                    print(f"  🔧 Contextual correction: '{mistranslation}' → '{context_data['correct_term']}'")
                    print(f"     Confidence: {confidence:.2f}")
                    
                    return corrected, confidence, "context_dictionary"
        
        # Step 2: Use zero-shot classification to verify meaning
        # This helps with completely wrong words (e.g., "I can't watch" when asking about walking)
        try:
            # Create semantic candidates based on expected context
            if expected_context == "walking_ability":
                candidates = [
                    "patient can walk or move",
                    "patient cannot walk or is immobile",
                    "unclear or ambiguous response"
                ]
            elif expected_context == "bleeding_assessment":
                candidates = [
                    "bleeding or hemorrhaging present",
                    "no bleeding mentioned",
                    "unclear injury description"
                ]
            elif expected_context == "breathing_assessment":
                candidates = [
                    "breathing normally",
                    "difficulty breathing or respiratory distress",
                    "not breathing"
                ]
            else:
                # Generic candidates
                candidates = [
                    "affirmative response",
                    "negative response", 
                    "unclear response"
                ]
            
            result = self.classifier(
                transcribed_text,
                candidate_labels=candidates,
                hypothesis_template="This response indicates {}"
            )
            
            # If we have high confidence in semantic meaning, use that
            if result['scores'][0] > 0.6:
                return transcribed_text, result['scores'][0], "zero_shot_semantic"
        
        except Exception as e:
            print(f"  ⚠️ Zero-shot correction failed: {e}")
        
        # Step 3: No correction needed or possible
        return transcribed_text, 0.5, "no_correction"
    
    def phonetic_similarity(self, word1: str, word2: str) -> float:
        """
        Calculate phonetic similarity between two words.
        Uses Levenshtein distance approximation.
        """
        # Simple phonetic similarity (can be enhanced with phonetic algorithms)
        w1, w2 = word1.lower(), word2.lower()
        
        # Levenshtein distance
        if len(w1) < len(w2):
            w1, w2 = w2, w1
        
        distances = range(len(w2) + 1)
        for i1, c1 in enumerate(w1):
            distances_ = [i1 + 1]
            for i2, c2 in enumerate(w2):
                if c1 == c2:
                    distances_.append(distances[i2])
                else:
                    distances_.append(1 + min((distances[i2], distances[i2 + 1], distances_[-1])))
            distances = distances_
        
        # Convert to similarity score (0-1)
        max_len = max(len(w1), len(w2))
        similarity = 1 - (distances[-1] / max_len)
        
        return similarity


# ============================================================
# MODULE 2: INTELLIGENT RESPONSE ANALYZER
# ============================================================
class IntelligentResponseAnalyzer:
    """
    Enhanced response analysis with contextual understanding.
    
    Combines:
    - Regex pattern matching (fast, reliable for clear responses)
    - Zero-shot classification (handles ambiguous cases)
    - Contextual correction (fixes translation errors)
    - Confidence scoring (multi-method verification)
    """
    
    def __init__(self, classifier_pipeline, context_corrector):
        self.classifier = classifier_pipeline
        self.corrector = context_corrector
    
    def analyze_response(self,
                        question: str,
                        patient_response: str,
                        target_entity: str,
                        expected_context: str) -> Dict:
        """
        Comprehensive response analysis with multi-method approach.
        
        Returns:
            {
                'value': extracted value (True/False/number/None),
                'confidence': 0.0-1.0,
                'method': which method succeeded,
                'evidence': explanation string,
                'needs_clarification': bool
            }
        """
        # Apply contextual correction first
        corrected_response, correction_conf, correction_method = \
            self.corrector.correct_with_context(
                patient_response,
                question,
                expected_context
            )
        
        if correction_method != "no_correction":
            print(f"  📝 Using corrected text: '{corrected_response}'")
        
        text_lower = corrected_response.lower()
        
        result = {
            "value": None,
            "confidence": 0.0,
            "method": "unknown",
            "evidence": "",
            "needs_clarification": False
        }
        
        # ===== VOICE RESPONSIVENESS =====
        if target_entity == "responds_to_voice":
            return self._analyze_voice_response(corrected_response)
        
        # ===== WALKING ABILITY =====
        elif target_entity == "can_walk":
            return self._analyze_walking_ability(corrected_response, text_lower)
        
        # ===== COMMAND OBEDIENCE =====
        elif target_entity == "obeys_commands":
            return self._analyze_command_response(corrected_response, text_lower)
        
        # ===== BLEEDING ASSESSMENT =====
        elif target_entity == "bleeding_severe":
            return self._analyze_bleeding(corrected_response, text_lower)
        
        # ===== RESPIRATORY RATE =====
        elif target_entity == "resp_rate":
            return self._analyze_respiratory_rate(corrected_response, text_lower)
        
        return result
    
    def _analyze_voice_response(self, response: str) -> Dict:
        """Analyze if patient responded to voice"""
        if len(response.strip()) > 0:
            try:
                result = self.classifier(
                    response,
                    candidate_labels=[
                        "affirmative or acknowledgment",
                        "negative or refusal",
                        "confused or unclear",
                        "question or request for help"
                    ],
                    hypothesis_template="This is a {}"
                )
                
                return {
                    "value": True,  # Any verbal response = responsive
                    "confidence": max(0.7, result['scores'][0]),
                    "method": "zero_shot_classification",
                    "evidence": f"Patient verbally responsive: '{response[:50]}...'",
                    "needs_clarification": False
                }
            except:
                return {
                    "value": True,
                    "confidence": 0.6,
                    "method": "length_check",
                    "evidence": "Verbal response detected",
                    "needs_clarification": False
                }
        else:
            return {
                "value": False,
                "confidence": 0.95,
                "method": "silence_detection",
                "evidence": "No verbal response",
                "needs_clarification": False
            }
    
    def _analyze_walking_ability(self, response: str, text_lower: str) -> Dict:
        """
        Analyze walking ability with enhanced pattern matching.
        Uses both regex and AI classification.
        """
        # PHASE 1: High-confidence regex patterns
        strong_yes_patterns = [
            r'\b(yes|yeah|yep|yup)\b.*\b(walk|walking|stand|standing)\b',
            r'\b(i can)\b.*\b(walk|move|stand)\b',
            r'\b(walking|standing|stood up)\b',
        ]
        
        strong_no_patterns = [
            r'\b(no|cannot|can\'t|cant|unable)\b.*\b(walk|stand|move)\b',
            r'\b(leg|ankle|foot|knee).*(broken|fractured|hurt|injured|damaged)\b',
            r'\b(stuck|trapped|pinned|can\'t move)\b',
            r'\b(too (hurt|injured|weak))\b.*\b(walk|stand)\b'
        ]
        
        # Check strong negative patterns first (more specific)
        for pattern in strong_no_patterns:
            if re.search(pattern, text_lower):
                return {
                    "value": False,
                    "confidence": 0.9,
                    "method": "regex_strong_negative",
                    "evidence": f"Strong inability pattern: {pattern}",
                    "needs_clarification": False
                }
        
        # Check strong positive patterns
        for pattern in strong_yes_patterns:
            if re.search(pattern, text_lower):
                return {
                    "value": True,
                    "confidence": 0.9,
                    "method": "regex_strong_positive",
                    "evidence": f"Strong ability pattern: {pattern}",
                    "needs_clarification": False
                }
        
        # PHASE 2: Weaker patterns (lower confidence)
        weak_yes = [r'\b(yes|ok|okay)\b', r'\b(can)\b', r'\b(able)\b']
        weak_no = [r'\b(no|nope)\b', r'\b(hurt|pain)\b', r'\b(difficult|hard)\b']
        
        for pattern in weak_no:
            if re.search(pattern, text_lower):
                # Need AI verification for weak patterns
                break
        else:
            for pattern in weak_yes:
                if re.search(pattern, text_lower):
                    break
        
        # PHASE 3: AI Classification with lowered threshold
        try:
            result = self.classifier(
                response,
                candidate_labels=[
                    "patient can walk or is mobile",
                    "patient cannot walk or is immobile",
                    "patient is struggling or uncertain",
                    "response is unclear or ambiguous"
                ],
                hypothesis_template="The response indicates {}"
            )
            
            # Lowered threshold from 0.5 to 0.35
            if result['scores'][0] > 0.35:
                top_label = result['labels'][0]
                
                if "can walk" in top_label or "mobile" in top_label:
                    return {
                        "value": True,
                        "confidence": result['scores'][0],
                        "method": "zero_shot_classification",
                        "evidence": f"AI detected mobility ({result['scores'][0]:.2f})",
                        "needs_clarification": False
                    }
                
                elif "cannot walk" in top_label or "immobile" in top_label or "struggling" in top_label:
                    return {
                        "value": False,
                        "confidence": result['scores'][0],
                        "method": "zero_shot_classification",
                        "evidence": f"AI detected immobility ({result['scores'][0]:.2f})",
                        "needs_clarification": False
                    }
                
                elif "unclear" in top_label or "ambiguous" in top_label:
                    return {
                        "value": None,
                        "confidence": result['scores'][0],
                        "method": "zero_shot_classification",
                        "evidence": "Response too ambiguous",
                        "needs_clarification": True
                    }
        
        except Exception as e:
            print(f"  ⚠️ Classification error: {e}")
        
        # PHASE 4: Unable to determine
        return {
            "value": None,
            "confidence": 0.0,
            "method": "failed",
            "evidence": "Could not determine walking ability",
            "needs_clarification": True
        }
    
    def _analyze_command_response(self, response: str, text_lower: str) -> Dict:
        """Analyze if patient can follow commands"""
        # Compliance indicators
        comply_patterns = [
            r'\b(yes|okay|ok|squeezing|got it|understand|doing it)\b',
            r'\b(i\'m (squeezing|doing|trying))\b',
        ]
        
        non_comply_patterns = [
            r'\b(no|can\'t|cannot|what|huh|confused|don\'t understand)\b',
            r'\b(thank you|thanks|sorry)\b',  # Polite but not following command
        ]
        
        for pattern in comply_patterns:
            if re.search(pattern, text_lower):
                return {
                    "value": True,
                    "confidence": 0.85,
                    "method": "regex_compliance",
                    "evidence": f"Compliance detected: {pattern}",
                    "needs_clarification": False
                }
        
        for pattern in non_comply_patterns:
            if re.search(pattern, text_lower):
                return {
                    "value": False,
                    "confidence": 0.8,
                    "method": "regex_non_compliance",
                    "evidence": f"Non-compliance detected: {pattern}",
                    "needs_clarification": False
                }
        
        # AI classification fallback
        try:
            result = self.classifier(
                response,
                candidate_labels=[
                    "following the command",
                    "not following or refusing",
                    "confused or unclear"
                ],
                hypothesis_template="The patient is {}"
            )
            
            if result['scores'][0] > 0.4:
                top_label = result['labels'][0]
                
                if "following" in top_label:
                    return {
                        "value": True,
                        "confidence": result['scores'][0],
                        "method": "zero_shot_classification",
                        "evidence": f"AI detected compliance ({result['scores'][0]:.2f})",
                        "needs_clarification": False
                    }
                else:
                    return {
                        "value": False,
                        "confidence": result['scores'][0],
                        "method": "zero_shot_classification",
                        "evidence": f"AI detected non-compliance ({result['scores'][0]:.2f})",
                        "needs_clarification": False
                    }
        except:
            pass
        
        return {
            "value": None,
            "confidence": 0.0,
            "method": "failed",
            "evidence": "Could not determine command compliance",
            "needs_clarification": True
        }
    
    def _analyze_bleeding(self, response: str, text_lower: str) -> Dict:
        """Analyze bleeding severity"""
        # Severe bleeding indicators
        severe_patterns = [
            r'\b(bleeding|blood|hemorrhag).*(bad|severe|heavy|lot|lots|gushing|spurting)\b',
            r'\b(lot of blood|lots of blood|bleeding out)\b',
            r'\b(tourniquet|pressure|can\'t stop)\b',
            r'\b(artery|arterial)\b',
        ]
        
        # Any bleeding mention
        bleeding_patterns = [
            r'\b(bleeding|blood|hemorrhag)\b',
        ]
        
        # Minor indicators
        minor_patterns = [
            r'\b(small|minor|little|scratch|scrape|cut)\b.*\b(bleeding|blood)\b',
            r'\b(bleeding|blood)\b.*(small|minor|little)\b'
        ]
        
        # Check for severe bleeding
        for pattern in severe_patterns:
            if re.search(pattern, text_lower):
                return {
                    "value": True,
                    "confidence": 0.95,
                    "method": "regex_severe_bleeding",
                    "evidence": f"Severe bleeding pattern: {pattern}",
                    "needs_clarification": False
                }
        
        # Check for minor bleeding
        for pattern in minor_patterns:
            if re.search(pattern, text_lower):
                return {
                    "value": False,
                    "confidence": 0.85,
                    "method": "regex_minor_bleeding",
                    "evidence": "Minor bleeding only",
                    "needs_clarification": False
                }
        
        # Check for any bleeding mention (assume severe if not specified as minor)
        for pattern in bleeding_patterns:
            if re.search(pattern, text_lower):
                # Verify with AI classification
                try:
                    result = self.classifier(
                        response,
                        candidate_labels=[
                            "severe or life-threatening bleeding",
                            "minor bleeding or small wound",
                            "no bleeding"
                        ],
                        hypothesis_template="The patient describes {}"
                    )
                    
                    if result['scores'][0] > 0.4:
                        if "severe" in result['labels'][0]:
                            return {
                                "value": True,
                                "confidence": result['scores'][0],
                                "method": "zero_shot_bleeding_severity",
                                "evidence": f"AI classified as severe ({result['scores'][0]:.2f})",
                                "needs_clarification": False
                            }
                        else:
                            return {
                                "value": False,
                                "confidence": result['scores'][0],
                                "method": "zero_shot_bleeding_severity",
                                "evidence": f"AI classified as non-severe ({result['scores'][0]:.2f})",
                                "needs_clarification": False
                            }
                except:
                    pass
                
                # Default: bleeding mentioned = assume severe (safer)
                return {
                    "value": True,
                    "confidence": 0.7,
                    "method": "regex_bleeding_detected",
                    "evidence": "Bleeding mentioned, severity uncertain (defaulting to severe)",
                    "needs_clarification": False
                }
        
        # No bleeding mentioned
        return {
            "value": False,
            "confidence": 0.8,
            "method": "no_bleeding_mentioned",
            "evidence": "No bleeding indicators found",
            "needs_clarification": False
        }
    
    def _analyze_respiratory_rate(self, response: str, text_lower: str) -> Dict:
        """Extract respiratory rate from observation"""
        # Look for numeric patterns
        resp_patterns = [
            r'(\d+)\s*breaths?',
            r'(\d+)\s*per\s*minute',
            r'(\d+)\s*(?:bpm|respirations)',
            r'counted\s*(\d+)',
            r'rate\s*(?:of\s*)?(\d+)',
            r'breathing\s*(\d+)',
        ]
        
        for pattern in resp_patterns:
            match = re.search(pattern, text_lower)
            if match:
                rate = int(match.group(1))
                
                # Validate reasonable range (0-60)
                if 0 <= rate <= 60:
                    return {
                        "value": rate,
                        "confidence": 0.95,
                        "method": "regex_extraction",
                        "evidence": f"Extracted respiratory rate: {rate}/min",
                        "needs_clarification": False
                    }
        
        return {
            "value": None,
            "confidence": 0.0,
            "method": "failed",
            "evidence": "Could not extract respiratory rate",
            "needs_clarification": True
        }


# ============================================================
# MODULE 3: PATIENT ASSESSMENT SESSION MANAGER
# ============================================================
class PatientAssessmentSession:
    """
    Manages the sequential patient assessment dialogue.
    Tracks state, history, and collected data throughout assessment.
    """
    
    def __init__(self, patient_id: Optional[str] = None):
        self.patient_id = patient_id or f"PAT_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Collected patient data
        self.entities = {
            "responds_to_voice": None,
            "can_walk": None,
            "bleeding_severe": None,
            "obeys_commands": None,
            "resp_rate": None,
            "radial_pulse": None,
            "mental_status": None,
            "cap_refill_sec": None,
            "visible_injury": None
        }
        
        # Evidence trail for explainability
        self.evidence = []
        
        # Interaction history
        self.interaction_history = []
        
        # Assessment state
        self.current_step = 0
        self.assessment_complete = False
        
        # SALT-based question sequence
        self.question_sequence = [
            {
                "id": "initial_contact",
                "question": "Can you hear me? If you can hear me, say yes or make a sound.",
                "target": "responds_to_voice",
                "context": "voice_response",
                "critical": True,
                "timeout": 5
            },
            {
                "id": "walking_test",
                "question": "Can you walk? If you can walk, please stand up and take a step.",
                "target": "can_walk",
                "context": "walking_ability",
                "critical": True,
                "timeout": 10,
                "visual_cue": True
            },
            {
                "id": "command_response",
                "question": "Can you squeeze my hand? Squeeze if you understand me.",
                "target": "obeys_commands",
                "context": "command_following",
                "critical": True,
                "timeout": 5
            },
            {
                "id": "injury_check",
                "question": "Where are you hurt? Tell me if you have any bleeding or pain.",
                "target": "bleeding_severe",
                "context": "bleeding_assessment",
                "critical": True,
                "timeout": 10
            },
            {
                "id": "breathing_assessment",
                "question": "I'm going to count your breaths. Just breathe normally for 15 seconds.",
                "target": "resp_rate",
                "context": "breathing_assessment",
                "critical": True,
                "timeout": 20,
                "observational": True
            }
        ]
    
    def get_next_question(self) -> Optional[Dict]:
        """Get next question in sequence"""
        if self.current_step >= len(self.question_sequence):
            self.assessment_complete = True
            return None
        return self.question_sequence[self.current_step]
    
    def record_interaction(self, question: str, response: str, analysis: Dict):
        """Record each interaction"""
        self.interaction_history.append({
            "timestamp": datetime.now().isoformat(),
            "step": self.current_step,
            "question": question,
            "response": response,
            "analysis": analysis
        })
    
    def update_entity(self, entity: str, value, evidence: str):
        """Update patient entity with evidence"""
        self.entities[entity] = value
        self.evidence.append(evidence)
    
    def advance_step(self):
        """Move to next step"""
        self.current_step += 1
    
    def get_completion_percentage(self) -> float:
        """Calculate assessment completion"""
        total = len(self.entities)
        filled = sum(1 for v in self.entities.values() if v is not None)
        return (filled / total) * 100


# ============================================================
# MODULE 4: RANDOM FOREST TRIAGE CLASSIFIER
# ============================================================
class RandomForestTriageClassifier:
    """
    Machine learning classifier for combat triage.
    
    Integrates:
    - Verbal patient data from LLM analysis
    - Sensor data (when available) from medical devices
    - Demographic and situational factors
    
    Outputs:
    - Triage category (Minimal/Delayed/Immediate/Expectant)
    - Criticality score (0-100)
    - Confidence level
    - Feature importance for explainability
    - Transmission package for first responders
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Feature definitions with sensor integration points
        self.feature_names = [
            # === VERBAL ASSESSMENT FEATURES (from LLM) ===
            'responds_to_voice',          # Binary: patient verbal responsiveness
            'can_walk',                   # Binary: mobility assessment
            'bleeding_severe',            # Binary: hemorrhage severity
            'obeys_commands',             # Binary: cognitive function
            
            # === SENSOR DATA FEATURES (hardware integration) ===
            'resp_rate',                  # [SENSOR] Respiratory rate (breaths/min)
            'heart_rate',                 # [SENSOR] Pulse rate (bpm)
            'spo2',                       # [SENSOR] Blood oxygen saturation (%)
            'systolic_bp',                # [SENSOR] Blood pressure (mmHg)
            'cap_refill_sec',             # [SENSOR] Capillary refill time (sec)
            'skin_temp',                  # [SENSOR] Skin temperature (°C)
            
            # === DERIVED FEATURES ===
            'shock_index',                # HR / systolic_bp (shock indicator)
            'gcs_estimate',               # Glasgow Coma Scale (3-15)
            'respiratory_distress',       # Binary: abnormal breathing
            
            # === CONTEXTUAL FEATURES ===
            'age',                        # Patient age (years)
            'mechanism_of_injury'         # Injury type code
        ]
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            # Initialize untrained model
            self.model = RandomForestClassifier(
                n_estimators=100,           # 100 decision trees
                max_depth=12,               # Prevent overfitting
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced',     # Handle class imbalance
                n_jobs=-1                    # Use all CPU cores
            )
    
    def encode_features(self, 
                       entities: Dict, 
                       sensor_data: Dict) -> np.ndarray:
        """
        Convert patient data into feature vector.
        
        SENSOR INTEGRATION POINTS:
        - sensor_data dict contains readings from medical hardware
        - Missing sensor values use physiological defaults
        - System degrades gracefully without sensors (uses verbal data only)
        
        Args:
            entities: Extracted from patient verbal responses
            sensor_data: Direct readings from medical sensors
        
        Returns:
            Feature vector for classification
        """
        features = []
        
# === VERBAL ASSESSMENT FEATURES ===
        features.append(1 if entities.get('responds_to_voice') else 0)
        features.append(1 if entities.get('can_walk') else 0)
        features.append(1 if entities.get('bleeding_severe') else 0)
        features.append(1 if entities.get('obeys_commands') else 0)
        
        # === SENSOR DATA FEATURES (with graceful degradation) ===
        # Respiratory rate: normal 12-20, critical <10 or >30
        resp_rate = sensor_data.get('resp_rate', entities.get('resp_rate', 16))
        features.append(resp_rate if resp_rate else 16)
        
        # Heart rate: normal 60-100, critical <50 or >120
        heart_rate = sensor_data.get('heart_rate', 80)
        features.append(heart_rate)
        
        # SpO2: normal >95%, critical <90%
        spo2 = sensor_data.get('spo2', 95)
        features.append(spo2)
        
        # Systolic BP: normal 90-140, critical <90 or >180
        systolic_bp = sensor_data.get('systolic_bp', 120)
        features.append(systolic_bp)
        
        # Capillary refill: normal <2 sec, critical >3 sec
        cap_refill = sensor_data.get('cap_refill_sec', 2.0)
        features.append(cap_refill)
        
        # Skin temperature: normal 36-37°C
        skin_temp = sensor_data.get('skin_temp', 36.5)
        features.append(skin_temp)
        
        # === DERIVED FEATURES ===
        # Shock Index: HR/SBP (normal <0.7, shock >1.0)
        shock_index = heart_rate / systolic_bp if systolic_bp > 0 else 0.7
        features.append(shock_index)
        
        # GCS estimate from verbal responses
        gcs = 15  # Start with normal
        if not entities.get('responds_to_voice'):
            gcs -= 4  # No verbal response
        if not entities.get('obeys_commands'):
            gcs -= 3  # No motor response
        features.append(gcs)
        
        # Respiratory distress indicator
        resp_distress = 1 if (resp_rate and (resp_rate < 10 or resp_rate > 30)) else 0
        features.append(resp_distress)
        
        # === CONTEXTUAL FEATURES ===
        age = sensor_data.get('age', 30)  # Default adult
        features.append(age)
        
        # Mechanism of injury (0=unknown, 1=blast, 2=GSW, 3=blunt, 4=burn)
        moi = sensor_data.get('mechanism_of_injury', 0)
        features.append(moi)
        
        return np.array(features).reshape(1, -1)
    
    def predict_triage_category(self, 
                               entities: Dict, 
                               sensor_data: Dict = None) -> Dict:
        """
        Predict triage category and generate first responder package.
        
        Returns comprehensive assessment including:
        - Triage category (SALT-based)
        - Criticality score
        - Confidence level
        - Feature importance
        - Transmission package
        """
        if sensor_data is None:
            sensor_data = {}
        
        # Encode features
        X = self.encode_features(entities, sensor_data)
        
        # If model not trained, use rule-based fallback
        if not self.is_trained:
            return self._rule_based_triage(entities, sensor_data, X[0])
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict
        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]
        confidence = max(probabilities)
        
        # Get feature importance
        feature_importance = dict(zip(
            self.feature_names, 
            self.model.feature_importances_
        ))
        
        # Map prediction to triage category
        categories = ['MINIMAL', 'DELAYED', 'IMMEDIATE', 'EXPECTANT']
        category = categories[prediction]
        
        # Calculate criticality score (0-100)
        criticality_score = self._calculate_criticality(
            category, 
            entities, 
            sensor_data, 
            X[0]
        )
        
        # Generate transmission package
        transmission_package = self._generate_transmission_package(
            category,
            criticality_score,
            confidence,
            entities,
            sensor_data,
            feature_importance
        )
        
        return {
            'category': category,
            'criticality_score': criticality_score,
            'confidence': confidence,
            'probabilities': dict(zip(categories, probabilities)),
            'feature_importance': feature_importance,
            'transmission_package': transmission_package
        }
    
    def _rule_based_triage(self, entities: Dict, sensor_data: Dict, features: np.ndarray) -> Dict:
        """
        Rule-based triage fallback when ML model not trained.
        Implements SALT triage algorithm.
        """
        # Extract key indicators
        responsive = entities.get('responds_to_voice', False)
        can_walk = entities.get('can_walk', False)
        bleeding = entities.get('bleeding_severe', False)
        obeys = entities.get('obeys_commands', False)
        resp_rate = features[4]
        
        # SALT Algorithm Implementation
        category = 'DELAYED'
        criticality = 50
        
        # Step 1: Can walk? -> MINIMAL (unless other critical signs)
        if can_walk and not bleeding:
            category = 'MINIMAL'
            criticality = 20
        
        # Step 2: Not walking but responsive and stable -> DELAYED
        elif responsive and obeys and not bleeding:
            category = 'DELAYED'
            criticality = 40
        
        # Step 3: Life-threatening conditions -> IMMEDIATE
        elif bleeding or (resp_rate and (resp_rate < 10 or resp_rate > 30)):
            category = 'IMMEDIATE'
            criticality = 85
        
        # Step 4: Unresponsive with no pulse/breathing -> EXPECTANT
        elif not responsive and not obeys:
            if resp_rate and resp_rate < 6:
                category = 'EXPECTANT'
                criticality = 95
            else:
                category = 'IMMEDIATE'
                criticality = 90
        
        transmission_package = self._generate_transmission_package(
            category,
            criticality,
            0.7,  # Rule-based confidence
            entities,
            sensor_data,
            {}
        )
        
        return {
            'category': category,
            'criticality_score': criticality,
            'confidence': 0.7,
            'probabilities': {},
            'feature_importance': {},
            'transmission_package': transmission_package,
            'method': 'rule_based_SALT'
        }
    
    def _calculate_criticality(self, 
                              category: str, 
                              entities: Dict, 
                              sensor_data: Dict,
                              features: np.ndarray) -> float:
        """
        Calculate 0-100 criticality score for prioritization.
        Higher = more critical
        """
        base_scores = {
            'MINIMAL': 20,
            'DELAYED': 50,
            'IMMEDIATE': 85,
            'EXPECTANT': 95
        }
        
        score = base_scores.get(category, 50)
        
        # Adjust based on vital signs
        resp_rate = features[4]
        heart_rate = features[5]
        shock_index = features[10]
        
        # Respiratory distress
        if resp_rate < 10 or resp_rate > 30:
            score += 10
        
        # Tachycardia or bradycardia
        if heart_rate < 50 or heart_rate > 120:
            score += 8
        
        # Shock
        if shock_index > 1.0:
            score += 12
        
        # Severe bleeding
        if entities.get('bleeding_severe'):
            score += 15
        
        # Unresponsive
        if not entities.get('responds_to_voice'):
            score += 10
        
        # Cap at 100
        return min(score, 100)
    
    def _generate_transmission_package(self,
                                      category: str,
                                      criticality: float,
                                      confidence: float,
                                      entities: Dict,
                                      sensor_data: Dict,
                                      feature_importance: Dict) -> Dict:
        """
        Generate standardized package for first responder transmission.
        
        Includes:
        - Patient ID and timestamp
        - GPS coordinates (if available)
        - Triage category and criticality
        - Vital signs summary
        - Key findings
        - Recommended interventions
        """
        package = {
            'transmission_time': datetime.now().isoformat(),
            'patient_id': sensor_data.get('patient_id', 'UNKNOWN'),
            'location': {
                'latitude': sensor_data.get('gps_lat'),
                'longitude': sensor_data.get('gps_lon'),
                'altitude': sensor_data.get('gps_alt'),
                'accuracy': sensor_data.get('gps_accuracy')
            },
            'triage': {
                'category': category,
                'criticality_score': criticality,
                'confidence': confidence,
                'assessment_method': 'AI_ENHANCED_SALT'
            },
            'vitals': {
                'respiratory_rate': sensor_data.get('resp_rate', entities.get('resp_rate')),
                'heart_rate': sensor_data.get('heart_rate'),
                'spo2': sensor_data.get('spo2'),
                'blood_pressure': sensor_data.get('systolic_bp'),
                'temperature': sensor_data.get('skin_temp')
            },
            'assessment': {
                'responsive': entities.get('responds_to_voice'),
                'ambulatory': entities.get('can_walk'),
                'severe_bleeding': entities.get('bleeding_severe'),
                'follows_commands': entities.get('obeys_commands')
            },
            'interventions_needed': self._recommend_interventions(category, entities),
            'transport_priority': self._calculate_transport_priority(criticality),
            'metadata': {
                'ai_model_version': '2.0',
                'feature_importance': feature_importance,
                'data_completeness': self._calculate_data_completeness(entities, sensor_data)
            }
        }
        
        return package
    
    def _recommend_interventions(self, category: str, entities: Dict) -> List[str]:
        """Recommend immediate interventions based on assessment"""
        interventions = []
        
        if category == 'IMMEDIATE':
            if entities.get('bleeding_severe'):
                interventions.append('APPLY_TOURNIQUET')
                interventions.append('HEMOSTATIC_DRESSING')
            
            interventions.append('ESTABLISH_IV_ACCESS')
            interventions.append('OXYGEN_THERAPY')
            interventions.append('MONITOR_VITALS_CONTINUOUS')
        
        elif category == 'DELAYED':
            interventions.append('WOUND_CARE')
            interventions.append('PAIN_MANAGEMENT')
            interventions.append('MONITOR_VITALS_Q15MIN')
        
        elif category == 'EXPECTANT':
            interventions.append('COMFORT_CARE')
            interventions.append('PAIN_MANAGEMENT_PALLIATIVE')
        
        return interventions
    
    def _calculate_transport_priority(self, criticality: float) -> int:
        """Calculate transport priority (1=highest, 5=lowest)"""
        if criticality >= 85:
            return 1
        elif criticality >= 70:
            return 2
        elif criticality >= 50:
            return 3
        elif criticality >= 30:
            return 4
        else:
            return 5
    
    def _calculate_data_completeness(self, entities: Dict, sensor_data: Dict) -> float:
        """Calculate percentage of data fields populated"""
        total_fields = len(self.feature_names)
        populated = 0
        
        for field in self.feature_names:
            if field in entities and entities[field] is not None:
                populated += 1
            elif field in sensor_data and sensor_data[field] is not None:
                populated += 1
        
        return (populated / total_fields) * 100
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the Random Forest model on historical data"""
        # Scale features
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        self.model.fit(X_scaled, y_train)
        self.is_trained = True
        
        print(f"✓ Model trained on {len(X_train)} samples")
        print(f"  Feature importance:")
        for name, importance in sorted(
            zip(self.feature_names, self.model.feature_importances_),
            key=lambda x: x[1],
            reverse=True
        )[:5]:
            print(f"    {name}: {importance:.3f}")
    
    def save_model(self, path: str):
        """Save trained model to disk"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'is_trained': self.is_trained,
            'feature_names': self.feature_names
        }, path)
        print(f"✓ Model saved to {path}")
    
    def load_model(self, path: str):
        """Load trained model from disk"""
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.is_trained = data['is_trained']
        self.feature_names = data['feature_names']
        print(f"✓ Model loaded from {path}")


# ============================================================
# MODULE 5: AUDIO INTERFACE
# ============================================================
def record_audio(duration: int = 5, sample_rate: int = 16000) -> np.ndarray:
    """
    Record audio from microphone.
    
    Args:
        duration: Recording length in seconds
        sample_rate: Audio sample rate (16kHz for Whisper)
    
    Returns:
        Audio data as numpy array
    """
    if not CONFIG['enable_audio']:
        print("  ⚠️ Audio disabled in config")
        return None
    
    print(f"  🎤 Recording for {duration} seconds...")
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype='float32'
    )
    sd.wait()
    print("  ✓ Recording complete")
    
    return audio.flatten()


def transcribe_audio(audio_data: np.ndarray) -> str:
    """
    Transcribe audio using Whisper ASR.
    
    Args:
        audio_data: Audio samples
    
    Returns:
        Transcribed text
    """
    if audio_data is None:
        return ""
    
    try:
        result = asr(audio_data)
        transcription = result['text'].strip()
        print(f"  📝 Transcription: '{transcription}'")
        return transcription
    
    except Exception as e:
        print(f"  ⚠️ Transcription error: {e}")
        return ""


# ============================================================
# MODULE 6: MAIN ASSESSMENT WORKFLOW
# ============================================================
def conduct_patient_assessment(
    patient_id: Optional[str] = None,
    sensor_data: Optional[Dict] = None,
    use_audio: bool = True
) -> Dict:
    """
    Main workflow for conducting patient assessment.
    
    Args:
        patient_id: Optional patient identifier
        sensor_data: Optional sensor readings
        use_audio: Whether to use audio recording
    
    Returns:
        Complete assessment results with triage recommendation
    """
    print("\n" + "="*80)
    print("INITIATING PATIENT ASSESSMENT")
    print("="*80 + "\n")
    
    # Initialize components
    session = PatientAssessmentSession(patient_id)
    corrector = ContextualTranslationCorrector(classifier)
    analyzer = IntelligentResponseAnalyzer(classifier, corrector)
    triage_classifier = RandomForestTriageClassifier(CONFIG['model_save_path'])
    
    if sensor_data is None:
        sensor_data = {}
    
    # Assessment loop
    while not session.assessment_complete:
        question_data = session.get_next_question()
        
        if question_data is None:
            break
        
        print(f"\n{'─'*80}")
        print(f"STEP {session.current_step + 1}/{len(session.question_sequence)}: {question_data['id'].upper()}")
        print(f"{'─'*80}")
        print(f"\n🗣️  QUESTION: {question_data['question']}\n")
        
        # Get patient response
        if use_audio and CONFIG['enable_audio']:
            audio = record_audio(duration=question_data.get('timeout', 5))
            patient_response = transcribe_audio(audio)
        else:
            # Manual input fallback
            patient_response = input("  💬 Enter patient response: ")
        
        if not patient_response:
            print("  ⚠️ No response detected")
            patient_response = "[NO RESPONSE]"
        
        # Analyze response
        print(f"\n  🔍 Analyzing response...")
        analysis = analyzer.analyze_response(
            question=question_data['question'],
            patient_response=patient_response,
            target_entity=question_data['target'],
            expected_context=question_data['context']
        )
        
        # Display analysis
        print(f"\n  📊 ANALYSIS RESULTS:")
        print(f"     Value: {analysis['value']}")
        print(f"     Confidence: {analysis['confidence']:.2f}")
        print(f"     Method: {analysis['method']}")
        print(f"     Evidence: {analysis['evidence']}")
        
        if analysis['needs_clarification']:
            print(f"     ⚠️ May need clarification")
        
        # Update session
        session.record_interaction(
            question_data['question'],
            patient_response,
            analysis
        )
        
        session.update_entity(
            question_data['target'],
            analysis['value'],
            analysis['evidence']
        )
        
        session.advance_step()
        
        # Progress indicator
        completion = session.get_completion_percentage()
        print(f"\n  📈 Assessment Progress: {completion:.0f}%")
    
    # Generate triage recommendation
    print(f"\n{'='*80}")
    print("GENERATING TRIAGE RECOMMENDATION")
    print(f"{'='*80}\n")
    
    triage_result = triage_classifier.predict_triage_category(
        session.entities,
        sensor_data
    )
    
    # Display results
    print(f"\n🏥 TRIAGE CATEGORY: {triage_result['category']}")
    print(f"⚠️  CRITICALITY SCORE: {triage_result['criticality_score']:.0f}/100")
    print(f"📊 CONFIDENCE: {triage_result['confidence']:.2%}")
    
    if triage_result.get('method'):
        print(f"🔧 METHOD: {triage_result['method']}")
    
    # Save assessment
    output_dir = CONFIG['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(
        output_dir,
        f"{session.patient_id}_assessment.json"
    )
    
    assessment_data = {
        'patient_id': session.patient_id,
        'timestamp': datetime.now().isoformat(),
        'entities': session.entities,
        'evidence': session.evidence,
        'interaction_history': session.interaction_history,
        'triage_result': triage_result,
        'sensor_data': sensor_data
    }
    
    with open(output_file, 'w') as f:
        json.dump(assessment_data, f, indent=2)
    
    print(f"\n💾 Assessment saved to: {output_file}")
    
    # Display transmission package
    print(f"\n{'='*80}")
    print("FIRST RESPONDER TRANSMISSION PACKAGE")
    print(f"{'='*80}\n")
    print(json.dumps(triage_result['transmission_package'], indent=2))
    
    return assessment_data


# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("COMBAT TRIAGE AI SYSTEM v2.0")
    print("Enhanced with Contextual Understanding & Random Forest Classification")
    print("="*80 + "\n")
    
    # Example usage
    print("Starting patient assessment...\n")
    
    # Simulated sensor data (replace with actual sensor readings)
    mock_sensor_data = {
        'patient_id': 'COMBAT_001',
        'gps_lat': 34.0522,
        'gps_lon': -118.2437,
        'heart_rate': 110,
        'spo2': 92,
        'systolic_bp': 95,
        'skin_temp': 36.2,
        'age': 28,
        'mechanism_of_injury': 2  # GSW
    }
    
    # Conduct assessment
    results = conduct_patient_assessment(
        patient_id="COMBAT_001",
        sensor_data=mock_sensor_data,
        use_audio=CONFIG['enable_audio']
    )
    
    print("\n" + "="*80)
    print("ASSESSMENT COMPLETE")
    print("="*80 + "\n")