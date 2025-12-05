#!/usr/bin/env python3
"""
RATS AI TRIAGE CLASSIFIER - MARCH Protocol + SALT Mass Casualty Sorting
========================================================================
Raspberry Pi-optimized combat triage system using:
- MARCH protocol (Individual patient assessment)
- SALT protocol (Mass casualty sorting and prioritization)
- Random Forest classifier with enhanced realistic training
- Real-time audio transcription with contextual understanding
- GPS coordinate transmission to first responders

Author: Jonathon Watson
Version: 3.1
"""

import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# filters out all transformers warnings
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

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
    "whisper_model": "openai/whisper-small.en",  # Faster on Pi
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
print("RATS AI TRIAGE CLASSIFIER - MARCH + SALT PROTOCOL")
print("="*80)

# Load Whisper ASR
print("\n[1/3] Loading Whisper speech recognition...")
asr = pipeline(
    "automatic-speech-recognition",
    model=CONFIG["whisper_model"],
    device=CONFIG["device"],
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
                        "text": "Can you speak clearly? Say your name for me please.",
                        "timeout": 10
                    }
                ]
            },
            {
                "category": "RESPIRATION",
                "priority": 3,
                "questions": [
                    {
                        "id": "breathing_check",
                        "text": "Are you having trouble breathing? Try taking a deep breath then respond.",
                        "timeout": 10
                    }
                ]
            },
            {
                "category": "CIRCULATION",
                "priority": 4,
                "questions": [
                    {
                        "id": "consciousness_check",
                        "text": "Are you feeling dizzy, lightheaded, faint, or unusually weak?",
                        "timeout": 10
                    }
                ]
            },
            {
                "category": "HYPOTHERMIA",
                "priority": 5,
                "questions": [
                    {
                        "id": "temperature_check",
                        "text": "Are you wet, feeling cold, shivering now, or did you recently stop shivering?",
                        "timeout": 10
                    }
                ]
            }
        ]
    
    
    def analyze_march_response(self, 
                        category: str,
                        question_id: str,
                        patient_response: str,
                        question_text: str = "") -> Dict:
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
        text_lower = patient_response.lower().rstrip('.,!?')
        
        # Route to appropriate analyzer
        if category == "MASSIVE_HEMORRHAGE":
            return self._analyze_hemorrhage(patient_response, text_lower, question_text)
        
        elif category == "AIRWAY":
            return self._analyze_airway(patient_response, text_lower, question_text)
        
        elif category == "RESPIRATION":
                return self._analyze_breathing(patient_response, text_lower, question_text)
        
        elif category == "CIRCULATION":
            return self._analyze_circulation(patient_response, text_lower, question_text)
        
        elif category == "HYPOTHERMIA":
            return self._analyze_hypothermia(patient_response, text_lower, question_text)
        
        return {
            'value': None,
            'severity': 0,
            'confidence': 0.0,
            'evidence': 'Unknown category'
    }

    def _analyze_hemorrhage(self, response: str, text_lower: str, question_text: str = "") -> Dict:
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
            r'\b(stopped|controlled|under control|fixed)\b',
            r'\b(tourniquet).*\b(stopped|working|applied)\b',
            r'\b(pressure).*\b(stopped|working|helping)\b',
            r'\b(bleeding).*\b(stopped|slowed|better|fixed)\b',
            r'\b(no longer|not anymore).*\b(bleeding)\b',
            r'\b(bleeding).*\b(under control|controlled|fixed)\b'
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
            r'\b(massive hemorrhage)\b'
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
                context = f"Q: {question_text}\nA: {response}"

                result = self.classifier(
                    context,
                    candidate_labels=[
                        "patient describes massive uncontrolled bleeding that will not stop and is life threatening",
                        "patient describes heavy or severe bleeding that is soaking clothing or the ground",
                        "patient describes moderate bleeding or a wound that is bleeding but somewhat controlled",
                        "patient describes only minor bleeding, a small cut, or a scratch"
                    ]
                )
                
                # Map AI classification to severity
                top_label = result['labels'][0]
                confidence = result['scores'][0]
                label = top_label.lower()
                
                if "massive uncontrolled bleeding" in label or "life threatening" in label:
                    severity = 4
                elif "heavy or severe bleeding" in label:
                    severity = 3
                elif "moderate bleeding" in label:
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
        
        # No bleeding mentioned - but check if patient actually spoke
        if response == "[NO SPEECH DETECTED]" or response == "[NO RESPONSE]":
            return {
                'value': None,
                'severity': 2,
                'confidence': 0.50,
                'evidence': 'Unable to assess - no response from patient'
            }
        
        # No bleeding mentioned
        return {
            'value': False,
            'severity': 0,
            'confidence': 0.85,
            'evidence': 'No bleeding indicators found'
        }

    def _analyze_airway(self, response: str, text_lower: str,  question_text: str = "") -> Dict:
        """
        Analyze airway patency (A in MARCH).
        If patient can speak clearly, airway is patent (open).
        """
        # Check for no speech detected
        if response == "[NO SPEECH DETECTED]" or response == "[NO RESPONSE]":
            return {
                'value': False,
                'severity': 4,
                'confidence': 0.90,
                'evidence': 'No speech detected - potential airway obstruction'
            }
        
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

    def _analyze_breathing(self, response: str, text_lower: str,  question_text: str = "") -> Dict:
        """
        Analyze breathing adequacy (R in MARCH).
        """
        
        if response == "[NO SPEECH DETECTED]" or response == "[NO RESPONSE]":
            return {
                'value': None,
                'severity': 2,
                'confidence': 0.50,
                'evidence': 'Patient unresponsive - unable to assess breathing'
            }
        
        # Respiratory distress patterns
        distress_patterns = [
            r'\b(can\'t breathe|can\'t breath|gasping|choking)\b',
            r'\b(hard to breathe|difficult|struggling)\b',
            r'\b(chest.*tight|chest.*pain)\b'
        ]
        
        # Normal breathing patterns
        normal_patterns = [
            r'\b(breathing fine|breathing okay|no problem)\b',
            r'\b(okay|fine|good)\b',
            r'\b(no|not|nope|nah)\b.*\b(trouble|problem|difficulty|hard|struggling)\b',
            r'\b(no)\s+(i\'m|im|i am)\s+(not)\b',
            r'\b(no|nope|nah)\b'
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
            context = f"Q: {question_text}\nA: {response}"

            result = self.classifier(
                context,
                candidate_labels=[
                    "patient is struggling to breathe, gasping, or can barely speak",
                    "patient has some shortness of breath or mild trouble breathing but can still talk",
                    "patient says they are breathing comfortably with no trouble"
                ]
            )

            top_label = result['labels'][0]
            top_score = result['scores'][0]
            label = top_label.lower()

            if "struggling to breathe" in label or "gasping" in label or "can barely speak" in label:
                severity = 3
                adequate = False
            elif "shortness of breath" in label or "mild trouble breathing" in label:
                severity = 2
                adequate = False
            else:
                severity = 0
                adequate = True

            return {
                'value': adequate,
                'severity': severity,
                'confidence': top_score,
                'evidence': f'AI assessment: {top_label}'
            }
        except:
            pass
        
        return {
            'value': None,
            'severity': 2,
            'confidence': 0.50,
            'evidence': 'Unable to assess breathing adequacy'
        }

    def _analyze_circulation(self, response: str, text_lower: str,  question_text: str = "") -> Dict:
        """
        Analyze circulation adequacy (C in MARCH).

        Uses:
        - Presence/absence of response
        - Explicit symptom patterns (dizzy, faint, weak, shock)
        - Zero-shot classification fallback
        """

        # No speech or no response
        if response == "[NO SPEECH DETECTED]" or response == "[NO RESPONSE]":
            return {
                'value': False,
                'severity': 4,
                'confidence': 0.90,
                'evidence': 'No response to circulation question - possible severe compromise'
            }

        # Words that indicate symptoms at all
        symptom_words = r"(dizzy|dizziness|lightheaded|faint|weak)"

        # 1) Clear denials: no symptoms at all
        denial_patterns = [
            r'^\s*(no|nope|nah)[\.\!\?]*\s*$',  # "no.", "no!", etc.
            r'^\s*(no|nope|nah)[,\s]+(no|nope|nah)',  # "no, no", "no no no"
            r'\b(no|not|don\'t)\s+(feel|feeling)?\s*(dizzy|lightheaded|faint|weak)\b',
        ]

        for pattern in denial_patterns:
            if re.search(pattern, text_lower):
                return {
                    'value': True,
                    'severity': 0,
                    'confidence': 0.85,
                    'evidence': 'Patient denies dizziness/faintness/weakness'
                }

        # 2) Mild / qualified symptoms FIRST
        mild_patterns = [
            r'\b(a little|a bit|slightly|kinda|kind of|somewhat)\s+(dizzy|lightheaded|faint|weak)\b',
            r'\b(dizzy|lightheaded|faint|weak)\s+but\s+(okay|fine|alright|i\'m okay|im okay|not too bad)\b'
        ]

        for pattern in mild_patterns:
            if re.search(pattern, text_lower):
                return {
                    'value': False,
                    'severity': 2,  # mild/moderate issue
                    'confidence': 0.80,
                    'evidence': 'Patient reports mild dizziness/weakness but says they are managing'
                }

        # 3) Clear severe distress patterns ONLY
        severe_patterns = [
            r'\b(about to pass out|going to pass out|almost passed out|nearly passed out)\b',
            r'\b(pass(ed)? out|black(ed)? out|blacking out)\b',
            r'\b(very dizzy|so dizzy|really dizzy)\b',
            r'\b(very weak|unusually weak|so weak|too weak)\b',
            r'\b(cold and clammy|clammy)\b',
            r'\b(in shock|feel in shock|going into shock)\b',
            r'\b(can\'t stand|cannot stand|can\'t stay upright)\b'
        ]

        for pattern in severe_patterns:
            if re.search(pattern, text_lower):
                return {
                    'value': False,
                    'severity': 3,
                    'confidence': 0.90,
                    'evidence': 'Patient reports severe dizziness/weakness or shock suggesting poor circulation'
                }

        # 4) "I’m okay" ONLY counts as normal if no symptom words at all
        normal_patterns = [
            r'\b(i(\'m| am)?\s*(fine|okay|ok|normal|alright))\b',
            r'\b(feel|feeling)\s*(fine|okay|normal|alright)\b'
        ]

        if not re.search(symptom_words, text_lower):
            for pattern in normal_patterns:
                if re.search(pattern, text_lower):
                    return {
                        'value': True,
                        'severity': 0,
                        'confidence': 0.85,
                        'evidence': 'Patient feels normal and reports no circulation concerns'
                    }

        # 5) AI classification fallback for everything else
        try:
            context = f"Q: {question_text}\nA: {response}"

            result = self.classifier(
                context,
                candidate_labels=[
                    "patient feels very dizzy or faint, may pass out, or feels in shock",
                    "patient feels somewhat dizzy, lightheaded, or unusually weak but is still responsive",
                    "patient denies dizziness, faintness, or unusual weakness and feels normal"
                ]
            )

            top_label = result['labels'][0]
            top_score = float(result['scores'][0])
            label = top_label.lower()

            if top_score < 0.55:
                return {
                    'value': None,
                    'severity': 2,
                    'confidence': top_score,
                    'evidence': f'Low-confidence AI assessment (circulation): {top_label}'
                }

            if "very dizzy or faint" in label or "in shock" in label or "may pass out" in label:
                severity = 3
                adequate = False
            elif "somewhat dizzy" in label or "lightheaded" in label or "unusually weak" in label:
                severity = 2
                adequate = False
            else:
                severity = 0
                adequate = True

            return {
                'value': adequate,
                'severity': severity,
                'confidence': top_score,
                'evidence': f'AI assessment (circulation): {top_label}'
            }
        except:
            pass
        
        return {
            'value': None,
            'severity': 2,
            'confidence': 0.50,
            'evidence': 'Unable to assess circulation adequacy'
        }

    def _analyze_hypothermia(self, response: str, text_lower: str,  question_text: str = "") -> Dict:
        """
        Analyze hypothermia risk (H in MARCH) based on:
        - wet exposure
        - feeling cold
        - active shivering
        - recently stopped shivering (late-stage danger)
        - zero-shot fallback when patterns are unclear
        """

        import re  # ensure re is imported

        # No speech fallback
        if response in ["[NO SPEECH DETECTED]", "[NO RESPONSE]"]:
            return {
                'value': None,
                'severity': 1,
                'confidence': 0.50,
                'evidence': 'Patient unresponsive - unable to assess hypothermia'
            }

        # WET patterns
        wet_patterns = [
            r'\b(wet|soaked|drenched|soaking|soaked through)\b',
            r'\b(clothes (are )?wet|uniform (is )?wet)\b',
            r'\b(lying in (water|snow|mud)|fell in (water|river|lake|stream))\b',
            r'\b(rained on|got caught in the rain)\b'
        ]

        # COLD patterns
        cold_patterns = [
            r'\b(feel(ing)? cold|very cold|so cold|really cold)\b',
            r'\b(freezing|freezin|ice cold|frozen|chilled to the bone)\b',
            r'\b(chilled|super cold|crazy cold)\b'
        ]

        # SHIVERING patterns
        shivering_patterns = [
            r'\b(shivering|shiverin|shivers)\b',
            r'\b(shaking from cold|shaking because.*cold)\b',
            r'\b(teeth (are )?chattering|teeth chatter(ing)?)\b',
            r'\b(can(\'t|not) stop shaking|trembling from cold)\b'
        ]

        # STOPPED SHIVERING patterns
        stopped_shivering_patterns = [
            r'\b(stopped shivering|not shivering anymore)\b',
            r'\b(used to shiver|was shivering and (then )?stopped)\b',
            r'\b(i was shivering but (now )?i\'?m not)\b'
        ]

        # Explicit denials
        no_hypo_patterns = [
            r'\b(not cold|don\'t feel cold|do not feel cold)\b',
            r'\b(no shivering|not shivering)\b',
            r'\b(i(\'m| am)?\s*(dry|pretty dry|mostly dry))\b',
            r'\b(clothes (are )?dry|i\'m dry not wet)\b',
            r'^\s*(no|nope|nah)\s*$'
        ]

        # Check denials first
        for pattern in no_hypo_patterns:
            if re.search(pattern, text_lower):
                return {
                    'value': False,
                    'severity': 0,
                    'confidence': 0.80,
                    'evidence': 'Patient denies being cold, shivering, or wet'
                }

        # Detect factors
        is_wet = any(re.search(p, text_lower) for p in wet_patterns)
        is_cold = any(re.search(p, text_lower) for p in cold_patterns)
        is_shivering = any(re.search(p, text_lower) for p in shivering_patterns)
        stopped_shivering = any(re.search(p, text_lower) for p in stopped_shivering_patterns)

        # Late-stage hypothermia
        if stopped_shivering:
            return {
                'value': True,
                'severity': 4,
                'confidence': 0.90,
                'evidence': 'Patient reports they recently stopped shivering'
            }

        # Active shivering
        if is_shivering:
            ev = 'Patient reporting active shivering'
            if is_cold:
                ev += ' and feeling cold'
            if is_wet:
                ev += ' while wet'
            return {
                'value': True,
                'severity': 3,
                'confidence': 0.85,
                'evidence': ev
            }

        # Cold + wet
        if is_cold and is_wet:
            return {
                'value': True,
                'severity': 3,
                'confidence': 0.80,
                'evidence': 'Patient is wet and feeling cold'
            }

        # Cold only
        if is_cold:
            return {
                'value': True,
                'severity': 2,
                'confidence': 0.75,
                'evidence': 'Patient reports feeling cold'
            }

        # Wet only
        if is_wet:
            return {
                'value': True,
                'severity': 1,
                'confidence': 0.70,
                'evidence': 'Patient is wet, increasing hypothermia risk'
            }

        # Zero-shot fallback
        try:
            context = f"Q: {question_text}\nA: {response}"

            result = self.classifier(
                context,
                candidate_labels=[
                    "patient reports they were shivering before and have now stopped or feel extremely cold and confused",
                    "patient reports they are shivering, very cold, or soaked and cannot warm up",
                    "patient mentions feeling slightly cold or damp but says they are mostly okay",
                    "patient denies feeling cold, denies being wet, and denies shivering"
                ]
            )

            top_label = result['labels'][0]
            top_score = result['scores'][0]
            label = top_label.lower()

            if "have now stopped" in label or "extremely cold and confused" in label:
                severity = 4
                risk = True
            elif "shivering, very cold" in label or "cannot warm up" in label:
                severity = 3
                risk = True
            elif "slightly cold" in label or "mostly okay" in label:
                severity = 1
                risk = True
            elif "denies feeling cold" in label or "denies being wet" in label:
                severity = 0
                risk = False
            else:
                # generic mild risk if it falls through
                severity = 2
                risk = True

            return {
                'value': risk,
                'severity': severity,
                'confidence': float(top_score),
                'evidence': f'AI assessment (hypothermia): {top_label}'
            }

        except Exception:
            pass

        # Unclear fallback
        return {
            'value': None,
            'severity': 1,
            'confidence': 0.50,
            'evidence': 'Hypothermia status unclear from response'
        }



# ============================================================
# MODULE 2: SALT MASS CASUALTY SORTING
# ============================================================
class SALTMassCasualtySorting:
    """
    SALT Protocol for Mass Casualty Incident (MCI) sorting.
    
    SALT = Sort, Assess, Lifesaving interventions, Treatment/Transport
    
    Used to prioritize multiple patients after individual MARCH assessments.
    This class takes a list of MARCH-assessed patients and applies SALT sorting.
    """
    
    SALT_CATEGORIES = {
        'IMMEDIATE': {
            'color': 'RED',
            'priority': 1,
            'description': 'Life-threatening injuries requiring immediate intervention'
        },
        'DELAYED': {
            'color': 'YELLOW',
            'priority': 2,
            'description': 'Serious injuries but can wait for treatment'
        },
        'MINIMAL': {
            'color': 'GREEN',
            'priority': 3,
            'description': 'Minor injuries, can walk and wait'
        },
        'EXPECTANT': {
            'color': 'BLACK',
            'priority': 4,
            'description': 'Injuries incompatible with life given available resources'
        },
        'DEAD': {
            'color': 'BLACK',
            'priority': 5,
            'description': 'Deceased'
        }
    }
    
    def __init__(self):
        pass
    
    def sort_patients(self, patient_list: List[Dict]) -> List[Dict]:
        """
        Apply SALT sorting to a list of MARCH-assessed patients.
        
        Args:
            patient_list: List of patient assessment dicts with MARCH results
        
        Returns:
            Sorted list with SALT categories and priority rankings
        """
        print("\n" + "="*80)
        print("SALT MASS CASUALTY SORTING - ANALYZING MULTIPLE PATIENTS")
        print("="*80 + "\n")
        
        sorted_patients = []
        
        # Step 1: SORT - Global sorting
        print("📊 SALT STEP 1: GLOBAL SORTING")
        print("─" * 80)
        
        for idx, patient in enumerate(patient_list):
            patient_id = patient.get('patient_id', f'UNKNOWN_{idx}')
            march_results = patient.get('march_results', {})
            triage_result = patient.get('triage_result', {})
            
            # Apply SALT logic
            salt_category = self._determine_salt_category(
                march_results,
                triage_result
            )
            
            # Calculate SALT priority score (lower = higher priority)
            priority_score = self._calculate_priority_score(
                march_results,
                triage_result,
                salt_category
            )
            
            sorted_patient = {
                **patient,
                'salt_category': salt_category,
                'salt_color': self.SALT_CATEGORIES[salt_category]['color'],
                'salt_priority_score': priority_score,
                'salt_description': self.SALT_CATEGORIES[salt_category]['description']
            }
            
            sorted_patients.append(sorted_patient)
            
            print(f"  Patient {patient_id}:")
            print(f"    MARCH Category: {triage_result.get('category', 'UNKNOWN')}")
            print(f"    SALT Category:  {salt_category} ({self.SALT_CATEGORIES[salt_category]['color']})")
            print(f"    Priority Score: {priority_score:.2f}")
            print()
        
        # Step 2: Rank by priority
        sorted_patients.sort(key=lambda x: x['salt_priority_score'])
        
        print("="*80)
        print("SALT FINAL PRIORITY RANKING (Highest → Lowest)")
        print("="*80 + "\n")
        
        for rank, patient in enumerate(sorted_patients, 1):
            patient['salt_rank'] = rank
            print(f"  #{rank} - {patient.get('patient_id')} - "
                  f"{patient['salt_category']} ({patient['salt_color']}) - "
                  f"Score: {patient['salt_priority_score']:.2f}")
        
        print("\n" + "="*80 + "\n")
        
        return sorted_patients
    
    def _determine_salt_category(self, 
                                  march_results: Dict,
                                  triage_result: Dict) -> str:
        """
        Determine SALT category based on MARCH assessment.
        
        SALT Decision Logic:
        1. Can walk? → MINIMAL (unless other severe injuries)
        2. Obeys commands/purposeful movement? → Assess further
        3. Lifesaving interventions likely to help? → IMMEDIATE
        4. Injuries incompatible with survival? → EXPECTANT
        """
        
        # Extract MARCH severities
        hemorrhage_sev = march_results.get('MASSIVE_HEMORRHAGE', {}).get('severity', 0)
        airway_sev = march_results.get('AIRWAY', {}).get('severity', 0)
        respiration_sev = march_results.get('RESPIRATION', {}).get('severity', 0)
        circulation_sev = march_results.get('CIRCULATION', {}).get('severity', 0)
        hypothermia_sev = march_results.get('HYPOTHERMIA', {}).get('severity', 0)
        
        criticality = triage_result.get('criticality_score', 0)
        
        # SALT Logic Tree
        
        # EXPECTANT: Multiple critical failures or MARCH says EXPECTANT
        if triage_result.get('category') == 'EXPECTANT':
            return 'EXPECTANT'
        
        if (airway_sev >= 4 and circulation_sev >= 4) or \
           (sum([hemorrhage_sev >= 4, airway_sev >= 4, circulation_sev >= 4]) >= 2):
            return 'EXPECTANT'
        
        # IMMEDIATE: Any life-threatening component
        if hemorrhage_sev >= 3 or airway_sev >= 3 or respiration_sev >= 3 or circulation_sev >= 3:
            return 'IMMEDIATE'
        
        if criticality >= 75:
            return 'IMMEDIATE'
        
        # DELAYED: Moderate injuries
        if hemorrhage_sev >= 2 or respiration_sev >= 2 or circulation_sev >= 2:
            return 'DELAYED'
        
        if 40 <= criticality < 75:
            return 'DELAYED'
        
        # MINIMAL: Minor injuries
        return 'MINIMAL'
    
    def _calculate_priority_score(self,
                                   march_results: Dict,
                                   triage_result: Dict,
                                   salt_category: str) -> float:
        """
        Calculate numerical priority score for fine-grained sorting within SALT categories.
        Lower score = higher priority.
        
        Score components:
        - Base score from SALT category
        - MARCH severity modifiers
        - Criticality score
        - Time-sensitive factors
        """
        
        # Base scores by category
        base_scores = {
            'IMMEDIATE': 10.0,
            'DELAYED': 50.0,
            'MINIMAL': 80.0,
            'EXPECTANT': 95.0,
            'DEAD': 100.0
        }
        
        score = base_scores.get(salt_category, 50.0)
        
        # Modifier from criticality (inverse - higher criticality = lower score)
        criticality = triage_result.get('criticality_score', 50)
        score -= (criticality / 100.0) * 10.0
        
        # Modifier from MARCH component severities
        hemorrhage_sev = march_results.get('MASSIVE_HEMORRHAGE', {}).get('severity', 0)
        airway_sev = march_results.get('AIRWAY', {}).get('severity', 0)
        respiration_sev = march_results.get('RESPIRATION', {}).get('severity', 0)
        
        # Massive hemorrhage gets highest priority within category
        if hemorrhage_sev >= 3:
            score -= 5.0
        
        # Airway issues are time-critical
        if airway_sev >= 3:
            score -= 4.0
        
        # Respiratory distress
        if respiration_sev >= 3:
            score -= 3.0
        
        return max(0.0, score)  # Never negative
    
    def generate_salt_report(self, sorted_patients: List[Dict]) -> str:
        """
        Generate a formatted SALT mass casualty report.
        """
        report = []
        report.append("\n" + "="*80)
        report.append("SALT MASS CASUALTY INCIDENT REPORT")
        report.append("="*80 + "\n")
        
        # Count by category
        category_counts = {cat: 0 for cat in self.SALT_CATEGORIES.keys()}
        for patient in sorted_patients:
            cat = patient.get('salt_category', 'UNKNOWN')
            if cat in category_counts:
                category_counts[cat] += 1
        
        report.append("PATIENT COUNT BY CATEGORY:")
        report.append("─" * 80)
        for cat, info in self.SALT_CATEGORIES.items():
            count = category_counts.get(cat, 0)
            report.append(f"  {info['color']:8} ({cat:10}): {count:3} patients")
        
        report.append(f"\n  TOTAL PATIENTS: {len(sorted_patients)}")
        report.append("\n" + "="*80 + "\n")
        
        # Priority list
        report.append("TREATMENT PRIORITY ORDER:")
        report.append("="*80 + "\n")
        
        for patient in sorted_patients:
            rank = patient.get('salt_rank', '?')
            patient_id = patient.get('patient_id', 'UNKNOWN')
            cat = patient.get('salt_category', 'UNKNOWN')
            color = patient.get('salt_color', '?')
            score = patient.get('salt_priority_score', 0)
            march = patient.get('triage_result', {}).get('march_summary', 'N/A')
            
            report.append(f"#{rank:2} | {color:6} | {patient_id:15} | Score: {score:5.1f}")
            report.append(f"     MARCH: {march}")
            report.append("")
        
        report.append("="*80 + "\n")
        
        return "\n".join(report)


# ============================================================
# MODULE 3: RANDOM FOREST MARCH CLASSIFIER (ENHANCED TRAINING)
# ============================================================
class MARCHRandomForestClassifier:
    """
    Random Forest classifier for MARCH-based triage.
    
    Features:
    - MARCH component scores (from patient responses)
    - Sensor data (heart rate, respiration, temperature)
    - Combined criticality prediction
    
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

        # Map triage category → transport priority (1 = highest)
        if category == "IMMEDIATE":
            transport_priority = 1
        elif category == "DELAYED":
            transport_priority = 2
        elif category == "MINIMAL":
            transport_priority = 3
        else:
            # EXPECTANT / DEAD
            transport_priority = 4

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
            'transport_priority': transport_priority
        }
    
    def _recommend_march_interventions(self, march_summary: str) -> List[str]:
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
    
    def train_on_synthetic_data(self, n_samples: int = 5000):
        """
        ENHANCED: Generate realistic synthetic training data with variation.
        
        Improvements over hard-coded rules:
        - Probabilistic decision boundaries (not hard cutoffs)
        - Contextual factors (age, comorbidities, resource constraints)
        - Conflicting signals and edge cases
        - Realistic noise in measurements
        - Multi-factor interactions
        - Outcome-based labeling with variation
        """
        print(f"\n🔧 ENHANCED TRAINING: Generating {n_samples} realistic samples...")
        print("    ✓ Probabilistic boundaries")
        print("    ✓ Contextual factors")
        print("    ✓ Conflicting signals")
        print("    ✓ Measurement noise")
        
        X_train = []
        y_train = []
        
        for _ in range(n_samples):
            # ═══════════════════════════════════════════════════════
            # STEP 1: Generate MARCH severities with realistic distributions
            # ═══════════════════════════════════════════════════════
            
            # Most patients have no/minor hemorrhage, few have severe
            hemorrhage = np.random.choice([0, 0, 0, 0, 1, 2, 3, 4], 
                                         p=[0.50, 0.20, 0.10, 0.05, 0.08, 0.04, 0.02, 0.01])
            
            # Airway usually patent unless severe trauma
            airway = np.random.choice([0, 0, 0, 0, 1, 3, 4], 
                                     p=[0.75, 0.10, 0.08, 0.03, 0.02, 0.01, 0.01])
            
            # Respiration varies more
            respiration = np.random.choice([0, 0, 0, 1, 2, 3], 
                                          p=[0.55, 0.20, 0.10, 0.08, 0.05, 0.02])
            
            # Circulation problems common in trauma
            circulation = np.random.choice([0, 0, 0, 1, 2, 3, 4], 
                                          p=[0.50, 0.20, 0.10, 0.10, 0.05, 0.03, 0.02])
            
            # Hypothermia depends on environment
            hypothermia = np.random.choice([0, 0, 0, 1, 2], 
                                          p=[0.65, 0.20, 0.10, 0.04, 0.01])
            
            # ═══════════════════════════════════════════════════════
            # STEP 2: Add contextual factors that affect outcomes
            # ═══════════════════════════════════════════════════════
            
            # Age affects resilience
            age = np.random.choice(['young', 'adult', 'elderly'], p=[0.15, 0.65, 0.20])
            age_modifier = {'young': 0.8, 'adult': 1.0, 'elderly': 1.3}[age]
            
            # Pre-existing conditions
            has_comorbidity = np.random.choice([False, True], p=[0.75, 0.25])
            comorbidity_modifier = 1.2 if has_comorbidity else 1.0
            
            # Time since injury (earlier = better outcomes)
            time_factor = np.random.uniform(0.9, 1.3)  # 0.9 = recent, 1.3 = delayed
            
            # Combined risk multiplier
            risk_multiplier = age_modifier * comorbidity_modifier * time_factor
            
            # ═══════════════════════════════════════════════════════
            # STEP 3: Generate vitals with realistic correlations + NOISE
            # ═══════════════════════════════════════════════════════
            
            # Heart rate correlates with hemorrhage/circulation but has variance
            if hemorrhage >= 3 or circulation >= 3:
                base_hr = np.random.randint(110, 150)
            elif hemorrhage >= 2 or circulation >= 2:
                base_hr = np.random.randint(95, 115)
            elif hemorrhage >= 1 or circulation >= 1:
                base_hr = np.random.randint(85, 105)
            else:
                base_hr = np.random.randint(60, 90)
            
            # Add measurement noise and individual variation
            hr_noise = np.random.randint(-10, 15)
            heart_rate = np.clip(base_hr + hr_noise, 40, 180)
            
            # Respiratory rate - can be high OR low in distress
            if respiration >= 3:
                # Critical: either bradypnea or tachypnea
                resp_rate = np.random.choice([
                    np.random.randint(5, 10),   # Too slow
                    np.random.randint(32, 42)   # Too fast
                ])
            elif respiration >= 2:
                resp_rate = np.random.randint(22, 32)
            elif respiration >= 1:
                resp_rate = np.random.randint(18, 24)
            else:
                resp_rate = np.random.randint(12, 20)
            
            # Add noise
            resp_rate = np.clip(resp_rate + np.random.randint(-2, 3), 4, 50)
            
            # Body temperature
            if hypothermia >= 2:
                body_temp = np.random.uniform(31, 34.5)
            elif hypothermia >= 1:
                body_temp = np.random.uniform(34.5, 36)
            else:
                body_temp = np.random.uniform(36, 37.8)
            
            # Add measurement noise
            body_temp += np.random.uniform(-0.3, 0.3)
            body_temp = np.clip(body_temp, 30, 42)
            
            # ═══════════════════════════════════════════════════════
            # STEP 4: Calculate derived features
            # ═══════════════════════════════════════════════════════
            
            total_score = hemorrhage + airway + respiration + circulation + hypothermia
            critical_count = sum(1 for sev in [hemorrhage, airway, respiration, 
                                               circulation, hypothermia] if sev >= 3)
            
            # Shock index (HR / SBP proxy) - higher = worse
            shock_index_proxy = heart_rate / 100.0  # Simplified
            
            # ═══════════════════════════════════════════════════════
            # STEP 5: Determine triage category with PROBABILISTIC logic
            # ═══════════════════════════════════════════════════════
            
            # Base category from MARCH
            if airway >= 4 or (circulation >= 4 and hemorrhage >= 3):
                base_category = 3  # EXPECTANT
            elif hemorrhage >= 3 or airway >= 3 or respiration >= 3 or circulation >= 3:
                base_category = 2  # IMMEDIATE
            elif total_score >= 5 or critical_count >= 1:
                base_category = 1  # DELAYED
            else:
                base_category = 0  # MINIMAL
            
            # ═══════════════════════════════════════════════════════
            # STEP 6: Apply contextual modifiers and boundary fuzziness
            # ═══════════════════════════════════════════════════════
            
            # Add probabilistic variation at boundaries
            if base_category == 2 and risk_multiplier > 1.2:
                # High-risk IMMEDIATE might become EXPECTANT
                if np.random.random() < 0.15:  # 15% chance
                    base_category = 3
            
            if base_category == 1 and risk_multiplier < 0.9:
                # Low-risk DELAYED might become MINIMAL
                if np.random.random() < 0.20:  # 20% chance
                    base_category = 0
            
            if base_category == 2 and risk_multiplier < 0.95 and critical_count == 1:
                # Borderline IMMEDIATE might be DELAYED if low risk
                if np.random.random() < 0.25:  # 25% chance
                    base_category = 1
            
            # Vital signs can override MARCH assessment
            if heart_rate > 140 and resp_rate > 35 and base_category < 2:
                # Severe vital derangement → upgrade to IMMEDIATE
                if np.random.random() < 0.40:
                    base_category = 2
            
            # Conflicting signals: high MARCH but stable vitals
            if total_score >= 8 and 60 <= heart_rate <= 100 and 12 <= resp_rate <= 20:
                # Maybe assessment was overcautious
                if np.random.random() < 0.15:
                    base_category = max(0, base_category - 1)
            
            category = base_category
            
            # ═══════════════════════════════════════════════════════
            # STEP 7: Store sample
            # ═══════════════════════════════════════════════════════
            
            features = [
                hemorrhage, airway, respiration, circulation, hypothermia,
                heart_rate, resp_rate, body_temp,
                total_score, critical_count
            ]
            
            X_train.append(features)
            y_train.append(category)
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # ═══════════════════════════════════════════════════════
        # STEP 8: Train model
        # ═══════════════════════════════════════════════════════
        
        print(f"\n🎯 Training Random Forest on {n_samples} realistic samples...")
        print(f"   Distribution: MINIMAL={np.sum(y_train==0)}, DELAYED={np.sum(y_train==1)}, "
              f"IMMEDIATE={np.sum(y_train==2)}, EXPECTANT={np.sum(y_train==3)}")
        
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
        
        print(f"\n✓ Model trained successfully with enhanced realism!")
    
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
# MODULE 4: AUDIO INTERFACE
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
        result = asr(audio_data)
        transcription = result['text'].strip()
        
        # Filter out Whisper hallucinations
        # 1. Common hallucination phrases (from YouTube training data)
        hallucination_phrases = [
            'thank you', 'thanks for watching', 'please subscribe', 
            'bye', 'goodbye', 'thank you for watching', 'thanks',
            'good morning', 'good night', 'good evening'
        ]
        
        transcription_lower = transcription.lower().strip('.,!?')

        # Check if transcription is EXACTLY one of the hallucination phrases
        if transcription_lower in hallucination_phrases:
            transcription = "[NO SPEECH DETECTED]"
        
        # 2. Excessive repetition (same word or tiny vocabulary repeated many times)
        words = transcription.split()
        if len(words) > 15:  # only analyze repetition if the transcript is LONG
            # Normalize words (lowercase, strip punctuation)
            norm_words = [
                w.lower().strip('.,!?')
                for w in words
                if w.strip('.,!?')  # remove pure punctuation
            ]

            unique_norm_words = set(norm_words)

            # Case A: tiny vocabulary repeated for a long transcript
            # e.g., "good morning good morning good morning ..."
            if len(unique_norm_words) <= 5:
                transcription = "[NO SPEECH DETECTED]"
        
            # Case B: one word dominates 95%+ of the transcript
            else:
                word_counts = {}
                for w in norm_words:
                    word_counts[w] = word_counts.get(w, 0) + 1
                
                max_count = max(word_counts.values()) if word_counts else 0
                if max_count > len(norm_words) * 0.95:
                    transcription = "[NO SPEECH DETECTED]"

        
        # 3. Excessive dots/periods (silence hallucination)
        elif len(transcription) > 50 and transcription.count('.') > len(transcription) * 0.5:
            transcription = "[NO SPEECH DETECTED]"
        
        # 4. Only dots and spaces
        elif transcription.replace('.', '').replace(' ', '') == '':
            transcription = "[NO SPEECH DETECTED]"
        
        print(f"  📝 Transcription: '{transcription}'")
        return transcription
    except Exception as e:
        print(f"  ⚠️ Transcription error: {e}")
        return ""


# ============================================================
# MODULE 5: MAIN TRIAGE WORKFLOW
# ============================================================
def conduct_march_triage(
    patient_id: Optional[str] = None,
    sensor_data: Optional[Dict] = None,
    use_audio: bool = True
) -> Dict:
    """
    Main MARCH protocol triage workflow for INDIVIDUAL patient.
    
    Returns complete assessment with GPS and MARCH summary.
    """
    print("\n" + "="*80)
    print("RATS AI TRIAGE - MARCH PROTOCOL ASSESSMENT (INDIVIDUAL)")
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
        march_classifier.train_on_synthetic_data(n_samples=5000)
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
                question['text']
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


def apply_salt_sorting(patient_assessments: List[Dict]) -> List[Dict]:
    """
    Apply SALT protocol to sort multiple MARCH-assessed patients.
    
    This is used for MASS CASUALTY situations after individual assessments.
    
    Args:
        patient_assessments: List of patient assessment dicts from conduct_march_triage()
    
    Returns:
        SALT-sorted patient list with priority rankings
    """
    salt_sorter = SALTMassCasualtySorting()
    sorted_patients = salt_sorter.sort_patients(patient_assessments)
    
    # Generate and print SALT report
    report = salt_sorter.generate_salt_report(sorted_patients)
    print(report)
    
    # Save SALT report
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    report_file = os.path.join(
        CONFIG['output_dir'],
        f"SALT_MCI_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )
    
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"💾 SALT report saved to: {report_file}")
    
    # Save sorted patient data
    json_file = report_file.replace('.txt', '.json')
    with open(json_file, 'w') as f:
        json.dump(sorted_patients, f, indent=2)
    
    return sorted_patients


# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("RATS AI TRIAGE CLASSIFIER v3.1")
    print("MARCH Individual Assessment + SALT Mass Casualty Sorting")
    print("="*80 + "\n")
    
    print("USAGE OPTIONS:")
    print("  1. Single patient MARCH assessment")
    print("  2. Multiple patients + SALT mass casualty sorting")
    print("="*80 + "\n")
    
    # Example: Single patient assessment
    mock_sensor_data = {
        'patient_id': 'COMBAT_001',
        'gps_lat': 34.0522,
        'gps_lon': -118.2437,
        'gps_accuracy': 5,
        'heart_rate': 115,
        'resp_rate': 24,
        'body_temp': 34.5 # slightly colder than average
    }
    
    # Conduct individual MARCH assessment
    patient_1 = conduct_march_triage(
        patient_id="COMBAT_001",
        sensor_data=mock_sensor_data,
        use_audio=CONFIG['enable_audio']
    )
    
    # For MASS CASUALTY: Assess multiple patients, then apply SALT
    # Uncomment below for multi-patient scenario:
    
    """
    # Assess multiple patients
    patients = []
    
    for i in range(1, 4):  # 3 patients
        sensor_data = {
            'patient_id': f'COMBAT_00{i}',
            'gps_lat': 34.0522 + (i * 0.001),
            'gps_lon': -118.2437 + (i * 0.001),
            'heart_rate': np.random.randint(70, 140),
            'resp_rate': np.random.randint(12, 30),
            'body_temp': np.random.uniform(35.5, 37.5)
        }
        
        assessment = conduct_march_triage(
            patient_id=f"COMBAT_00{i}",
            sensor_data=sensor_data,
            use_audio=False  # Use text input for demo
        )
        patients.append(assessment)
    
    # Apply SALT sorting
    sorted_patients = apply_salt_sorting(patients)
    """
    
    print("\n" + "="*80)
    print("ASSESSMENT COMPLETE")
    print("="*80 + "\n")