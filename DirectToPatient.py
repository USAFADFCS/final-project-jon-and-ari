#!/usr/bin/env python3
"""
Combat Triage AI System - Direct Patient Interaction
AI Agent that directly assesses patients through dialogue
Ready for Raspberry Pi Deployment
"""

import os
import torch
import torchaudio
import re
import time
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from transformers import pipeline
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_ID = "openai/whisper-medium"
DEVICE = "cpu"
ENABLE_AUDIO = False

print("Loading Whisper model...")
asr = pipeline("automatic-speech-recognition", model=MODEL_ID, device=DEVICE)
print(f"✓ Model loaded on {DEVICE}")

print("Loading DistilBERT zero-shot classification model...")
classifier = pipeline(
    "zero-shot-classification",
    model="typeform/distilbert-base-uncased-mnli",
    device=-1
)
print(f"✓ DistilBERT model loaded")

# ============================================================
# PATIENT INTERACTION STATE MACHINE
# ============================================================
class PatientAssessmentSession:
    """Manages a sequential assessment dialogue with a patient"""
    
    def __init__(self, patient_id=None):
        self.patient_id = patient_id or f"Patient_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.entities = {
            "can_walk": None,
            "bleeding_severe": False,
            "obeys_commands": None,
            "resp_rate": None,
            "radial_pulse": None,
            "mental_status": None,
            "cap_refill_sec": None,
            "responds_to_voice": None,
            "visible_injury": None
        }
        self.evidence = []
        self.interaction_history = []
        self.current_step = 0
        self.assessment_complete = False
        
        # SALT-based question sequence
        self.question_sequence = [
            {
                "id": "initial_contact",
                "question": "Can you hear me? If you can hear me, say yes or make a sound.",
                "target": "responds_to_voice",
                "critical": True,
                "timeout": 5
            },
            {
                "id": "walking_test",
                "question": "Can you walk? If you can walk, please stand up and take a step toward me.",
                "target": "can_walk",
                "critical": True,
                "timeout": 10,
                "visual_cue": True  # AI should observe movement
            },
            {
                "id": "command_response",
                "question": "Can you squeeze my hand? Squeeze if you understand me.",
                "target": "obeys_commands",
                "critical": True,
                "timeout": 5
            },
            {
                "id": "injury_check",
                "question": "Where are you hurt? Tell me if you have any bleeding or pain.",
                "target": "bleeding_severe",
                "critical": True,
                "timeout": 10
            },
            {
                "id": "breathing_assessment",
                "question": "I'm going to count your breaths. Just breathe normally for 15 seconds.",
                "target": "resp_rate",
                "critical": True,
                "timeout": 20,
                "observational": True
            }
        ]
    
    def get_next_question(self):
        """Get the next question in the assessment sequence"""
        if self.current_step >= len(self.question_sequence):
            self.assessment_complete = True
            return None
        
        return self.question_sequence[self.current_step]
    
    def record_interaction(self, question, patient_response, extracted_data):
        """Record each interaction for analysis"""
        self.interaction_history.append({
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "response": patient_response,
            "extracted": extracted_data
        })
    
    def advance_step(self):
        """Move to next assessment step"""
        self.current_step += 1


# ============================================================
# TRANSCRIPTION
# ============================================================
def transcribe(path: str) -> dict:
    """Transcribe audio file"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    print(f"  Transcribing {os.path.basename(path)}...")
    result = asr(path)
    print(f"  ✓ Done: {len(result['text'])} characters")
    return result


def transcribe_with_vad(path):
    """Transcribe with voice activity detection"""
    print(f"📂 Loading: {path}")
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio file not found: {path}")
    
    file_size = os.path.getsize(path) / 1024
    print(f"  📊 File size: {file_size:.2f} KB")
    
    out = {"text": "", "chunks": []}
    
    try:
        r = transcribe(path)
        out["text"] = r["text"].strip()
        if "chunks" in r:
            out["chunks"] = r["chunks"]
        print(f"  ✓ Transcription complete")
    except Exception as e:
        print(f"  ❌ Transcription failed: {type(e).__name__}: {e}")
        out["text"] = f"[Error: {str(e)}]"
    
    return out


# ============================================================
# INTELLIGENT RESPONSE ANALYSIS
# ============================================================
def analyze_patient_response(question_context, patient_response, target_entity):
    """
    Use DistilBERT to intelligently interpret patient responses
    Context-aware analysis based on what question was asked
    """
    text_lower = patient_response.lower()
    result = {
        "value": None,
        "confidence": 0.0,
        "method": "unknown",
        "evidence": ""
    }
    
    # ===== VOICE RESPONSE (Initial Contact) =====
    if target_entity == "responds_to_voice":
        # Check for ANY verbal response
        if len(patient_response.strip()) > 0:
            # Use AI to detect affirmative vs negative vs unclear
            try:
                voice_result = classifier(
                    patient_response,
                    candidate_labels=["affirmative response", "negative response", "unclear or confused"],
                    hypothesis_template="This is an {}"
                )
                
                if voice_result['scores'][0] > 0.4:
                    top_label = voice_result['labels'][0]
                    result["value"] = True  # They responded verbally
                    result["confidence"] = voice_result['scores'][0]
                    result["method"] = "DistilBERT"
                    result["evidence"] = f"Patient responded verbally: '{patient_response[:50]}...'"
                else:
                    result["value"] = True  # Any speech indicates responsiveness
                    result["confidence"] = 0.6
                    result["method"] = "length_check"
                    result["evidence"] = "Patient produced verbal response"
            except:
                result["value"] = True
                result["confidence"] = 0.5
                result["method"] = "fallback"
                result["evidence"] = "Verbal response detected"
        else:
            result["value"] = False
            result["confidence"] = 0.9
            result["method"] = "silence"
            result["evidence"] = "No verbal response detected"
    
    # ===== WALKING ABILITY =====
    elif target_entity == "can_walk":
        try:
            walk_result = classifier(
                patient_response,
                candidate_labels=[
                    "patient can walk or is walking",
                    "patient cannot walk or refuses to walk",
                    "patient is trying but struggling",
                    "unclear response"
                ],
                hypothesis_template="The patient's response indicates that the {}"
            )
            
            if walk_result['scores'][0] > 0.5:
                top_label = walk_result['labels'][0]
                
                if "can walk" in top_label or "is walking" in top_label:
                    result["value"] = True
                    result["confidence"] = walk_result['scores'][0]
                    result["method"] = "DistilBERT"
                    result["evidence"] = f"AI detected walking ability ({walk_result['scores'][0]:.2f})"
                
                elif "cannot walk" in top_label or "refuses" in top_label:
                    result["value"] = False
                    result["confidence"] = walk_result['scores'][0]
                    result["method"] = "DistilBERT"
                    result["evidence"] = f"AI detected inability to walk ({walk_result['scores'][0]:.2f})"
                
                elif "struggling" in top_label:
                    result["value"] = False  # Struggling = cannot walk effectively
                    result["confidence"] = walk_result['scores'][0] * 0.8
                    result["method"] = "DistilBERT"
                    result["evidence"] = f"AI detected difficulty walking ({walk_result['scores'][0]:.2f})"
        
        except Exception as e:
            # Fallback regex
            walk_yes = ["yes", "i can", "i'm walking", "standing", "walked"]
            walk_no = ["no", "can't", "cannot", "hurt", "broken", "stuck"]
            
            for phrase in walk_yes:
                if phrase in text_lower:
                    result["value"] = True
                    result["confidence"] = 0.7
                    result["method"] = "regex"
                    result["evidence"] = f"Phrase '{phrase}' detected"
                    break
            
            for phrase in walk_no:
                if phrase in text_lower:
                    result["value"] = False
                    result["confidence"] = 0.7
                    result["method"] = "regex"
                    result["evidence"] = f"Phrase '{phrase}' detected"
                    break
    
    # ===== COMMAND RESPONSE (Obeys Commands) =====
    elif target_entity == "obeys_commands":
        try:
            cmd_result = classifier(
                patient_response,
                candidate_labels=[
                    "patient is following the command",
                    "patient is not following or refusing",
                    "patient is confused or unclear"
                ],
                hypothesis_template="The patient's behavior shows that the {}"
            )
            
            if cmd_result['scores'][0] > 0.5:
                top_label = cmd_result['labels'][0]
                
                if "following" in top_label:
                    result["value"] = True
                    result["confidence"] = cmd_result['scores'][0]
                    result["method"] = "DistilBERT"
                    result["evidence"] = f"AI detected command compliance ({cmd_result['scores'][0]:.2f})"
                
                elif "not following" in top_label or "refusing" in top_label:
                    result["value"] = False
                    result["confidence"] = cmd_result['scores'][0]
                    result["method"] = "DistilBERT"
                    result["evidence"] = f"AI detected non-compliance ({cmd_result['scores'][0]:.2f})"
        
        except:
            # Fallback
            obey_yes = ["yes", "okay", "squeezing", "got it", "i understand"]
            obey_no = ["no", "can't", "what", "confused", "don't understand"]
            
            for phrase in obey_yes:
                if phrase in text_lower:
                    result["value"] = True
                    result["confidence"] = 0.7
                    result["method"] = "regex"
                    result["evidence"] = f"Phrase '{phrase}' detected"
                    break
            
            for phrase in obey_no:
                if phrase in text_lower:
                    result["value"] = False
                    result["confidence"] = 0.7
                    result["method"] = "regex"
                    result["evidence"] = f"Phrase '{phrase}' detected"
                    break
    
    # ===== BLEEDING/INJURY ASSESSMENT =====
    elif target_entity == "bleeding_severe":
        try:
            bleed_result = classifier(
                patient_response,
                candidate_labels=[
                    "severe bleeding or hemorrhage",
                    "minor bleeding or wounds",
                    "no bleeding mentioned",
                    "unclear injury description"
                ],
                hypothesis_template="The patient describes {}"
            )
            
            if bleed_result['scores'][0] > 0.5:
                top_label = bleed_result['labels'][0]
                
                if "severe" in top_label:
                    result["value"] = True
                    result["confidence"] = bleed_result['scores'][0]
                    result["method"] = "DistilBERT"
                    result["evidence"] = f"AI detected severe bleeding ({bleed_result['scores'][0]:.2f})"
                
                elif "minor" in top_label:
                    result["value"] = False
                    result["confidence"] = bleed_result['scores'][0]
                    result["method"] = "DistilBERT"
                    result["evidence"] = f"AI detected minor injury only ({bleed_result['scores'][0]:.2f})"
        
        except:
            # Fallback
            severe_keywords = ["bleeding", "blood", "hemorrhage", "gushing", "lot of blood", "tourniquet"]
            
            for keyword in severe_keywords:
                if keyword in text_lower:
                    result["value"] = True
                    result["confidence"] = 0.7
                    result["method"] = "regex"
                    result["evidence"] = f"Keyword '{keyword}' detected"
                    break
    
    # ===== RESPIRATORY RATE (Observational) =====
    elif target_entity == "resp_rate":
        # Extract number from observation
        resp_patterns = [
            r'(\d+)\s*breaths',
            r'(\d+)\s*per\s*minute',
            r'counted\s*(\d+)',
            r'rate\s*(?:of\s*)?(\d+)',
        ]
        
        for pattern in resp_patterns:
            match = re.search(pattern, text_lower)
            if match:
                result["value"] = int(match.group(1))
                result["confidence"] = 0.9
                result["method"] = "regex"
                result["evidence"] = f"Extracted respiratory rate: {match.group(1)}"
                break
    
    return result


# ============================================================
# INTERACTIVE PATIENT ASSESSMENT
# ============================================================
def conduct_patient_assessment(audio_response_path=None, session=None):
    """
    Conduct one step of patient assessment
    Can use pre-recorded audio or live recording
    """
    if session is None:
        session = PatientAssessmentSession()
    
    # Get next question
    question_data = session.get_next_question()
    
    if question_data is None:
        print("\n✓ Assessment complete!")
        return session, None
    
    # Present question to patient
    print(f"\n{'='*60}")
    print(f"ASSESSMENT STEP {session.current_step + 1}/{len(session.question_sequence)}")
    print(f"{'='*60}")
    print(f"\n🤖 AI: {question_data['question']}")
    
    # Record or load patient response
    if audio_response_path:
        print(f"📂 Loading patient response: {audio_response_path}")
        response_audio = audio_response_path
    else:
        print(f"\n⏳ Waiting {question_data['timeout']} seconds for patient response...")
        response_audio = record_audio_assessment(
            duration=question_data['timeout'],
            output_dir="patient_responses"
        )
    
    # Transcribe patient response
    print("\n🎧 Processing patient response...")
    transcription = transcribe_with_vad(response_audio)
    patient_response = transcription["text"]
    
    print(f"📝 Patient said: \"{patient_response}\"")
    
    # Analyze response with AI
    print(f"🧠 Analyzing response for: {question_data['target']}")
    analysis = analyze_patient_response(
        question_context=question_data['question'],
        patient_response=patient_response,
        target_entity=question_data['target']
    )
    
    # Update session state
    if analysis['value'] is not None:
        session.entities[question_data['target']] = analysis['value']
        session.evidence.append(analysis['evidence'])
        
        print(f"✓ Extracted: {question_data['target']} = {analysis['value']}")
        print(f"  Confidence: {analysis['confidence']:.2f} | Method: {analysis['method']}")
        print(f"  Evidence: {analysis['evidence']}")
    else:
        print(f"⚠️  Could not extract {question_data['target']} from response")
    
    # Record interaction
    session.record_interaction(
        question=question_data['question'],
        patient_response=patient_response,
        extracted_data=analysis
    )
    
    # Advance to next step
    session.advance_step()
    
    return session, question_data


# ============================================================
# SALT TRIAGE RULES (Adapted for Direct Patient Data)
# ============================================================
def salt_rules_direct_patient(entities, sensors=None):
    """
    SALT triage adapted for direct patient interaction
    """
    s = sensors or {}
    
    # Check if patient is responsive at all
    responds = entities.get("responds_to_voice")
    if responds is False:
        # Unresponsive patient - check vitals
        resp = entities.get("resp_rate") or s.get("resp_rate")
        if resp == 0 or resp is None:
            return "Expectant"  # Not breathing, unresponsive
        else:
            return "Immediate"  # Breathing but unresponsive
    
    # Patient is responsive - continue SALT algorithm
    can_walk = entities.get("can_walk") or s.get("can_walk")
    severe_bleed = entities.get("bleeding_severe") or s.get("bleeding_detected")
    resp = entities.get("resp_rate") or s.get("resp_rate")
    obeys = entities.get("obeys_commands") or s.get("obeys_commands")
    radial_pulse = entities.get("radial_pulse") or s.get("radial_pulse")
    
    # Step 1: Can walk?
    if can_walk is True:
        return "Minimal"
    
    # Step 2: Severe bleeding?
    if severe_bleed:
        return "Immediate"
    
    # Step 3: Respirations
    if resp is not None:
        if resp == 0:
            return "Expectant"
        if resp >= 30:
            return "Immediate"
    
    # Step 4: Mental status / commands
    if obeys is False:
        return "Immediate"
    
    # Step 5: Perfusion
    if radial_pulse is False:
        return "Immediate"
    
    # Default: stable but injured
    return "Delayed"


def calculate_confidence(entities):
    """Calculate confidence based on data completeness"""
    total_fields = len(entities)
    filled_fields = sum(1 for v in entities.values() if v is not None and v is not False)
    return filled_fields / total_fields


# ============================================================
# AUDIO RECORDING
# ============================================================
def record_audio_assessment(duration=10, sample_rate=16000, output_dir="recordings"):
    """Record audio from microphone"""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🔴 Recording for {duration} seconds...")
    
    try:
        recording = sd.rec(
            int(duration * sample_rate), 
            samplerate=sample_rate, 
            channels=1, 
            dtype='int16'
        )
        sd.wait()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{output_dir}/response_{timestamp}.wav"
        
        write(filename, sample_rate, recording)
        print(f"✓ Saved to: {filename}")
        
        return filename
        
    except Exception as e:
        print(f"❌ Recording failed: {e}")
        raise


# ============================================================
# COMPLETE PATIENT TRIAGE WORKFLOW
# ============================================================
def autonomous_patient_triage(patient_id=None, sensor_data=None):
    """
    Fully autonomous patient assessment through direct interaction
    """
    start_time = time.time()
    
    print(f"\n{'='*80}")
    print(f"🚁 AUTONOMOUS PATIENT TRIAGE INITIATED")
    print(f"{'='*80}\n")
    
    # Initialize session
    session = PatientAssessmentSession(patient_id=patient_id)
    
    print(f"Patient ID: {session.patient_id}")
    print(f"Total assessment steps: {len(session.question_sequence)}")
    
    # Conduct sequential assessment
    while not session.assessment_complete:
        session, question_data = conduct_patient_assessment(session=session)
        
        if question_data and question_data.get('critical'):
            # Check if we can make early triage decision
            preliminary_triage = salt_rules_direct_patient(session.entities, sensor_data)
            
            if preliminary_triage in ["Expectant", "Immediate"]:
                print(f"\n⚠️  EARLY TRIAGE DECISION: {preliminary_triage}")
                print(f"Critical condition detected. Completing assessment...")
    
    # Final triage decision
    print(f"\n{'='*80}")
    print(f"FINAL TRIAGE ASSESSMENT")
    print(f"{'='*80}")
    
    triage_category = salt_rules_direct_patient(session.entities, sensor_data)
    confidence = calculate_confidence(session.entities)
    processing_time = time.time() - start_time
    
    result = {
        "patient_id": session.patient_id,
        "triage_category": triage_category,
        "confidence": confidence,
        "entities": session.entities,
        "evidence": session.evidence,
        "interaction_history": session.interaction_history,
        "processing_time_sec": round(processing_time, 2)
    }
    
    # Display results
    print(f"\n🚑 TRIAGE CATEGORY: {triage_category}")
    print(f"📊 Confidence: {confidence*100:.0f}%")
    print(f"⏱️  Total Time: {processing_time:.2f}s")
    
    print(f"\n📋 Patient Status:")
    for key, value in session.entities.items():
        if value is not None:
            print(f"  • {key}: {value}")
    
    print(f"\n🔍 Evidence Trail:")
    for evidence in session.evidence:
        print(f"  • {evidence}")
    
    print(f"\n{'='*80}\n")
    
    return result


# ============================================================
# MAIN PROGRAM
# ============================================================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("AUTONOMOUS PATIENT TRIAGE SYSTEM")
    print("Direct AI-to-Patient Interaction")
    print("="*80)
    print("\nThis system will:")
    print("1. Ask patients direct questions")
    print("2. Listen to and analyze patient responses")
    print("3. Make autonomous triage decisions")
    print("4. Follow SALT protocol throughout")
    print("="*80)
    
    mode = input("\nStart autonomous assessment? (yes/no): ").strip().lower()
    
    if mode == "yes" or mode == "y":
        result = autonomous_patient_triage()
        
        print("\n✓ Assessment complete!")
        print(f"Patient {result['patient_id']} triaged as: {result['triage_category']}")
    else:
        print("\nSystem ready. Call autonomous_patient_triage() to begin.")

print("\n✓ System loaded and ready")