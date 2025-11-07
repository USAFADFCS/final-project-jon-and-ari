#!/usr/bin/env python3
"""
Combat Triage AI System - Complete Implementation
RATS AI Triage Classifier with SALT Protocol
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
MODEL_ID = "openai/whisper-tiny.en"
DEVICE = "cpu"  # Pi doesn't have GPU
ENABLE_AUDIO = False  # Set to True to enable audio generation

# Combat medical vocabulary for better transcription
COMBAT_MEDICAL_LEXICON = """
tourniquet, hemorrhage, massive hemorrhage, capillary refill, 
obey commands, airway patent, airway obstructed, 
respirations per minute, respiratory rate, breathing adequately,
radial pulse present, radial pulse absent, carotid pulse,
shock, hypotensive, pale, clammy, cold,
GSW, gunshot wound, blast injury, shrapnel, amputation,
conscious, unconscious, alert, verbal, pain, unresponsive,
chest seal, needle decompression, nasopharyngeal airway,
combat gauze, hemostatic agent, pressure dressing,
walking wounded, litter urgent, urgent surgical,
can walk, cannot walk, ambulatory, unable to walk
"""

# ============================================================
# MODEL LOADING
# ============================================================
print("Loading Whisper model...")
asr = pipeline(
    "automatic-speech-recognition",
    model=MODEL_ID,
    device=DEVICE
)
print(f"✓ Model loaded on {DEVICE}")
print(f"✓ Audio generation: {'Enabled' if ENABLE_AUDIO else 'Disabled'}")

# ============================================================
# TRANSCRIPTION FUNCTIONS
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
    
    # Check file exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio file not found: {path}")
    
    # Check file size
    file_size = os.path.getsize(path) / 1024  # KB
    print(f"  📊 File size: {file_size:.2f} KB")
    
    out = {"text": "", "chunks": []}
    
    try:
        # Transcribe directly
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
# ENTITY EXTRACTION
# ============================================================
def extract_triage_entities(transcription_text):
    """Extract SALT-relevant medical information from transcription"""
    text_lower = transcription_text.lower()
    
    entities = {
        "can_walk": None,
        "bleeding_severe": False,
        "obeys_commands": None,
        "resp_rate": None,
        "radial_pulse": None,
        "mental_status": None,
        "cap_refill_sec": None
    }
    
    evidence = []
    
    # Walking ability
    walk_yes = ["can walk", "walking", "ambulatory", "able to walk"]
    walk_no = ["cannot walk", "can't walk", "unable to walk", "not walking"]
    
    for phrase in walk_yes:
        if phrase in text_lower:
            entities["can_walk"] = True
            evidence.append(f"Walking: '{phrase}' detected")
            break
    
    for phrase in walk_no:
        if phrase in text_lower:
            entities["can_walk"] = False
            evidence.append(f"Not walking: '{phrase}' detected")
            break
    
    # Severe bleeding
    bleeding_phrases = ["severe bleeding", "hemorrhage", "massive hemorrhage", 
                       "tourniquet applied", "massive bleeding", "heavy bleeding"]
    for phrase in bleeding_phrases:
        if phrase in text_lower:
            entities["bleeding_severe"] = True
            evidence.append(f"Severe bleeding: '{phrase}' detected")
            break
    
    # Command response
    obey_yes = ["obeys commands", "follows commands", "responsive to commands", "responding"]
    obey_no = ["does not obey", "doesn't obey", "unresponsive", "no response", 
               "not responding", "not obeying"]
    
    for phrase in obey_yes:
        if phrase in text_lower:
            entities["obeys_commands"] = True
            evidence.append(f"Obeys commands: '{phrase}' detected")
            break
    
    for phrase in obey_no:
        if phrase in text_lower:
            entities["obeys_commands"] = False
            evidence.append(f"Does not obey: '{phrase}' detected")
            break
    
    # Respiratory rate extraction
    resp_patterns = [
        r'(\d+)\s*breaths?\s*(?:per\s*minute)?',
        r'(\d+)\s*respirations?\s*(?:per\s*minute)?',
        r'respiratory\s*rate\s*(?:of\s*)?(\d+)',
        r'breathing\s*(?:at\s*)?(\d+)',
        r'(\d+)\s*rpm'
    ]
    
    for pattern in resp_patterns:
        match = re.search(pattern, text_lower)
        if match:
            entities["resp_rate"] = int(match.group(1))
            evidence.append(f"Respiratory rate: {match.group(1)} detected")
            break
    
    # Radial pulse
    pulse_yes = ["radial pulse present", "has radial pulse", "pulse present"]
    pulse_no = ["no radial pulse", "radial pulse absent", "no pulse"]
    
    for phrase in pulse_yes:
        if phrase in text_lower:
            entities["radial_pulse"] = True
            evidence.append(f"Radial pulse: '{phrase}' detected")
            break
    
    for phrase in pulse_no:
        if phrase in text_lower:
            entities["radial_pulse"] = False
            evidence.append(f"No radial pulse: '{phrase}' detected")
            break
    
    # Mental status
    if "alert" in text_lower:
        entities["mental_status"] = "alert"
        evidence.append("Mental status: alert")
    elif "verbal" in text_lower or "responds to verbal" in text_lower:
        entities["mental_status"] = "verbal"
        evidence.append("Mental status: verbal")
    elif "pain" in text_lower or "responds to pain" in text_lower:
        entities["mental_status"] = "pain"
        evidence.append("Mental status: pain")
    elif "unresponsive" in text_lower:
        entities["mental_status"] = "unresponsive"
        evidence.append("Mental status: unresponsive")
    
    return entities, evidence


# ============================================================
# SALT TRIAGE RULES
# ============================================================
def salt_rules(entities, sensors=None):
    """
    Implement SALT (Sort, Assess, Lifesaving interventions, Treatment/Transport) triage
    
    Categories:
    - Immediate (Red): Life-threatening injuries, needs immediate care
    - Delayed (Yellow): Serious injuries, can wait for treatment
    - Minimal (Green): Minor injuries, walking wounded
    - Expectant (Black): Injuries incompatible with life
    """
    s = sensors or {}
    
    # Merge sensor data
    can_walk = entities.get("can_walk") or s.get("can_walk")
    severe_bleed = entities.get("bleeding_severe") or s.get("bleeding_detected")
    resp = entities.get("resp_rate") or s.get("resp_rate")
    obeys = entities.get("obeys_commands") or s.get("obeys_commands")
    radial_pulse = entities.get("radial_pulse") or s.get("radial_pulse")
    
    # SALT Algorithm
    # Step 1: Can the patient walk?
    if can_walk is True:
        return "Minimal"
    
    # Step 2: Assess for life-threatening bleeding
    if severe_bleed:
        return "Immediate"
    
    # Step 3: Check respirations
    if resp is None:
        return "Unknown"  # Need more data
    
    if resp == 0:
        return "Expectant"  # Not breathing
    
    if resp >= 30:
        return "Immediate"  # Respiratory distress
    
    # Step 4: Check mental status / obeys commands
    if obeys is False:
        return "Immediate"  # Altered mental status
    
    # Step 5: Check radial pulse (perfusion)
    if radial_pulse is False:
        return "Immediate"  # Poor perfusion
    
    # Default: injuries present but stable
    return "Delayed"


def calculate_confidence(entities):
    """Calculate confidence based on how much data we have"""
    total_fields = len(entities)
    filled_fields = sum(1 for v in entities.values() if v is not None and v is not False)
    return filled_fields / total_fields


def suggest_next_question(entities):
    """Ask medic for missing critical SALT info"""
    
    if entities["can_walk"] is None:
        return "Can the patient walk?"
    
    if not entities["bleeding_severe"] and entities.get("bleeding_severe") is None:
        return "Is there severe bleeding or hemorrhage?"
    
    if entities["resp_rate"] is None:
        return "What is the respiratory rate per minute?"
    
    if entities["obeys_commands"] is None:
        return "Does the patient obey commands?"
    
    if entities["radial_pulse"] is None:
        return "Is there a radial pulse present?"
    
    return None  # All critical data collected


def fuse_sensor_data(audio_entities, drone_sensors):
    """Combine voice transcription with drone sensor data"""
    final_entities = audio_entities.copy()
    
    # Sensor data overrides uncertain voice data
    if drone_sensors.get("thermal_bleeding_detected") is not None:
        if audio_entities["bleeding_severe"] is None or not audio_entities["bleeding_severe"]:
            final_entities["bleeding_severe"] = drone_sensors["thermal_bleeding_detected"]
    
    if drone_sensors.get("movement_detected") is not None:
        if audio_entities["can_walk"] is None:
            final_entities["can_walk"] = drone_sensors["movement_detected"]
    
    if drone_sensors.get("heart_rate") is not None:
        # Estimate respiratory rate from heart rate if not available
        if audio_entities["resp_rate"] is None:
            # Rough estimate: normal resp is ~1/4 of heart rate
            final_entities["resp_rate"] = int(drone_sensors["heart_rate"] / 4)
    
    return final_entities


# ============================================================
# AUDIO GENERATION (OPTIONAL)
# ============================================================
def generate_triage_audio(result, output_dir="triage_audio"):
    """Generate audio file from triage results"""
    if not ENABLE_AUDIO:
        return None
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Build the spoken message
    message_parts = []
    
    # Triage category announcement
    category = result['triage_category']
    message_parts.append(f"Triage category: {category}.")
    
    # Add urgency based on category
    if category == "Immediate":
        message_parts.append("This patient requires immediate medical attention.")
    elif category == "Delayed":
        message_parts.append("This patient has serious injuries but can wait for treatment.")
    elif category == "Minimal":
        message_parts.append("This patient has minor injuries.")
    elif category == "Expectant":
        message_parts.append("This patient has injuries incompatible with life.")
    elif category == "Unknown":
        message_parts.append("Additional assessment data needed.")
    
    # Confidence level
    confidence_pct = int(result['confidence'] * 100)
    message_parts.append(f"Assessment confidence: {confidence_pct} percent.")
    
    # Next question
    if result['next_question']:
        message_parts.append(f"Recommended next question: {result['next_question']}")
    
    # Combine all parts
    full_message = " ".join(message_parts)
    
    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    patient_id = result['patient_id'].replace('.mp3', '').replace('.wav', '').replace(' ', '_')
    
    print(f"\n🔊 Generating audio output...")
    print(f"📝 Message: {full_message}")
    
    # Try to generate audio with pyttsx3
    try:
        import pyttsx3
        engine = pyttsx3.init()
        
        voices = engine.getProperty('voices')
        if voices:
            engine.setProperty('voice', voices[0].id)
        
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 0.9)
        
        output_filename = f"{output_dir}/triage_{patient_id}_{timestamp}.wav"
        engine.save_to_file(full_message, output_filename)
        engine.runAndWait()
        
        print(f"✓ Audio saved to: {output_filename}")
        return output_filename
        
    except Exception as e:
        print(f"⚠️  Audio generation failed: {e}")
        output_filename = f"{output_dir}/triage_{patient_id}_{timestamp}.txt"
        with open(output_filename, 'w') as f:
            f.write(full_message)
        print(f"✓ Text saved to: {output_filename}")
        return output_filename


# ============================================================
# AUDIO RECORDING
# ============================================================
def record_audio_assessment(duration=10, sample_rate=16000, output_dir="recordings"):
    """Record audio from microphone for triage assessment"""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n🎤 RECORDING TRIAGE ASSESSMENT")
    print(f"{'='*60}")
    print(f"Duration: {duration} seconds")
    print(f"Speak clearly and include SALT assessment details:")
    print(f"  • Can the patient walk?")
    print(f"  • Is there severe bleeding?")
    print(f"  • What is the respiratory rate?")
    print(f"  • Does the patient obey commands?")
    print(f"  • Is there a radial pulse?")
    print(f"{'='*60}\n")
    
    # Countdown
    for i in range(3, 0, -1):
        print(f"Recording starts in {i}...")
        time.sleep(1)
    
    print("🔴 RECORDING NOW - Speak your assessment!")
    
    try:
        # Record audio
        recording = sd.rec(
            int(duration * sample_rate), 
            samplerate=sample_rate, 
            channels=1, 
            dtype='int16'
        )
        sd.wait()
        
        print("✓ Recording complete!\n")
        
        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{output_dir}/assessment_{timestamp}.wav"
        
        write(filename, sample_rate, recording)
        
        print(f"💾 Saved to: {filename}")
        
        # Verify file
        if os.path.exists(filename):
            file_size = os.path.getsize(filename)
            print(f"📊 File size: {file_size / 1024:.2f} KB")
            
            if file_size < 1000:
                print("⚠️  Warning: Audio file is very small. Did you speak?")
        else:
            raise FileNotFoundError(f"Recording file was not created: {filename}")
        
        return filename
        
    except Exception as e:
        print(f"❌ Recording failed: {type(e).__name__}: {e}")
        raise


# ============================================================
# MAIN TRIAGE FUNCTION
# ============================================================
def triage_patient(audio_path, sensor_data=None, generate_audio=True):
    """Complete combat triage pipeline with optional audio output"""
    start_time = time.time()
    
    print(f"\n{'='*60}")
    print(f"TRIAGE ASSESSMENT INITIATED")
    print(f"{'='*60}\n")
    
    # Step 1: Transcribe audio
    print("📝 Transcribing audio...")
    transcription = transcribe_with_vad(audio_path)
    print(f"✓ Transcription: {transcription['text'][:100]}...")
    
    # Step 2: Extract medical entities
    print("\n🔍 Extracting medical information...")
    entities, evidence = extract_triage_entities(transcription["text"])
    
    # Step 3: Fuse with sensor data if available
    if sensor_data:
        print("🤖 Fusing with sensor data...")
        entities = fuse_sensor_data(entities, sensor_data)
    
    # Step 4: Apply SALT triage rules
    print("\n🏥 Applying SALT triage protocol...")
    triage_category = salt_rules(entities, sensor_data)
    
    # Step 5: Calculate confidence and suggest next question
    confidence = calculate_confidence(entities)
    next_question = suggest_next_question(entities)
    
    processing_time = time.time() - start_time
    
    # Format results
    result = {
        "patient_id": os.path.basename(audio_path),
        "transcription": transcription["text"],
        "entities": entities,
        "evidence": evidence,
        "triage_category": triage_category,
        "confidence": confidence,
        "next_question": next_question,
        "processing_time_sec": round(processing_time, 2),
        "timestamp": transcription.get("chunks", [])
    }
    
    # Print results
    print(f"\n{'='*60}")
    print(f"TRIAGE RESULTS")
    print(f"{'='*60}")
    print(f"🚑 Category: {triage_category}")
    print(f"📊 Confidence: {confidence*100:.0f}%")
    print(f"⏱️  Processing Time: {processing_time:.2f}s")
    print(f"\n📋 Extracted Information:")
    for key, value in entities.items():
        if value is not None:
            print(f"  • {key}: {value}")
    
    if next_question:
        print(f"\n❓ Recommended Question: {next_question}")
    
    if evidence:
        print(f"\n📝 Evidence:")
        for item in evidence:
            print(f"  • {item}")
    
    print(f"{'='*60}\n")
    
    # Generate audio output
    if generate_audio and ENABLE_AUDIO:
        audio_file = generate_triage_audio(result)
        result['audio_output'] = audio_file
    else:
        if generate_audio:
            print("⚠️  Audio generation disabled (set ENABLE_AUDIO = True to enable)")
    
    return result


# ============================================================
# INTERACTIVE MODES
# ============================================================
def interactive_triage_session(sensor_data=None):
    """Run an interactive triage session with live recording"""
    print("\n" + "="*60)
    print("🚁 INTERACTIVE COMBAT TRIAGE SYSTEM")
    print("="*60)
    print("\nThis system will:")
    print("1. Record your verbal patient assessment")
    print("2. Transcribe and analyze the information")
    print("3. Apply SALT triage protocol")
    print("4. Provide triage category and next steps")
    print("\n" + "="*60)
    
    # Get recording duration from user
    try:
        duration = int(input("\nHow many seconds do you need? (default: 10): ") or "10")
    except ValueError:
        duration = 10
    
    # Record audio
    audio_file = record_audio_assessment(duration=duration)
    
    # Process with triage system
    print("\n🔄 Processing assessment...")
    result = triage_patient(audio_file, sensor_data=sensor_data, generate_audio=True)
    
    return result


def continuous_triage_mode(sensor_data=None):
    """Continuous triage mode - keeps asking for new assessments"""
    print("\n" + "="*60)
    print("🔁 CONTINUOUS TRIAGE MODE")
    print("="*60)
    print("Press Ctrl+C to exit\n")
    
    patient_count = 0
    
    try:
        while True:
            patient_count += 1
            print(f"\n{'='*60}")
            print(f"PATIENT #{patient_count}")
            print(f"{'='*60}")
            
            # Run triage session
            result = interactive_triage_session(sensor_data=sensor_data)
            
            # Ask if user wants to continue
            continue_input = input("\n\nAssess another patient? (y/n): ").lower()
            if continue_input != 'y':
                break
                
    except KeyboardInterrupt:
        print("\n\n✓ Triage mode ended")
        print(f"Total patients assessed: {patient_count}")


class TriageQueue:
    """Manage multiple patients in a triage queue"""
    
    def __init__(self):
        self.patients = []
        self.patient_counter = 0
    
    def add_patient(self, result):
        """Add a patient to the queue"""
        self.patient_counter += 1
        patient_data = {
            'id': self.patient_counter,
            'timestamp': datetime.now().strftime("%H:%M:%S"),
            'category': result['triage_category'],
            'confidence': result['confidence'],
            'transcription': result['transcription'],
            'entities': result['entities'],
            'next_question': result['next_question']
        }
        self.patients.append(patient_data)
        return patient_data
    
    def get_priority_order(self):
        """Sort patients by triage priority"""
        priority_map = {
            'Immediate': 1,
            'Delayed': 2,
            'Minimal': 3,
            'Expectant': 4,
            'Unknown': 5
        }
        return sorted(self.patients, key=lambda x: priority_map.get(x['category'], 99))
    
    def display_queue(self):
        """Display current triage queue"""
        print("\n" + "="*80)
        print("🏥 CURRENT TRIAGE QUEUE")
        print("="*80)
        
        if not self.patients:
            print("No patients in queue")
            print("="*80)
            return
        
        priority_patients = self.get_priority_order()
        
        # Group by category
        categories = {}
        for patient in priority_patients:
            cat = patient['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(patient)
        
        # Display by priority
        category_icons = {
            'Immediate': '🔴',
            'Delayed': '🟡',
            'Minimal': '🟢',
            'Expectant': '⚫',
            'Unknown': '⚪'
        }
        
        for category in ['Immediate', 'Delayed', 'Minimal', 'Expectant', 'Unknown']:
            if category in categories:
                print(f"\n{category_icons[category]} {category.upper()} ({len(categories[category])} patients)")
                print("-" * 80)
                for patient in categories[category]:
                    print(f"  Patient #{patient['id']} | Time: {patient['timestamp']} | "
                          f"Confidence: {patient['confidence']*100:.0f}%")
                    if patient['next_question']:
                        print(f"    ❓ Next: {patient['next_question']}")
        
        print("="*80)


def multi_patient_triage_interface():
    """Interactive interface for triaging multiple patients"""
    queue = TriageQueue()
    
    print("\n" + "="*80)
    print("🚁 MULTI-PATIENT COMBAT TRIAGE SYSTEM")
    print("="*80)
    print("\nThis system manages multiple patients in a mass casualty scenario.")
    print("You can assess patients using:")
    print("  1. Real-time audio recording")
    print("  2. Pre-recorded audio files")
    print("\nThe system will maintain a priority queue and recommend treatment order.")
    print("="*80)
    
    try:
        while True:
            print("\n" + "="*80)
            print(f"PATIENT ASSESSMENT #{queue.patient_counter + 1}")
            print("="*80)
            
            # Choose input method
            print("\nHow would you like to assess this patient?")
            print("1. Record audio now (real-time)")
            print("2. Use existing audio file")
            print("3. View current triage queue")
            print("4. Exit and show final queue")
            
            choice = input("\nEnter choice (1/2/3/4): ").strip()
            
            if choice == "1":
                # Real-time recording
                print("\n📍 Preparing for real-time assessment...")
                
                # Get recording duration
                try:
                    duration = int(input("\nRecording duration in seconds (default: 10): ") or "10")
                except ValueError:
                    duration = 10
                
                # Record and process
                audio_file = record_audio_assessment(duration=duration)
                result = triage_patient(audio_file, sensor_data=None, generate_audio=True)
                
                # Add to queue
                patient_data = queue.add_patient(result)
                print(f"\n✓ Patient #{patient_data['id']} added to queue as {patient_data['category']}")
                
            elif choice == "2":
                # Use existing file
                audio_path = input("\nEnter audio file path: ").strip()
                
                if not os.path.exists(audio_path):
                    print(f"❌ File not found: {audio_path}")
                    continue
                
                # Process file
                result = triage_patient(audio_path, sensor_data=None, generate_audio=True)
                
                # Add to queue
                patient_data = queue.add_patient(result)
                print(f"\n✓ Patient #{patient_data['id']} added to queue as {patient_data['category']}")
                
            elif choice == "3":
                # Display queue
                queue.display_queue()
                input("\nPress Enter to continue...")
                continue
                
            elif choice == "4":
                # Exit
                break
                
            else:
                print("❌ Invalid choice")
                continue
            
            # Show updated queue after each patient
            queue.display_queue()
            
            # Ask to continue
            continue_input = input("\nAssess another patient? (y/n): ").lower()
            if continue_input != 'y':
                break
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Assessment interrupted")
    
    # Final summary
    print("\n" + "="*80)
    print("📊 FINAL TRIAGE SUMMARY")
    print("="*80)
    print(f"Total patients assessed: {queue.patient_counter}")
    
    queue.display_queue()
    
    # Treatment recommendations
    print("\n" + "="*80)
    print("💉 RECOMMENDED TREATMENT ORDER")
    print("="*80)
    
    priority_patients = queue.get_priority_order()
    for i, patient in enumerate(priority_patients, 1):
        print(f"{i}. Patient #{patient['id']} - {patient['category']} "
              f"(Confidence: {patient['confidence']*100:.0f}%)")
        if patient['next_question']:
            print(f"   ⚠️  Incomplete assessment: {patient['next_question']}")
    
    print("="*80)
    
    return queue


# ============================================================
# MAIN PROGRAM
# ============================================================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("TRIAGE SYSTEM READY")
    print("="*80)
    print("\nChoose a mode:")
    print("1. Single interactive assessment (with recording)")
    print("2. Continuous triage mode (multiple patients)")
    print("3. Multi-patient triage interface (RECOMMENDED)")
    print("4. Test with existing audio file")
    print("="*80)
    
    mode = input("\nEnter mode (1/2/3/4): ").strip()
    
    if mode == "1":
        # Single interactive session
        result = interactive_triage_session()
        
    elif mode == "2":
        # Continuous mode
        continuous_triage_mode()
        
    elif mode == "3":
        # Multi-patient interface
        triage_queue = multi_patient_triage_interface()
        
    elif mode == "4":
        # Test with existing file
        AUDIO = input("Enter audio file path: ").strip() or "EnglishTriageTest.wav"
        
        if not os.path.exists(AUDIO):
            print(f"\n❌ ERROR: File not found: {AUDIO}")
            print(f"Current directory: {os.getcwd()}")
            print(f"Files in directory: {[f for f in os.listdir('.') if f.endswith(('.wav', '.mp3'))]}")
        else:
            print("\nTEST 1: Audio transcription")
            result1 = triage_patient(AUDIO, generate_audio=True)
            
            print("\n\nTEST 2: Audio + sensor fusion")
            mock_sensor_data = {
                "thermal_bleeding_detected": False,
                "movement_detected": False,
                "heart_rate": 120
            }
            result2 = triage_patient(AUDIO, sensor_data=mock_sensor_data, generate_audio=True)
    else:
        print("Invalid mode selected")

print("\n✓ Program complete")