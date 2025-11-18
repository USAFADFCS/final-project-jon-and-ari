#!/usr/bin/env python3
"""
Agentic Combat Triage System
Adds LLM reasoning layer to your existing system
"""

import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

# ============================================================
# AGENT CONFIGURATION
# ============================================================

class AgentMode(Enum):
    LOCAL = "local"  # Use local LLM (Llama/Phi)
    CLOUD = "cloud"  # Use Claude API
    HYBRID = "hybrid"  # Try local, fallback to cloud

@dataclass
class TriageState:
    """Current state of triage assessment"""
    patient_id: str
    transcription: str
    entities: Dict
    evidence: List[str]
    confidence: float
    questions_asked: List[str]
    answers_received: List[str]
    sensor_data: Optional[Dict] = None
    reasoning_chain: List[str] = None
    
    def __post_init__(self):
        if self.reasoning_chain is None:
            self.reasoning_chain = []


# ============================================================
# AGENTIC TRIAGE AGENT
# ============================================================

class TriageAgent:
    """
    Agentic layer for intelligent triage decision-making
    Uses LLM to reason about medical information and guide assessment
    """
    
    def __init__(self, mode: AgentMode = AgentMode.LOCAL, api_key: Optional[str] = None):
        self.mode = mode
        self.api_key = api_key
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the appropriate LLM based on mode"""
        if self.mode == AgentMode.LOCAL or self.mode == AgentMode.HYBRID:
            self._initialize_local_model()
        
        if self.mode == AgentMode.CLOUD:
            self._initialize_cloud_model()
    
    def _initialize_local_model(self):
        """Initialize local LLM (Llama or Phi)"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            # Try Phi-3-mini first (smaller, faster)
            model_name = "microsoft/Phi-3-mini-4k-instruct"
            print(f"Loading local model: {model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,  # Use float32 for CPU
                device_map="cpu",
                trust_remote_code=True
            )
            print("✓ Local model loaded successfully")
            
        except Exception as e:
            print(f"⚠️  Failed to load local model: {e}")
            if self.mode == AgentMode.LOCAL:
                print("❌ LOCAL mode requires a local model. Exiting.")
                raise
            else:
                print("→ Falling back to cloud mode")
                self.mode = AgentMode.CLOUD
                self._initialize_cloud_model()
    
    def _initialize_cloud_model(self):
        """Initialize cloud API connection"""
        if not self.api_key:
            print("⚠️  No API key provided for cloud mode")
            print("Set ANTHROPIC_API_KEY environment variable or pass api_key parameter")
    
    def _call_local_llm(self, prompt: str, max_tokens: int = 500) -> str:
        """Call local LLM"""
        messages = [{"role": "user", "content": prompt}]
        
        inputs = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True
        )
        
        outputs = self.model.generate(
            inputs,
            max_new_tokens=max_tokens,
            temperature=0.3,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the assistant's response
        response = response.split("assistant")[-1].strip()
        return response
    
    def _call_cloud_llm(self, prompt: str, max_tokens: int = 500) -> str:
        """Call Claude API"""
        try:
            import anthropic
            
            client = anthropic.Anthropic(api_key=self.api_key)
            
            message = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return message.content[0].text
            
        except Exception as e:
            print(f"❌ Cloud API call failed: {e}")
            return ""
    
    def _call_llm(self, prompt: str, max_tokens: int = 500) -> str:
        """Route to appropriate LLM based on mode"""
        if self.mode == AgentMode.CLOUD:
            return self._call_cloud_llm(prompt, max_tokens)
        elif self.mode == AgentMode.LOCAL:
            return self._call_local_llm(prompt, max_tokens)
        else:  # HYBRID
            try:
                return self._call_local_llm(prompt, max_tokens)
            except:
                print("→ Local model failed, trying cloud...")
                return self._call_cloud_llm(prompt, max_tokens)
    
    def interpret_transcription(self, state: TriageState) -> Dict:
        """
        Use LLM to interpret ambiguous or incomplete transcriptions
        Extracts medical entities using natural language understanding
        """
        prompt = f"""You are a combat medic AI assistant. Analyze this patient assessment transcription and extract ALL relevant SALT triage information.

TRANSCRIPTION: "{state.transcription}"

Extract the following information if mentioned (respond with JSON only):
{{
  "can_walk": true/false/null,
  "bleeding_severe": true/false,
  "obeys_commands": true/false/null,
  "resp_rate": number or null,
  "radial_pulse": true/false/null,
  "mental_status": "alert"/"verbal"/"pain"/"unresponsive"/null,
  "injuries_described": "brief summary",
  "ambiguities": ["list of unclear points"],
  "confidence": 0.0-1.0
}}

Return ONLY valid JSON, no other text."""

        response = self._call_llm(prompt, max_tokens=300)
        
        try:
            # Try to extract JSON from response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]
            else:
                json_str = response
            
            entities = json.loads(json_str.strip())
            state.reasoning_chain.append(f"LLM interpretation: {entities}")
            return entities
            
        except json.JSONDecodeError as e:
            print(f"⚠️  Failed to parse LLM response as JSON: {e}")
            print(f"Response was: {response}")
            return {}
    
    def resolve_contradictions(self, state: TriageState, 
                              audio_entities: Dict, 
                              sensor_data: Dict) -> Dict:
        """
        Use LLM to intelligently resolve contradictions between audio and sensors
        """
        prompt = f"""You are a combat medic AI. Resolve contradictions between voice assessment and sensor data.

VOICE ASSESSMENT:
{json.dumps(audio_entities, indent=2)}

SENSOR DATA:
{json.dumps(sensor_data, indent=2)}

TRANSCRIPTION: "{state.transcription}"

Consider:
1. Which source is more reliable for each measurement?
2. Are there legitimate medical reasons for discrepancies?
3. What is the most accurate assessment?

Respond with JSON containing:
{{
  "resolved_entities": {{...}},
  "reasoning": "explanation of resolution",
  "confidence": 0.0-1.0,
  "warnings": ["any concerns"]
}}

Return ONLY valid JSON."""

        response = self._call_llm(prompt, max_tokens=400)
        
        try:
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            else:
                json_str = response.strip()
            
            resolution = json.loads(json_str)
            state.reasoning_chain.append(f"Contradiction resolution: {resolution['reasoning']}")
            return resolution
            
        except:
            print("⚠️  Failed to resolve contradictions via LLM")
            # Fallback to sensor priority
            return {"resolved_entities": {**audio_entities, **sensor_data}}
    
    def generate_next_question(self, state: TriageState, 
                               current_entities: Dict,
                               triage_category: str) -> Optional[str]:
        """
        Use LLM to generate the most valuable next question
        More intelligent than rule-based approach
        """
        prompt = f"""You are a combat medic AI conducting a triage assessment.

PATIENT STATUS:
- Current triage category: {triage_category}
- Confidence: {state.confidence:.0%}
- Already collected: {json.dumps({k: v for k, v in current_entities.items() if v is not None}, indent=2)}
- Missing: {json.dumps({k: v for k, v in current_entities.items() if v is None}, indent=2)}

PREVIOUS QUESTIONS: {state.questions_asked}

TRANSCRIPTION: "{state.transcription}"

What is the SINGLE MOST IMPORTANT question to ask next to:
1. Increase assessment confidence
2. Confirm or change triage category
3. Identify life-threatening conditions

Respond with JSON:
{{
  "question": "the specific question to ask",
  "reasoning": "why this question is most important",
  "expected_impact": "how answer will affect triage"
}}

If assessment is complete, set "question" to null.
Return ONLY valid JSON."""

        response = self._call_llm(prompt, max_tokens=300)
        
        try:
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            else:
                json_str = response.strip()
            
            result = json.loads(json_str)
            
            if result.get("question"):
                state.reasoning_chain.append(f"Next question: {result['question']} - {result['reasoning']}")
                return result["question"]
            else:
                return None
                
        except:
            print("⚠️  Failed to generate next question via LLM")
            return None
    
    def make_triage_decision(self, state: TriageState, 
                            entities: Dict,
                            rule_based_category: str) -> Tuple[str, str, float]:
        """
        Final agentic decision with reasoning
        Can override rule-based system when appropriate
        """
        prompt = f"""You are an expert combat medic making a final triage decision.

PATIENT DATA:
{json.dumps(entities, indent=2)}

RULE-BASED ASSESSMENT: {rule_based_category}

FULL TRANSCRIPTION: "{state.transcription}"

SENSOR DATA: {json.dumps(state.sensor_data or {}, indent=2)}

REASONING CHAIN:
{chr(10).join(state.reasoning_chain)}

Using SALT triage protocol, determine:
1. Final triage category: Immediate/Delayed/Minimal/Expectant
2. Whether to agree with or override rule-based assessment
3. Detailed medical reasoning

Respond with JSON:
{{
  "final_category": "Immediate/Delayed/Minimal/Expectant",
  "agrees_with_rules": true/false,
  "reasoning": "detailed medical justification",
  "confidence": 0.0-1.0,
  "critical_findings": ["key factors in decision"],
  "recommended_interventions": ["immediate actions needed"]
}}

Return ONLY valid JSON."""

        response = self._call_llm(prompt, max_tokens=600)
        
        try:
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            else:
                json_str = response.strip()
            
            decision = json.loads(json_str)
            
            return (
                decision["final_category"],
                decision["reasoning"],
                decision["confidence"]
            )
            
        except Exception as e:
            print(f"⚠️  Failed to make LLM decision: {e}")
            # Fallback to rule-based
            return rule_based_category, "Fallback to rule-based assessment", 0.7
    
    def explain_decision(self, state: TriageState, 
                        final_category: str,
                        entities: Dict) -> str:
        """
        Generate human-readable explanation of triage decision
        """
        prompt = f"""Generate a clear, concise explanation of this triage decision for medical personnel.

PATIENT: {state.patient_id}
TRIAGE CATEGORY: {final_category}

KEY FINDINGS:
{json.dumps(entities, indent=2)}

TRANSCRIPTION: "{state.transcription}"

REASONING:
{chr(10).join(state.reasoning_chain)}

Write a 2-3 sentence explanation suitable for handoff to receiving medical facility.
Focus on: critical findings, category justification, and urgent interventions needed."""

        explanation = self._call_llm(prompt, max_tokens=200)
        return explanation.strip()


# ============================================================
# INTEGRATION WITH EXISTING SYSTEM
# ============================================================

def agentic_triage_patient(audio_path: str, 
                          sensor_data: Optional[Dict] = None,
                          agent_mode: AgentMode = AgentMode.HYBRID,
                          api_key: Optional[str] = None):
    """
    Enhanced triage function with agentic reasoning
    Integrates with your existing system
    """
    # Import your existing functions
    # from your_module import transcribe_with_vad, extract_triage_entities, salt_rules, etc.
    
    print(f"\n{'='*60}")
    print(f"AGENTIC TRIAGE ASSESSMENT - {agent_mode.value.upper()} MODE")
    print(f"{'='*60}\n")
    
    # Initialize agent
    agent = TriageAgent(mode=agent_mode, api_key=api_key)
    
    # Step 1: Transcribe (your existing function)
    print("📝 Transcribing audio...")
    # transcription = transcribe_with_vad(audio_path)
    transcription = {"text": "Patient cannot walk, severe bleeding from leg, not responding to commands, respiratory rate 35 per minute"}
    
    # Step 2: Create triage state
    state = TriageState(
        patient_id=audio_path,
        transcription=transcription["text"],
        entities={},
        evidence=[],
        confidence=0.0,
        questions_asked=[],
        answers_received=[],
        sensor_data=sensor_data
    )
    
    # Step 3: LLM interpretation (enhanced entity extraction)
    print("\n🧠 LLM analyzing transcription...")
    llm_entities = agent.interpret_transcription(state)
    state.entities = llm_entities
    
    # Step 4: Rule-based extraction for comparison
    print("\n🔍 Rule-based extraction...")
    # rule_entities, evidence = extract_triage_entities(transcription["text"])
    rule_entities = {
        "can_walk": False,
        "bleeding_severe": True,
        "obeys_commands": False,
        "resp_rate": 35,
        "radial_pulse": None
    }
    
    # Step 5: Resolve contradictions if sensor data available
    if sensor_data:
        print("\n🤖 Resolving sensor/audio contradictions...")
        resolution = agent.resolve_contradictions(state, llm_entities, sensor_data)
        state.entities = resolution["resolved_entities"]
    
    # Step 6: Apply SALT rules
    print("\n🏥 Applying SALT protocol...")
    # rule_based_category = salt_rules(state.entities, sensor_data)
    rule_based_category = "Immediate"
    
    # Step 7: Agentic decision (can override rules)
    print("\n🎯 Agent making final decision...")
    final_category, reasoning, confidence = agent.make_triage_decision(
        state, state.entities, rule_based_category
    )
    
    # Step 8: Generate next question
    print("\n❓ Determining next question...")
    next_question = agent.generate_next_question(
        state, state.entities, final_category
    )
    
    # Step 9: Generate explanation
    print("\n📋 Generating explanation...")
    explanation = agent.explain_decision(state, final_category, state.entities)
    
    # Format results
    result = {
        "patient_id": audio_path,
        "transcription": transcription["text"],
        "entities": state.entities,
        "rule_based_category": rule_based_category,
        "final_category": final_category,
        "agent_reasoning": reasoning,
        "confidence": confidence,
        "next_question": next_question,
        "explanation": explanation,
        "reasoning_chain": state.reasoning_chain,
        "agent_mode": agent_mode.value
    }
    
    # Print results
    print(f"\n{'='*60}")
    print(f"AGENTIC TRIAGE RESULTS")
    print(f"{'='*60}")
    print(f"🚑 Rule-Based: {rule_based_category}")
    print(f"🎯 Agent Decision: {final_category}")
    if final_category != rule_based_category:
        print(f"   ⚠️  Agent OVERRODE rule-based assessment")
    print(f"\n📊 Confidence: {confidence*100:.0f}%")
    print(f"\n💭 Agent Reasoning:\n{reasoning}")
    print(f"\n📝 Clinical Explanation:\n{explanation}")
    
    if next_question:
        print(f"\n❓ Next Question: {next_question}")
    
    print(f"\n🧠 Reasoning Chain:")
    for step in state.reasoning_chain:
        print(f"  • {step}")
    
    print(f"{'='*60}\n")
    
    return result


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    import os
    
    # Example 1: Local mode (no API key needed)
    print("\n" + "="*60)
    print("EXAMPLE 1: LOCAL MODE")
    print("="*60)
    
    result_local = agentic_triage_patient(
        audio_path="test_patient.wav",
        agent_mode=AgentMode.LOCAL
    )
    
    # Example 2: Cloud mode with sensor fusion
    print("\n" + "="*60)
    print("EXAMPLE 2: CLOUD MODE WITH SENSORS")
    print("="*60)
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    
    mock_sensor_data = {
        "thermal_bleeding_detected": True,
        "movement_detected": False,
        "heart_rate": 140,
        "oxygen_saturation": 88
    }
    
    result_cloud = agentic_triage_patient(
        audio_path="test_patient.wav",
        sensor_data=mock_sensor_data,
        agent_mode=AgentMode.CLOUD,
        api_key=api_key
    )
    
    # Example 3: Hybrid mode (best of both)
    print("\n" + "="*60)
    print("EXAMPLE 3: HYBRID MODE")
    print("="*60)
    
    result_hybrid = agentic_triage_patient(
        audio_path="test_patient.wav",
        agent_mode=AgentMode.HYBRID,
        api_key=api_key
    )