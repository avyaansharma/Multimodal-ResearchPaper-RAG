import os
import logging
import time
from typing import List, TypedDict, Annotated, Dict
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# State definition
class AnalyzerState(TypedDict):
    content: str
    topic: str
    claim: str
    evidence: str
    method: str
    dataset: str
    limitations: str
    citation: str
    critic_feedback: str
    iterations: int
    final_report: str

class PaperAnalyzer:
    def __init__(self, model_name: str = "gemini-2.5-flash-lite"):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.llm = ChatGoogleGenerativeAI(
            model=model_name, 
            google_api_key=self.api_key,
            max_retries=5,
            timeout=120
        )
        self.workflow = self._create_workflow()

    def _create_workflow(self):
        workflow = StateGraph(AnalyzerState)
        
        # Add basic extraction nodes
        workflow.add_node("extract_identity", self.extract_identity)
        workflow.add_node("extract_argument", self.extract_argument)
        workflow.add_node("extract_execution", self.extract_execution)
        workflow.add_node("extract_critique", self.extract_critique)
        
        # Add a critic node for non-linear flow
        workflow.add_node("critic", self.review_analysis)
        
        # Add synthesizer
        workflow.add_node("synthesizer", self.synthesize_report)

        # Build graph
        workflow.set_entry_point("extract_identity")
        
        # Parallel-ish flow (sequential for simplicity in LangGraph edges but logic is separate)
        workflow.add_edge("extract_identity", "extract_argument")
        workflow.add_edge("extract_argument", "extract_execution")
        workflow.add_edge("extract_execution", "extract_critique")
        workflow.add_edge("extract_critique", "critic")
        
        # Non-linear edge: Critic can send back to argument extractor if claim is too short/long
        workflow.add_conditional_edges(
            "critic",
            self.should_continue,
            {
                "refine": "extract_argument",
                "finish": "synthesizer"
            }
        )
        
        workflow.add_edge("synthesizer", END)

        return workflow.compile()

    def _safe_invoke(self, prompt: str):
        """Helper to invoke LLM with robust backoff for rate limiting."""
        for i in range(5):
            try:
                logger.info(f"Invoking LLM (attempt {i+1})...")
                response = self.llm.invoke([HumanMessage(content=prompt)]).content
                logger.info("LLM Invocation successful.")
                return response
            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "quota" in err_str or "rate limit" in err_str:
                    wait = (i + 1) * 10 
                    logger.warning(f"Rate limit hit. Waiting {wait}s...")
                    time.sleep(wait)
                else:
                    logger.error(f"LLM Error in _safe_invoke: {e}")
                    raise e
        return "Error: Rate limit persistent after retries."

    def extract_identity(self, state: AnalyzerState):
        logger.info("Extracting Identity (Topic & Citation)...")
        content = state["content"][:15000] # Use first 15k chars for identity
        prompt = f"""Extract the following from the research paper text:
1. Topic: A concise title/topic of the research.
2. Citation: Full academic citation if available, otherwise 'Author et al. (Year)'.

Text:
{content}

Format:
Topic: <topic>
Citation: <citation>"""
        response = self._safe_invoke(prompt)
        # Simple parsing
        lines = response.split("\n")
        topic = next((l.split("Topic:")[1].strip() for l in lines if "Topic:" in l), "Unknown")
        citation = next((l.split("Citation:")[1].strip() for l in lines if "Citation:" in l), "Unknown")
        return {"topic": topic, "citation": citation, "iterations": state.get("iterations", 0) + 1}

    def extract_argument(self, state: AnalyzerState):
        logger.info("Extracting Argument (Claim & Evidence)...")
        content = state["content"][:20000]
        feedback = state.get("critic_feedback", "")
        feedback_str = f"\nCorrection Needed: {feedback}" if feedback else ""
        
        prompt = f"""Perform a deep extraction of the core argument:
1. Claim: A dense, detailed paragraph explaining the main thesis, the technical problem solved, and the core innovation. This MUST be exactly 5-6 lines of high-quality, academic text.
2. Supporting Evidence: List the 3-5 most critical findings, metrics, or experimental results that prove the claim. Use specific numbers if available.
{feedback_str}

Text:
{content}

Format:
Claim: <detailed 5-6 line paragraph>
Evidence: <specific bullet points>"""
        response = self._safe_invoke(prompt)
        lines = response.split("\n")
        claim = ""
        evidence = ""
        in_claim = False
        in_evidence = False
        for line in lines:
            if "Claim:" in line:
                claim = line.split("Claim:")[1].strip()
                in_claim = True
                continue
            if "Evidence:" in line:
                evidence = line.split("Evidence:")[1].strip()
                in_claim = False
                in_evidence = True
                continue
            if in_claim and line.strip():
                claim += " " + line.strip()
            elif in_evidence and line.strip():
                evidence += "\n" + line.strip()
        
        return {"claim": claim, "evidence": evidence}

    def extract_execution(self, state: AnalyzerState):
        logger.info("Extracting Execution (Method & Dataset)...")
        # Methods are often in the middle, but we'll try to find sections
        content = state["content"]
        prompt = f"""Identify the methodology and datasets:
1. Method Used: Concise description of the experimental or theoretical approach.
2. Dataset: Minimal list of dataset names only. Do NOT add descriptors. Just the names.

Text:
{content[:30000]}

Format:
Method: <method>
Dataset: <dataset names>"""
        response = self._safe_invoke(prompt)
        lines = response.split("\n")
        method = next((l.split("Method:")[1].strip() for l in lines if "Method:" in l), "Unknown")
        dataset = next((l.split("Dataset:")[1].strip() for l in lines if "Dataset:" in l), "N/A")
        return {"method": method, "dataset": dataset}

    def extract_critique(self, state: AnalyzerState):
        logger.info("Extracting Critique (Limitations)...")
        content = state["content"][-15000:] # Limitations are usually at the end
        prompt = f"""Identify the limitations stated by the authors:
Limitations: <concise list>

Text:
{content}"""
        response = self._safe_invoke(prompt)
        limitations = response.replace("Limitations:", "").strip()
        return {"limitations": limitations}

    def review_analysis(self, state: AnalyzerState):
        logger.info("Critiquing analysis for quality...")
        claim = state["claim"]
        dataset = state["dataset"]
        
        # Count words as a proxy for lines (aiming for ~70-120 words for 5-6 detailed lines)
        word_count = len(claim.split())
        is_claim_ok = 70 < word_count < 130 # Target for detailed 5-6 lines
        
        if not is_claim_ok and state["iterations"] < 3:
            feedback = "The claim is too brief. Please expand it to a highly detailed paragraph of exactly 5-6 lines, covering the problem, the specific solution, and the primary innovation." if word_count <= 70 else "The claim is slightly too wordy. Please condense it to a very dense and informative 5-6 lines."
            return {"critic_feedback": feedback}
        
        return {"critic_feedback": ""}

    def should_continue(self, state: AnalyzerState):
        if state.get("critic_feedback") and state["iterations"] < 3:
            return "refine"
        return "finish"

    def synthesize_report(self, state: AnalyzerState):
        logger.info("Synthesizing final report...")
        report = f"""# 📄 Paper Overview (Short Notes)

## 📌 Topic
{state['topic']}

## 📝 Citation
{state['citation']}

---

## 🎯 Core Claim
{state['claim']}

## 📊 Supporting Evidence
{state['evidence']}

---

## ⚙️ Methodology
{state['method']}

## 💾 Dataset
{state['dataset']}

---

## ⚠️ Limitations
{state['limitations']}
"""
        return {"final_report": report}

    def run_analysis(self, paper_text: str):
        initial_state = {
            "content": paper_text,
            "topic": "",
            "claim": "",
            "evidence": "",
            "method": "",
            "dataset": "",
            "limitations": "",
            "citation": "",
            "critic_feedback": "",
            "iterations": 0,
            "final_report": ""
        }
        return self.workflow.invoke(initial_state)

if __name__ == "__main__":
    # Test stub
    analyzer = PaperAnalyzer()
    print("Paper Analyzer initialized.")
