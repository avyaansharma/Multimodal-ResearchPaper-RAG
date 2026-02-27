import os
import logging
import time
from typing import List, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# State definition
class AgentState(TypedDict):
    paper_a_content: str
    paper_b_content: str
    paper_a_summary: str
    paper_b_summary: str
    comparison: str
    gaps: str
    final_report: str

class ResearchAgentSystem:
    def __init__(self, model_name: str = "gemini-2.5-flash-lite"):
        self.api_key = os.getenv("GEMINI_API_KEY")
        # Initialize with retries and a focus on speed
        self.llm = ChatGoogleGenerativeAI(
            model=model_name, 
            google_api_key=self.api_key,
            max_retries=5,
            timeout=60
        )
        self.workflow = self._create_workflow()

    def _create_workflow(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("extractor", self.extract_info)
        workflow.add_node("comparer", self.compare_papers)
        workflow.add_node("gap_scout", self.scout_gaps)
        workflow.add_node("synthesizer", self.synthesize_report)

        workflow.set_entry_point("extractor")
        workflow.add_edge("extractor", "comparer")
        workflow.add_edge("comparer", "gap_scout")
        workflow.add_edge("gap_scout", "synthesizer")
        workflow.add_edge("synthesizer", END)

        return workflow.compile()

    def _safe_invoke(self, prompt: str):
        """Helper to invoke LLM with manual backoff if langchain retries aren't enough."""
        for i in range(3):
            try:
                return self.llm.invoke([HumanMessage(content=prompt)]).content
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    wait = (i + 1) * 5
                    logger.warning(f"Agent rate limit. Waiting {wait}s...")
                    time.sleep(wait)
                else:
                    raise e
        return "Internal Error: Rate limit exhausted even after retries."

    def extract_info(self, state: AgentState):
        logger.info("Agent: Extracting information (truncated to 10k chars)...")
        prompt = "Summarize core objective, methodology, and limitations:\n\n{content}"
        
        # Truncate to save tokens and avoid quota issues
        summary_a = self._safe_invoke(prompt.format(content=state['paper_a_content'][:10000]))
        summary_b = self._safe_invoke(prompt.format(content=state['paper_b_content'][:10000]))
        
        return {"paper_a_summary": summary_a, "paper_b_summary": summary_b}

    def compare_papers(self, state: AgentState):
        logger.info("Agent: Comparing evolution and improvements...")
        prompt = f"""Compare and find evolutions:\n
        A: {state['paper_a_summary']}\n
        B: {state['paper_b_summary']}"""
        return {"comparison": self._safe_invoke(prompt)}

    def scout_gaps(self, state: AgentState):
        logger.info("Agent: Searching for research gaps...")
        prompt = f"Find 3 research gaps based on:\nSummaries: {state['paper_a_summary']}, {state['paper_b_summary']}\nComparison: {state['comparison']}"
        return {"gaps": self._safe_invoke(prompt)}

    def synthesize_report(self, state: AgentState):
        logger.info("Agent: Finalizing report...")
        prompt = f"Generate final report with 'Scope for New Work':\nComparison: {state['comparison']}\nGaps: {state['gaps']}"
        report = self._safe_invoke(prompt)
        return {"report": report, "final_report": report}

    def run_analysis(self, paper_a_text: str, paper_b_text: str):
        initial_state = {
            "paper_a_content": paper_a_text,
            "paper_b_content": paper_b_text,
            "paper_a_summary": "",
            "paper_b_summary": "",
            "comparison": "",
            "gaps": "",
            "final_report": ""
        }
        return self.workflow.invoke(initial_state)

if __name__ == "__main__":
    # Test stub
    agent = ResearchAgentSystem()
    print("Agent system initialized.")
