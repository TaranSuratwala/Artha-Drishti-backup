"""
Validates that LLM-generated responses only contain numbers
that originated from tool calls in the same turn.
"""
import re
from typing import Dict, Set

class AgentOutputValidator:
    """Post-LLM validation to catch financial hallucinations."""

    def __init__(self):
        self.tool_numbers: Set[str] = set()

    def register_tool_output(self, tool_result: str):
        """Extract all numbers from a tool's output."""
        numbers = re.findall(r'[\d,]+\.?\d*', str(tool_result))
        self.tool_numbers.update(numbers)

    def validate_response(self, llm_response: str) -> Dict:
        """Check if the LLM cited numbers not from tool outputs."""
        # Extract numbers from LLM response
        response_numbers = re.findall(
            r'₹\s*([\d,]+\.?\d*)|'         # ₹-prefixed
            r'(\d+\.?\d*)\s*%|'             # Percentage
            r'\b(\d{2,}\.?\d{0,2})\b',      # 2+ digit numbers
            llm_response
        )
        cited = set()
        for groups in response_numbers:
            for n in groups:
                if n:
                    cited.add(n.replace(',', ''))

        # Numbers in LLM output but not in any tool output
        hallucinated = cited - self.tool_numbers
        # Filter out common non-financial numbers
        hallucinated = {n for n in hallucinated
                       if float(n) > 1 and n not in {'100', '10', '50', '20'}}

        return {
            "valid": len(hallucinated) == 0,
            "potentially_hallucinated_numbers": list(hallucinated),
            "total_numbers_cited": len(cited),
            "total_numbers_from_tools": len(self.tool_numbers),
        }
