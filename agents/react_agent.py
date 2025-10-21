import re, json
from typing import List, Dict, Any
from graph.openai_client import openai_generate as hf_generate
import logging
logger = logging.getLogger("agentic_chatbot.react")

class Reactor:
    def __init__(
        self,
        prompt_text: str,
        tools: List[Any],
        max_iters: int = 3,
        max_new_tokens: int = 600,
        temperature: float = 0.3
    ):
        self.base_prompt = prompt_text
        self.max_iters = max_iters
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.tools: Dict[str, Any] = {t.name: t for t in tools}
        logger.info(f"Initialized Reactor with {len(self.tools)} tools: {list(self.tools.keys())}")

    def _run_llm(self, prompt: str) -> str:
        logger.info(f"Running LLM call: tokens={len(prompt)}, temp={self.temperature}")
        try:
            result = hf_generate(prompt, max_output_tokens=self.max_new_tokens, temperature=self.temperature)
            logger.info(f"LLM call completed successfully ({len(result)} chars returned).")
            return result
        except Exception as e:
            logger.exception("LLM generation failed.")
            raise e

    @staticmethod
    def _extract_action(block: str) -> Dict[str, Any] | None:
        action_matches = list(re.finditer(r"(?i)^\s*Action\s*:\s*(.+)$", block, flags=re.MULTILINE))
        if not action_matches:
            return None
        tool_name = action_matches[-1].group(1).strip()
        after = block[action_matches[-1].end():]
        m_in = re.search(r"(?i)^\s*Action\s*Input\s*:\s*(.+)$", after, flags=re.MULTILINE | re.DOTALL)
        arg = (m_in.group(1).strip() if m_in else "").strip()
        if not arg:
            m2 = re.search(r"(?i)Action\s*Input\s*:\s*(.+)$", block, flags=re.MULTILINE | re.DOTALL)
            if not m2:
                return None
            arg = m2.group(1).strip()
        if arg.startswith("```"):
            arg = arg.split("\n", 1)[1]
            if arg.endswith("```"):
                arg = arg.rsplit("\n", 1)[0]
        return {"tool": tool_name, "input": arg}

    @staticmethod
    def _extract_final_json(text: str) -> dict:
        m = re.search(r"(?i)Final\s*:\s*(\{.*)$", text, flags=re.DOTALL)
        candidate = m.group(1).strip() if m else text.strip()
        if candidate.startswith("```"):
            candidate = candidate.split("\n", 1)[1]
            if candidate.endswith("```"):
                candidate = candidate.rsplit("\n", 1)[0]
        start, end = candidate.rfind("{"), candidate.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError("No JSON object found in Final output.")
        return json.loads(candidate[start:end+1])

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        transcript = self.base_prompt.strip()
        logger.info(f"Starting ReAct loop (max_iters={self.max_iters})")
        for iteration in range(self.max_iters):
            logger.info(f"Iteration {iteration+1}/{self.max_iters}")
            step_out = self._run_llm(transcript + "\n")

            # Try extracting final JSON directly
            try:
                data = self._extract_final_json(step_out)
                logger.info("Final JSON successfully extracted.")
                return {"output": step_out}
            except Exception:
                logger.debug("No valid Final JSON found yet; continuing ReAct loop.")

            # Extract action
            act = self._extract_action(step_out)
            if not act:
                logger.info("No Action found, checking for Final again or exiting loop.")
                try:
                    data = self._extract_final_json(step_out)
                    logger.info("Final JSON found after Action failure.")
                    return {"output": step_out}
                except Exception:
                    logger.warning("No Action or Final JSON found; breaking loop.")
                    break

            tool_name = act["tool"]
            logger.info(f"Detected tool action: {tool_name} with input snippet={act['input'][:80]}")

            tool = self.tools.get(tool_name)
            if not tool:
                logger.warning(f"Tool '{tool_name}' not available; skipping.")
                transcript += f"\nObservation: Tool '{tool_name}' not available. Choose a valid tool."
                continue

            try:
                logger.info(f"Invoking tool '{tool_name}'...")
                obs = tool.invoke(act["input"])
                logger.info(f"Tool '{tool_name}' completed successfully.")
            except Exception as e:
                logger.exception(f"Tool '{tool_name}' failed.")
                obs = json.dumps({"error": f"Tool '{tool_name}' failed: {e}"})

            # Log tool output (truncate for safety)
            logger.debug(f"Tool observation (truncated): {obs[:200]}")
            transcript += "\n" + step_out.strip() + f"\nObservation: {obs}\n"

        logger.warning("Max iterations reached or Final not found, forcing completion prompt.")
        final_out = self._run_llm(
            transcript + "\nPlease provide Final: the exact JSON now, nothing else.\n"
        )
        logger.info("ReAct fallback completion step executed.")
        return {"output": final_out}
