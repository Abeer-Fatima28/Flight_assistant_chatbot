import os
import json
import logging
from typing import Dict, Callable, Any, List
from openai import OpenAI

logger = logging.getLogger("agentic_chatbot.openai")

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def openai_generate(prompt: str, max_output_tokens: int = 700, temperature: float = 0.3) -> str:
    logger.info("openai_generate(chat.completions) call")
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_output_tokens,
    )
    return resp.choices[0].message.content or ""

def _chat_tools_from_responses_tools(responses_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    chat_tools = []
    for t in responses_tools:
        if t.get("type") != "function":
            continue
        chat_tools.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("parameters", {"type": "object", "properties": {}})
            }
        })
    return chat_tools

def openai_tool_loop(
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    dispatch: Dict[str, Callable[[str], str]],
    *,
    max_rounds: int = 4,
    temperature: float = 0.2,
    max_output_tokens: int = 700,
    finalizer_prompt: str = "Return ONLY the final JSON now. No backticks, no commentary.",
):
    logger.info(f"Starting CC tool loop with {len(tools)} tools; max_rounds={max_rounds}")

    chat_tools = _chat_tools_from_responses_tools(tools)
    chat_messages = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        chat_messages.append({"role": role, "content": content})

    for round_idx in range(1, max_rounds + 1):
        logger.info(f"CC Round {round_idx} -> calling model with {len(chat_messages)} messages")
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=chat_messages,
            tools=chat_tools if chat_tools else None,
            tool_choice="auto" if chat_tools else None,
            temperature=temperature,
            max_tokens=max_output_tokens,
        )

        msg = resp.choices[0].message
        tool_calls = msg.tool_calls or []

        if tool_calls:
            logger.info(f"Model issued {len(tool_calls)} tool call(s)")
            chat_messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    } for tc in tool_calls
                ]
            })

            for tc in tool_calls:
                name = tc.function.name
                args_str = tc.function.arguments or "{}"
                logger.info(f"Executing tool '{name}' with args: {args_str[:200]}")
                try:
                    fn = dispatch.get(name)
                    out_text = fn(args_str) if fn else json.dumps({"error": f"Unknown tool: {name}"})
                    logger.info(f"Tool '{name}' ok; output length={len(out_text)}")
                except Exception as e:
                    logger.exception(f"Tool '{name}' failed.")
                    out_text = json.dumps({"error": f"Tool '{name}' failed: {e}"})

                chat_messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": name,
                    "content": out_text
                })
            continue
        if msg.content and msg.content.strip():
            logger.info("Assistant produced final text (no further tool calls).")
            return resp

        logger.info("No content emitted; asking final JSON.")
        chat_messages.append({"role": "user", "content": finalizer_prompt})

    logger.warning("Max rounds reached; requesting final JSON once more.")
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=chat_messages + [{"role": "user", "content": finalizer_prompt}],
        tools=chat_tools if chat_tools else None,
        tool_choice="none",
        temperature=temperature,
        max_tokens=max_output_tokens,
    )
    return resp
