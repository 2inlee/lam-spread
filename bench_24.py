import openai
import os
import json
import re
from tqdm import tqdm
import backoff
from openai.error import RateLimitError
from datetime import datetime
import argparse
from datasets import load_dataset

openai.api_key = os.getenv("OPENAI_API_KEY")

@backoff.on_exception(backoff.expo, RateLimitError, max_tries=6, max_time=60)
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

# ✅ P-LAM 6단계 프롬프트
def plam_prompt(observation: str, numbers: list):
    return f"""
# [SYSTEM STATE]
Current environment state:
- Numbers: {numbers}
- Observation: {observation}

----

# [PHASE 1: Situation Rephrasing]
Rephrase the given environment state in natural language. Describe what this situation represents, so a human can easily understand it.

----

# [PHASE 2: Intervention Decision]
Determine whether proactive intervention is necessary in this situation.
- Should the system intervene proactively? [Yes / No]
- Reason:

----

# [PHASE 3: Intent Inference]
Based on the current situation, what is the user's likely underlying goal?

----

# [PHASE 4: High-Level Goal Abstraction]
Rephrase the inferred goal in high-level, abstract terms suitable for planning and reasoning.

----

# [PHASE 5: Action Planning]
List the actions required to achieve the goal, and decompose them into step-by-step sub-tasks.

----

# [PHASE 6: Sub-task Reasoning & Final Solution]
Carry out the sub-tasks step by step using logical reasoning to solve the problem.

Return the final result in the following JSON format:
{{"numbers": {numbers}, "solution": "(8/(3-1))*3"}}
"""

# ✅ 기존 프롬프트들
def Baseline1_zeroshot(observation: str, numbers: list):
    return f"""Given the numbers {numbers}, use +, -, *, / and parentheses to make 24.
Use each number exactly once. Return answer in JSON:

{{"numbers": {numbers}, "solution": "(1+1+1)*8"}}"""

def Baseline2_cot(observation: str, numbers: list):
    return f"""Solve step-by-step using arithmetic reasoning:

{observation}

Think step by step. Final answer in JSON format:
{{"numbers": {numbers}, "solution": "(1+1+1)*8"}}"""

def aot_structured_prompt(observation: str, numbers: list):
    return f"""You are an expert reasoning agent using Algorithm of Thoughts (AoT).

--- Observation ---
{observation}
--------------------

Respond in this exact format:

## Domain Identification:
...

## Problem Definition:
...

## Success Criteria:
...

## Step-by-Step Search Process:
...

## Final Answer (JSON format):
{{"numbers": {numbers}, "solution": "(1+1+1)*8"}}"""

# ✅ LLM 호출
def run_prompt(prompt: str, mode: str = "aot"):
    system_prompt = {
        "aot": "You solve problems using structured decomposition (AoT).",
        "cot": "You solve problems using step-by-step reasoning.",
        "zero_shot": "You solve math puzzles directly.",
        "plam": "You are a proactive reasoning agent. Analyze environment, determine whether to act, infer goal, plan actions, and solve."
    }[mode]

    try:
        response = completions_with_backoff(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        content = response['choices'][0]['message']['content']
        usage = response['usage']
        return content, usage
    except Exception as e:
        print(f"❌ API Error: {e}")
        return None, None

# ✅ 평가 함수 (JSON 파일 기반)
def evaluate_game24_from_json(path: str, max_samples: int = 50, mode: str = "aot"):
    with open(path, 'r') as f:
        data = json.load(f)["rows"]
    _evaluate_samples(data, mode, max_samples, from_json=True)

# ✅ 평가 함수 (HF 데이터셋)
def evaluate_game24_from_hf(max_samples: int = 50, mode: str = "aot"):
    dataset = load_dataset("nlile/24-game")["train"]
    data = [{"row_idx": i, "row": row} for i, row in enumerate(dataset.select(range(max_samples)))]
    _evaluate_samples(data, mode, max_samples, from_json=False)

# ✅ 공통 평가 루틴
def _evaluate_samples(data, mode, max_samples, from_json):
    correct = 0
    total = 0
    log_path = f"logs/game24_logs_{mode}.jsonl"
    os.makedirs("logs", exist_ok=True)

    print(f"\n🧠 Evaluating Game of 24 | Mode: {mode} | Samples: {max_samples}\n")

    with open(log_path, 'w') as logfile:
        for sample in tqdm(data[:max_samples]):
            row_id = sample["row_idx"]
            row = sample["row"]
            numbers = row["numbers"]
            ground_truth = row["solutions"][0] if row["solutions"] else "N/A"
            observation = f"Given the numbers {numbers}, use +, -, *, / and parentheses to make the number 24."

            # ✅ 모드에 따라 프롬프트 선택
            if mode == "aot":
                prompt = aot_structured_prompt(observation, numbers)
            elif mode == "cot":
                prompt = Baseline2_cot(observation, numbers)
            elif mode == "zero_shot":
                prompt = Baseline1_zeroshot(observation, numbers)
            elif mode == "plam":
                prompt = plam_prompt(observation, numbers)
            else:
                raise ValueError("Invalid mode. Use one of: aot, cot, zero_shot, plam")

            llm_response, usage = run_prompt(prompt, mode)
            error_type = None
            parsed_expr = None
            eval_result = None
            is_correct = False

            if not llm_response:
                error_type = "llm_no_response"
            else:
                try:
                    match = re.search(r"\{.*\}", llm_response, re.DOTALL)
                    parsed_json = json.loads(match.group()) if match else None
                    parsed_expr = parsed_json["solution"] if parsed_json else None
                except Exception:
                    error_type = "parse_error"

                if parsed_expr:
                    parsed_expr = parsed_expr.replace("×", "*").replace("x", "*")
                    parsed_expr = re.sub(r"[^\d\+\-\*\/\(\)\.]", "", parsed_expr)
                    try:
                        eval_result = eval(parsed_expr)
                        is_correct = abs(eval_result - 24) < 1e-4
                        error_type = None if is_correct else "wrong_result"
                    except Exception:
                        error_type = "eval_error"

            total += 1
            if is_correct:
                correct += 1

            print(f"\n🧩 Input:      {numbers}")
            print(f"🎯 Ground truth: {ground_truth}")
            print(f"🤖 LLM raw:    {llm_response[:150] if llm_response else 'N/A'}")
            print(f"✅ Parsed:     {parsed_expr}")
            print(f"📌 Result:     {'✅ Correct' if is_correct else '❌ Incorrect'}")

            log_entry = {
                "id": row_id,
                "input_numbers": numbers,
                "ground_truth": ground_truth,
                "prompt_type": mode,
                "prompt": prompt,
                "llm_response": llm_response,
                "parsed_expression": parsed_expr,
                "eval_result": eval_result,
                "is_correct": is_correct,
                "error_type": error_type,
                "input_tokens": usage["prompt_tokens"] if usage else None,
                "output_tokens": usage["completion_tokens"] if usage else None,
                "total_tokens": usage["total_tokens"] if usage else None,
                "timestamp": datetime.utcnow().isoformat()
            }
            logfile.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    accuracy = correct / total if total > 0 else 0.0
    print(f"\n✅ Final Accuracy ({mode}): {correct}/{total} ({accuracy:.2%})")
    print(f"📁 Logs saved to: {log_path}")

# ✅ CLI 엔트리포인트
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="aot", choices=["aot", "cot", "zero_shot", "plam"], help="Prompting strategy to use")
    parser.add_argument("--file", type=str, default="game24.json", help="Path to dataset file (use 'hf' to load from Hugging Face)")
    parser.add_argument("--samples", type=int, default=5, help="Number of samples to evaluate")
    args = parser.parse_args()

    if args.file == "hf":
        evaluate_game24_from_hf(max_samples=args.samples, mode=args.mode)
    else:
        evaluate_game24_from_json(path=args.file, max_samples=args.samples, mode=args.mode)



# P-LAM 방식 평가 (Hugging Face 데이터셋 사용)
#python3 bench_24.py --mode plam --file hf --samples 1362

# P-LAM 방식 평가 (로컬 JSON)
#python3 bench_24.py --mode plam --file game24.json --samples 20