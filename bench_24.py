import openai
import os
import json
import re
import sys
from tqdm import tqdm
import time

openai.api_key = os.getenv("OPENAI_API_KEY")

# ✅ AoT 방식 프롬프트
def aot_structured_prompt(observation: str):
    return f"""
You are an expert reasoning agent that solves problems using **Algorithm of Thoughts (AoT)**.

Your method:
- Begin with identifying the domain.
- Decompose the task into **explicit subtasks** using a structured search process.
- Mark each reasoning step using numbers and headings (e.g. "Subtask 1", "Atomic Subtask 1").
- Provide intermediate analyses for each subtask.
- At the end, **combine subtask results** into a final solution.

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

## Final Answer:
...
"""

# ✅ Baseline 1: Zero-shot 방식
def Baseline1_zeroshot(observation: str):
    return observation

# ✅ Baseline 2: Chain-of-Thought 방식
def Baseline2_cot(observation: str):
    return f"""
Solve the following problem step by step using arithmetic reasoning:

{observation}

Let's think step by step.
"""

# ✅ 공통 LLM 실행 함수
def run_prompt(prompt: str, mode: str = "aot"):
    system_prompt = {
        "aot": "You solve problems using structured decomposition and search as in Algorithm of Thoughts.",
        "cot": "You solve problems using step-by-step reasoning.",
        "zero_shot": "You solve math puzzles directly."
    }[mode]

    # time.sleep(1.2)  # ✅ 요청 전 딜레이 (초당 1~2회 허용 기준)

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        print(f"❌ API Error: {e}")
        return None

# ✅ 후처리: 수식 추출용 LLM 호출
def extract_expression_from_llm_response(llm_response: str) -> str:
    prompt = f"""
You are a math expression cleaner.

Given the following LLM output, extract only the **final numeric expression** used to compute 24.
- Remove any explanations or commentary.
- Return a clean, parsable Python expression (e.g., (1+1+1)*8).
- Replace '×', 'x', etc. with '*'
- Do NOT include '= 24' or any trailing numbers.

--- LLM Output ---
{llm_response}
------------------

Cleaned Expression:
"""
    # time.sleep(1.2)  # ✅ 요청 전 딜레이 (초당 1~2회 허용 기준)

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You extract clean mathematical expressions from noisy outputs."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"❌ Post-processing LLM error: {e}")
        return None

# ✅ 평가 함수
def evaluate_game24_from_json(path: str, max_samples: int = 50, mode: str = "aot"):
    with open(path, 'r') as f:
        data = json.load(f)["rows"]

    correct = 0
    total = 0

    print(f"\n🧠 Evaluating Game of 24 | Mode: {mode} | Samples: {max_samples}\n")

    for sample in tqdm(data[:max_samples]):
        row = sample["row"]
        numbers = row["numbers"]
        solutions = row["solutions"]
        observation = f"Given the numbers {numbers}, use +, -, *, / and parentheses to make the number 24."

        if mode == "aot":
            prompt = aot_structured_prompt(observation)
        elif mode == "cot":
            prompt = Baseline2_cot(observation)
        elif mode == "zero_shot":
            prompt = Baseline1_zeroshot(observation)
        else:
            raise ValueError("Invalid mode. Use one of: aot, cot, zero_shot")

        llm_response = run_prompt(prompt, mode)
        if not llm_response:
            print("⚠️ No LLM response.")
            continue

        cleaned = extract_expression_from_llm_response(llm_response)
        if not cleaned:
            print("⚠️ Failed to extract expression.")
            continue

        # 안전 처리
        expr = cleaned.replace("×", "*").replace("x", "*")
        expr = re.sub(r"[^\d\+\-\*\/\(\)\.]", "", expr)

        try:
            result = eval(expr)
            is_correct = abs(result - 24) < 1e-4
        except Exception as e:
            print(f"❌ Eval error: {e}")
            is_correct = False

        print(f"\n🧩 Input:      {numbers}")
        print(f"🎯 Ground truth: {solutions[0] if solutions else 'N/A'}")
        print(f"🤖 LLM raw:    {llm_response[:150]}...")
        print(f"✅ Parsed:     {expr}")
        print(f"📌 Result:     {'✅ Correct' if is_correct else '❌ Incorrect'}")

        total += 1
        if is_correct:
            correct += 1

    accuracy = correct / total if total > 0 else 0.0
    print(f"\n✅ Final Accuracy ({mode}): {correct}/{total} ({accuracy:.2%})")

# ✅ 명령행 파라미터 기반 실행
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="aot", choices=["aot", "cot", "zero_shot"], help="Prompting strategy to use")
    parser.add_argument("--file", type=str, default="game24.json", help="Path to dataset file")
    parser.add_argument("--samples", type=int, default=5, help="Number of samples to evaluate")
    args = parser.parse_args()

    evaluate_game24_from_json(path=args.file, max_samples=args.samples, mode=args.mode)


# AoT 방식 평가
# python3 bench_24.py --mode aot --samples 10

# Zero-shot baseline 평가
# python3 bench_24.py --mode zero_shot --samples 10

# Chain-of-Thought baseline 평가
# python3 bench_24.py --mode cot --samples 10