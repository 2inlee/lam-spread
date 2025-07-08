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

def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def run_prompt(prompt: str):
    try:
        response = completions_with_backoff(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a proactive reasoning agent. Understand the environment, infer goals, plan, and solve."},
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

def plam_stage1_prompt(observation: str, numbers: list):
    return f"""
You are a reflective reasoning agent. Your first task is to understand the situation, identify the domain, and extract relevant abstract knowledge before attempting a solution.

Follow these steps:

# [ENVIRONMENT STATE]
- Entities: {numbers}
- Observation: {observation}

----

# [STEP 1: Rephrase the Situation]
Briefly summarize what's happening based on the observation and entities.

----

# [STEP 2: Infer the User's Goal]
What is the user likely trying to accomplish?

----

# [STEP 3: Identify Domain & Task Type]
Classify the type of problem or domain (e.g., arithmetic puzzle, planning task, scheduling, manipulation, constraint satisfaction, etc.)

----

# [STEP 4: Extract Relevant High-Level Principles or Constraints]
Based on the domain, identify key concepts or constraints that should be considered when solving this type of task.  

----

# [STEP 5: Should You Intervene?]
- Intervention: [Yes / No]  
- Reason:

----

If Intervention is **Yes**, also return a structured context summary to inform the next planning phase.

Format:
{{"rephrased": "...", "goal": "...", "domain": "...", "principles": "...", "intervention": "Yes"}}  
"""

def plam_stage2_prompt(context: str, numbers: list):
    return f"""
You are now in execution mode.

Use the following context, which includes the problem setting, user goal, domain, and relevant reasoning principles, to generate a solution plan and execute it.

# [CONTEXT FROM STAGE 1]
{context}

----

Now follow these steps:

1. Plan a high-level approach based on the domain and constraints.
2. Break down the plan into subtasks.
3. Execute each subtask step-by-step.
4. After each trial, evaluate whether the goal is satisfied.
5. If failed, revise your plan and try again (up to 5 trials).

Use this format:

### Trial N:
- Subtask execution steps:
- Result (e.g., expression, configuration, decision):
- Evaluation: [Success / Failure]
- If failed: Briefly explain what went wrong and revise.

At the end, return:

{{"entities": {numbers}, "solution": "(...)"}}  
"""

def run_two_stage_plam(observation: str, numbers: list):
    stage1_prompt = plam_stage1_prompt(observation, numbers)
    stage1_output, _ = run_prompt(stage1_prompt)

    if not stage1_output or "Intervention: No" in stage1_output:
        return stage1_output, None, None, None

    context = stage1_output
    stage2_prompt = plam_stage2_prompt(context, numbers)
    stage2_output, usage = run_prompt(stage2_prompt)

    return stage1_output, stage2_output, usage, stage2_prompt

def evaluate_game24(samples: int, use_random: bool):
    dataset = load_dataset("nlile/24-game")['train']
    if use_random:
        dataset = dataset.shuffle(seed=42)
    data = [{"row_idx": i, "row": row} for i, row in enumerate(dataset.select(range(samples)))]

    correct = 0
    total = 0
    os.makedirs("logs", exist_ok=True)
    log_path = "logs/game24_logs_plam.jsonl"

    with open(log_path, 'w') as logfile:
        for sample in tqdm(data):
            row_id = sample["row_idx"]
            row = sample["row"]
            numbers = row["numbers"]
            ground_truth = row["solutions"][0] if row["solutions"] else "N/A"
            observation = f"Given the numbers {numbers}, use +, -, *, / and parentheses to make the number 24."

            stage1, stage2, usage, stage2_prompt = run_two_stage_plam(observation, numbers)
            prompt_used = (stage1 or "") + "\n---\n" + (stage2 or "")
            llm_response = stage2

            error_type = parsed_expr = eval_result = None
            is_correct = False
            solved_at_trial = None

            if llm_response:
                try:
                    trials = re.findall(r"### Trial (\d+):.*?Evaluation:\s*\[(Success|Failure)\]", llm_response, re.DOTALL)
                    for trial_num, result in trials:
                        if result.strip().lower() == "success":
                            solved_at_trial = int(trial_num)
                            break

                    match = re.search(r"\{.*\}", llm_response, re.DOTALL)
                    parsed_json = json.loads(match.group()) if match else None
                    parsed_expr = parsed_json["solution"] if parsed_json else None
                except:
                    error_type = "parse_error"

                if parsed_expr:
                    parsed_expr = parsed_expr.replace("×", "*").replace("x", "*")
                    parsed_expr = re.sub(r"[^\d\+\-\*\/\(\)\.]", "", parsed_expr)
                    try:
                        eval_result = eval(parsed_expr)
                        is_correct = abs(eval_result - 24) < 1e-4
                        error_type = None if is_correct else "wrong_result"
                    except:
                        error_type = "eval_error"
            else:
                error_type = "llm_no_response"

            total += 1
            if is_correct:
                correct += 1

            print(f"\n🧩 Input Numbers: {numbers}")
            print(f"🎯 Ground Truth : {ground_truth}")
            print(f"🧠 Stage 1 Output:\n{stage1}")
            print(f"🤖 Stage 2 Response:\n{stage2}")
            print(f"🔁 Solved At Trial: {solved_at_trial}")
            print(f"✅ Parsed       : {parsed_expr}")
            print(f"📌 Result       : {'✅ Correct' if is_correct else '❌ Incorrect'}")

            log_entry = {
                "id": row_id,
                "input_numbers": numbers,
                "ground_truth": ground_truth,
                "prompt": prompt_used,
                "llm_response": llm_response,
                "parsed_expression": parsed_expr,
                "eval_result": eval_result,
                "is_correct": is_correct,
                "error_type": error_type,
                "input_tokens": usage["prompt_tokens"] if usage else None,
                "output_tokens": usage["completion_tokens"] if usage else None,
                "total_tokens": usage["total_tokens"] if usage else None,
                "solved_at_trial": solved_at_trial,
                "timestamp": datetime.utcnow().isoformat()
            }
            logfile.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    accuracy = correct / total if total > 0 else 0.0
    print(f"\n✅ Final Accuracy: {correct}/{total} ({accuracy:.2%})")
    print(f"📁 Logs saved to: {log_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=5, help="Number of samples to evaluate")
    parser.add_argument("--random", action="store_true", help="Use random sampling of the dataset")
    args = parser.parse_args()
    evaluate_game24(samples=args.samples, use_random=args.random)
