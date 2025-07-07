import openai
import os
import json
import re
from tqdm import tqdm
import backoff
from openai.error import RateLimitError
from datetime import datetime
import argparse
from datasets import Dataset

openai.api_key = os.getenv("OPENAI_API_KEY")

@backoff.on_exception(backoff.expo, RateLimitError, max_tries=6, max_time=60)
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
        print(f"\u274c API Error: {e}")
        return None, None

def plam_stage1_prompt(observation: str, entities: list):
    return f"""
You are a reflective reasoning agent. Your first task is to understand the situation, identify the domain, and extract relevant abstract knowledge before attempting a solution.

Follow these steps:

# [ENVIRONMENT STATE]
- Entities: {entities}
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
Examples include:
- Arithmetic problems: order of operations, associativity, commutativity  
- Physical tasks: gravity, stacking stability, size and weight constraints  
- Scheduling tasks: precedence, time windows, dependencies  
- Logic puzzles: deduction, mutual exclusivity, state transitions

Be sure to extract **domain-relevant rules** that would inform your planning.

----

# [STEP 5: Should You Intervene?]
- Intervention: [Yes / No]  
- Reason:

----

If Intervention is **Yes**, also return a structured context summary to inform the next planning phase.

Format:
{{"rephrased": "...", "goal": "...", "domain": "...", "principles": "...", "intervention": "Yes"}}  
"""

def plam_stage2_prompt(context: str, entities: list):
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

{{"entities": {entities}, "solution": "Yes" or "No"}}  
"""

def run_two_stage_plam(observation: str, entities: list):
    stage1 = plam_stage1_prompt(observation, entities)
    stage1_output, _ = run_prompt(stage1)

    if not stage1_output or "Intervention: No" in stage1_output:
        return stage1_output, None, None

    planning_context = stage1_output
    stage2 = plam_stage2_prompt(planning_context, entities)
    stage2_output, usage = run_prompt(stage2)

    return stage1_output, stage2_output, usage

def load_clean_strategyqa_dataset(samples: int):
    import json
    json_path = os.path.expanduser("~/.cache/huggingface/hub/datasets--voidful--StrategyQA/snapshots/2279eaf9f2580aef77ed6fa0efd7846c381ab5a0/strategyqa_train.json")
    with open(json_path, "r") as f:
        raw_data = json.load(f)
    clean_data = [{k: v for k, v in ex.items() if k != "evidence"} for ex in raw_data[:samples]]
    return Dataset.from_list(clean_data)

def evaluate_strategyqa(samples: int):
    dataset = load_clean_strategyqa_dataset(samples)
    correct = 0
    total = 0
    os.makedirs("logs", exist_ok=True)
    log_path = f"logs/strategyqa_logs_plam.jsonl"

    with open(log_path, 'w') as logfile:
        for i, row in enumerate(tqdm(dataset)):
            question = row["question"]
            answer = "yes" if row["answer"] else "no"
            observation = f"Question: {question}"
            entities = [row["term"]]

            stage1, stage2, usage = run_two_stage_plam(observation, entities)
            prompt_used = stage1 + "\n---\n" + (stage2 or "")
            llm_response = stage2

            predicted_answer = None
            is_correct = False
            error_type = None

            if llm_response:
                try:
                    match = re.search(r"\{.*\}", llm_response, re.DOTALL)
                    parsed_json = json.loads(match.group()) if match else None
                    predicted_answer = parsed_json.get("solution", "").strip().lower()
                    if predicted_answer in ["yes", "no"]:
                        is_correct = predicted_answer == answer
                except:
                    error_type = "parse_error"
            else:
                error_type = "llm_no_response"

            total += 1
            if is_correct:
                correct += 1

            print(f"\n❓ Question: {question}")
            print(f"✅ Ground Truth : {answer}")
            print(f"🧠 Stage 1 Output:\n{stage1}")
            print(f"🤖 Stage 2 Response:\n{stage2}\n")
            print(f"📌 Predicted    : {predicted_answer}")
            print(f"📍 Result       : {'✅ Correct' if is_correct else '❌ Incorrect'}")

            log_entry = {
                "id": i,
                "question": question,
                "ground_truth": answer,
                "prompt": prompt_used,
                "llm_response": llm_response,
                "predicted_answer": predicted_answer,
                "is_correct": is_correct,
                "error_type": error_type,
                "input_tokens": usage["prompt_tokens"] if usage else None,
                "output_tokens": usage["completion_tokens"] if usage else None,
                "total_tokens": usage["total_tokens"] if usage else None,
                "timestamp": datetime.utcnow().isoformat()
            }
            logfile.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    accuracy = correct / total if total > 0 else 0.0
    print(f"\n✅ Final Accuracy: {correct}/{total} ({accuracy:.2%})")
    print(f"📁 Logs saved to: {log_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=5, help="Number of samples to evaluate")
    args = parser.parse_args()
    evaluate_strategyqa(samples=args.samples)
