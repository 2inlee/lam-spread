import openai
import os
import json
import re
from tqdm import tqdm
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

# Task-specific input preparation
def prepare_input(task, row):
    if task == "24game":
        obs = f"Given the numbers {row['numbers']}, use +, -, *, / and parentheses to make the number 24."
        entities = row['numbers']
        gold = row['solutions'][0] if row['solutions'] else None
    elif task == "gsm8k":
        obs = row['question']
        entities = []
        gold = row['answer'].split("####")[-1].strip()
    elif task == "strategyqa":
        facts = '\n'.join(row['facts'])
        obs = f"Question: {row['question']}\nFacts:\n{facts}"
        entities = []
        gold = str(row['answer'])
    return obs, entities, gold

# Main evaluation function
def evaluate_task(task, samples=5):
    dataset = {
        "24game": load_dataset("nlile/24-game")['train'],
        "gsm8k": load_dataset("gsm8k", "main")['train'],
        "strategyqa": load_dataset("json", data_files="data/train.json")['train']
    }[task]

    correct, total = 0, 0
    for row in tqdm(dataset.select(range(samples))):
        obs, entities, gold = prepare_input(task, row)
        s1, s2, usage, _ = run_two_stage_plam(obs, entities)

        if not s2:
            continue

        match = re.search(r'\{.*\}', s2, re.DOTALL)
        pred = json.loads(match.group())['solution'] if match else ""

        is_correct = (task == "24game" and abs(eval(pred)-24)<1e-4) or pred.strip().lower()==gold.strip().lower()
        correct += int(is_correct)
        total += 1

        print(f"\nSample {total}: {'✅' if is_correct else '❌'}")
        print(f"Input: {obs}\nGold: {gold}\nPred: {pred}")

    print(f"\n✅ {task.upper()} Accuracy: {correct}/{total} ({correct/total:.2%})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices=["24game", "gsm8k", "strategyqa"], required=True)
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()
    evaluate_task(args.task, args.samples)
