import openai
import os

# 환경 변수에서 OpenAI API 키 불러오기
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key

def dot_full_autonomous_prompt(observation: str):
    return f"""
You are an advanced reasoning assistant that dynamically identifies and solves tasks using structured thinking like Algorithm of Thoughts (AoT).

Your task is as follows:
1. From the user's observation, identify the domain (e.g., spreadsheet, math puzzle, logs, text).
2. Define a task the user is likely trying to solve.
3. Specify the success criteria.
4. Construct **three** in-context examples using **diverse reasoning or search strategies**. Each example should solve the same type of problem in a different way (e.g., different starting points, different order of operations, different heuristics).
5. Use those examples to solve the actual problem.

--- Observation ---
{observation}
--------------------

Respond in the following format:

## Domain Identification:
...

## Problem Definition:
...

## Success Criteria:
...

## In-Context Examples (AoT style):
### Example 1:
...

### Example 2:
...

### Example 3:
...

## Solution to the Actual Task:
...
"""

def run_autonomous_dot(observation: str, temperature: float = 0.3):
    prompt = dot_full_autonomous_prompt(observation)

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a reasoning assistant that can dynamically generate tasks, few-shot examples, and solve problems via algorithmic thinking."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature
    )

    return response['choices'][0]['message']['content']


# 🔁 사용자 입력을 통해 observation을 받아 실행
if __name__ == "__main__":
    print("🔍 Enter your environment observation (multi-line supported). End with an empty line:")
    lines = []
    while True:
        line = input()
        if line.strip() == "":
            break
        lines.append(line)
    observation_input = "\n".join(lines)

    # 실행
    print("\n⏳ Reasoning in progress...\n")
    result = run_autonomous_dot(observation_input)
    print(result)