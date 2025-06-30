import openai
import os

api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key

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

### 1. Start the Search:
Explain how you initiate the reasoning process and what the first operation to try is.

### 2. Subtask 1:
Decompose the main problem into a meaningful first-level subtask.

### 3. Subtask 2:
Break Subtask 1 further into atomic components.

### 4. Atomic Subtask 1:
Analyze and solve this atomic subtask.

### 5. Atomic Subtask 2:
Analyze and solve this atomic subtask.

### 6. Atomic Subtask 3:
(If needed) Analyze and solve.

...

### 7. Combine Results:
Integrate all atomic results to form the solution for Subtask 2.

### 8. Solve Remaining Subtasks:
Repeat decomposition or analysis for remaining parts of the original problem.

### 9. Final Consolidation:
Bring all intermediate results together into the final solution.

## Final Answer:
...
"""

def run_structured_aot(observation: str, temperature: float = 0.3):
    prompt = aot_structured_prompt(observation)

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You solve problems using structured decomposition and search as in Algorithm of Thoughts."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature
    )

    return response['choices'][0]['message']['content']


if __name__ == "__main__":
    print("🔍 Enter your environment observation (multi-line supported). End with an empty line:")
    lines = []
    while True:
        line = input()
        if line.strip() == "":
            break
        lines.append(line)
    observation_input = "\n".join(lines)

    print("\n⏳ Running AoT-style structured reasoning...\n")
    result = run_structured_aot(observation_input)
    print(result)