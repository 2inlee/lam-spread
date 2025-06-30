import openai
import json
import logging
import os

logging.basicConfig(level=logging.INFO)


class OpenAIModel:
    def __init__(self, api_key: str):
        openai.api_key = api_key

    def generate_thoughts(self, state: str, num_thoughts: int, initial_prompt: str):
        prompt = f"""{initial_prompt}

Current Steps:
{state.strip() if state else "None"}

Suggest the next {num_thoughts} valid next step(s).
Format each step as a single-line action. No "Step X:" prefix."""
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            content = response['choices'][0]['message']['content']
            lines = content.strip().split("\n")
            thoughts = [line.strip("-• ").strip() for line in lines if line.strip()]
            return thoughts[:num_thoughts]
        except Exception as e:
            logging.error(f"Error generating thoughts: {e}")
            return []

    def evaluate_states(self, thoughts, initial_prompt: str):
        prompt = f"""{initial_prompt}

    Evaluate how promising each of the following steps is for solving the problem.

    Give a JSON response in the format:
    {{
    "14 - 8 = 6": 0.5,
    "8 / 2 = 4": 1.0
    }}

    Steps:
    {chr(10).join(thoughts)}"""
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            content = response['choices'][0]['message']['content']
            parsed = json.loads(content)

            # ⚠️ 필터: 키가 thoughts 안에 없는 경우 제거
            filtered = {k.strip(): float(v) for k, v in parsed.items() if k.strip() in thoughts}
            missing = [t for t in thoughts if t not in filtered]
            if missing:
                print("⚠️ GPT 응답에서 평가 누락된 thoughts:", missing)

            return filtered
        except Exception as e:
            logging.warning(f"Evaluation failed: {e}")
            return {t: 0.0 for t in thoughts}

    def generate_solution(self, initial_prompt: str, best_state: str):
        prompt = f"""{initial_prompt}

Here is a proposed solution:
{best_state.strip()}

Does this successfully solve the problem? Explain briefly."""
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            return response['choices'][0]['message']['content']
        except Exception as e:
            logging.error(f"Error generating final solution: {e}")
            return None


class AoT:
    def __init__(
        self,
        num_thoughts=2,
        max_steps=5,
        value_threshold=0.6,
        pruning_threshold=0.3,
        backtracking_threshold=0.4,
        initial_prompt=None,
        openai_api_key=None,
        thought_cache=None,
    ):
        self.num_thoughts = num_thoughts
        self.max_steps = max_steps
        self.value_threshold = value_threshold
        self.pruning_threshold = pruning_threshold
        self.backtracking_threshold = backtracking_threshold
        self.initial_prompt = initial_prompt
        self.model = OpenAIModel(openai_api_key)
        self.output = []
        self.thought_cache = thought_cache or {"accepted": {}, "pruned": {}}

    def solve(self):
        self.dfs("", 1)

        if not self.output:
            print("❌ No valid thoughts found.")
            return None

        best_state, best_value = max(self.output, key=lambda x: x[1])
        print(f"\n✅ Best state found (score={best_value}):\n{best_state}")

        self.thought_cache["accepted"][best_state] = best_value
        solution = self.model.generate_solution(self.initial_prompt, best_state)

        print("\n🔍 Final Solution from GPT:\n", solution)

        with open("thought_cache.json", "w") as f:
            json.dump(self.thought_cache, f, indent=2)

        return solution or best_state

    def dfs(self, state: str, step: int):
        if step > self.max_steps:
            _, value = self.evaluate_thought(state)
            self.output.append((state, value))
            return

        if state in self.thought_cache["accepted"]:
            thoughts = [state]
        elif state in self.thought_cache["pruned"]:
            return
        else:
            thoughts = self.generate_and_filter_thoughts(state)

        for thought in thoughts:
            score = self.evaluated_thoughts.get(thought, 0.0)
            if score < self.value_threshold:
                self.thought_cache["pruned"][thought] = score
                continue

            new_state = (state + f"\nStep {step}: {thought}").strip()
            self.dfs(new_state, step + 1)

            best_val = max((v for _, v in self.output), default=0.0)
            if best_val < self.backtracking_threshold and self.output:
                self.output.pop()

    def generate_and_filter_thoughts(self, state):
        thoughts = self.model.generate_thoughts(state, self.num_thoughts, self.initial_prompt)
        self.evaluated_thoughts = self.model.evaluate_states(thoughts, self.initial_prompt)
        print("🧠 Thoughts generated:", thoughts)
        print("📊 Evaluated scores:", self.evaluated_thoughts)
        filtered = []
        for t in thoughts:
            score = self.evaluated_thoughts.get(t, 0.0)
            if score >= self.pruning_threshold:
                self.thought_cache["accepted"][t] = score
                filtered.append(t)
            else:
                self.thought_cache["pruned"][t] = score
        return filtered

    def evaluate_thought(self, state):
        value = self.model.evaluate_states([state], self.initial_prompt).get(state, 0.0)
        if value >= self.pruning_threshold:
            self.thought_cache["accepted"][state] = value
        else:
            self.thought_cache["pruned"][state] = value
        return state, value


if __name__ == "__main__":
    system_prompt = """
You are given 4 numbers. Use only basic arithmetic operations (+ - * /) to make 24.
Do not use parentheses. Intermediate results must be integers and non-negative.
Try not to repeat steps. Some examples:

(30 6) → 30 - 6 = 24 → valid  
(8 3) → 8 * 3 = 24 → valid  
(21 2) → 21 + 2 = 23 → invalid  

Here are the numbers: 14, 8, 8, 2
"""

    # 1. 환경변수로부터 API 키 가져오기
    api_key = os.getenv("OPENAI_API_KEY")

    # 2. 또는 직접 하드코딩 가능
    # api_key = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

    if not api_key:
        print("❌ OpenAI API key is missing. Set OPENAI_API_KEY env variable.")
        exit(1)

    aot = AoT(
        num_thoughts=2,
        max_steps=5,
        value_threshold=0.6,
        openai_api_key=api_key,
        initial_prompt=system_prompt,
    )

    aot.solve()