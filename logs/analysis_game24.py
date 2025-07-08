import json
import re
import argparse
from collections import Counter

def extract_successful_trial(text):
    """
    LLM 응답 텍스트에서 성공한 Trial 번호를 찾아 반환. 없으면 None
    """
    trial_blocks = re.findall(r'### Trial (\d+):\n(.*?)(?=\n### Trial |\Z)', text, re.DOTALL)
    for trial_num, block in trial_blocks:
        if 'Evaluation: Success' in block:
            return int(trial_num)
    return None

def safe_get(d, key, default=0):
    return d.get(key) if isinstance(d.get(key), (int, float)) else default

def analyze_jsonl_log(file_path):
    total = 0
    correct = 0
    wrong = 0
    solved_at_trial_counter = Counter()

    token_stats = {
        'correct': {'input': 0, 'output': 0, 'total': 0, 'count': 0},
        'wrong': {'input': 0, 'output': 0, 'total': 0, 'count': 0}
    }

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            total += 1
            data = json.loads(line)

            is_correct = data.get('is_correct', False)
            llm_text = data.get('llm_response', '') or data.get('output', '') or ''

            # Trial 파싱
            solved_at = extract_successful_trial(llm_text)

            # Token 값
            input_tokens = safe_get(data, 'input_tokens')
            output_tokens = safe_get(data, 'output_tokens')
            total_tokens = safe_get(data, 'total_tokens')

            category = 'correct' if is_correct else 'wrong'

            token_stats[category]['input'] += input_tokens
            token_stats[category]['output'] += output_tokens
            token_stats[category]['total'] += total_tokens
            token_stats[category]['count'] += 1

            if is_correct:
                correct += 1
                if solved_at is not None:
                    solved_at_trial_counter[solved_at] += 1
                else:
                    solved_at_trial_counter["unknown"] += 1
            else:
                wrong += 1

    # 출력
    print("총 문제 수:", total)
    print("정답 개수:", correct)
    print("오답 개수:", wrong)
    print("정확도:", round(correct / total * 100, 2), "%")

    print("\n🎯 성공한 Trial 분포:")
    for trial in sorted(solved_at_trial_counter.keys(), key=lambda x: (x if isinstance(x, int) else 999)):
        print(f" - Trial {trial}: {solved_at_trial_counter[trial]}회")

    print("\n📊 평균 토큰 사용량:")
    for key in ['correct', 'wrong']:
        count = token_stats[key]['count']
        if count == 0:
            continue
        print(f" - {key.capitalize()} ({count}개):")
        print(f"    • 평균 input_tokens  : {token_stats[key]['input'] / count:.2f}")
        print(f"    • 평균 output_tokens : {token_stats[key]['output'] / count:.2f}")
        print(f"    • 평균 total_tokens  : {token_stats[key]['total'] / count:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True, help='Path to JSONL log file')
    args = parser.parse_args()

    analyze_jsonl_log(args.file)