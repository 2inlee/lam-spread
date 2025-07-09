from statistics import mean
import json
import os

log_path = os.path.join(os.path.dirname(__file__), '..', 'logs', '1_evidence_logs_plam.jsonl')

def analyze_logs(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        log_lines = f.readlines()

    total = len(log_lines)
    correct = 0
    token_counts = []
    solved_steps = []
    failed = 0

    for line in log_lines:
        try:
            log = json.loads(line)

            if log.get("is_correct") is True:
                correct += 1
                if isinstance(log.get("solved_at_trial"), int):
                    solved_steps.append(log["solved_at_trial"])
            else:
                failed += 1

            tt = log.get("total_tokens")
            if isinstance(tt, (int, float)):
                token_counts.append(tt)

        except json.JSONDecodeError as e:
            print(f"Parsing error: {e}")
            continue

    avg_tokens = round(mean(token_counts), 2) if token_counts else 0
    avg_steps = round(mean(solved_steps), 2) if solved_steps else "N/A"

    return {
        "총 질문 수": total,
        "정답 수": correct,
        "오답 수": failed,
        "정확도 (%)": round(correct / total * 100, 2) if total else 0,
        "평균 토큰 수": avg_tokens,
        "평균 성공 스텝": avg_steps
    }

if __name__ == "__main__":
    stats = analyze_logs(log_path)
    for k, v in stats.items():
        print(f"{k}: {v}")