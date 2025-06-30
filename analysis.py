import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 📁 기본 설정
MODES = ["zero_shot", "cot", "aot"]
LOG_DIR = "logs"
CHART_DIR = "charts"
SUMMARY_CSV = "summary.csv"

os.makedirs(CHART_DIR, exist_ok=True)

# 📊 결과 저장용 리스트
summaries = []

# 📦 각 모드별 분석 수행
for mode in MODES:
    log_path = os.path.join(LOG_DIR, f"game24_logs_{mode}.jsonl")
    if not os.path.exists(log_path):
        print(f"⚠️ {log_path} not found. Skipping...")
        continue

    # 🧾 JSONL 로드
    records = []
    with open(log_path, 'r') as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not records:
        continue

    df = pd.DataFrame(records)
    df['mode'] = mode

    # ✅ 정량 분석
    total = len(df)
    correct = df['is_correct'].sum()
    accuracy = correct / total * 100 if total > 0 else 0
    avg_input_tokens = df['input_tokens'].mean()
    avg_output_tokens = df['output_tokens'].mean()
    avg_total_tokens = df['total_tokens'].mean()
    tokens_per_correct = df[df['is_correct']]['total_tokens'].mean()
    accuracy_per_1k_tokens = correct / df['total_tokens'].sum() * 1000

    # 🧠 에러 타입 통계
    error_counts = df['error_type'].value_counts()

    # 🧾 summary row 저장
    summaries.append({
        "Mode": mode,
        "Total Samples": total,
        "Correct": correct,
        "Accuracy (%)": round(accuracy, 2),
        "Avg Input Tokens": round(avg_input_tokens, 2),
        "Avg Output Tokens": round(avg_output_tokens, 2),
        "Avg Total Tokens": round(avg_total_tokens, 2),
        "Tokens per Correct": round(tokens_per_correct, 2) if correct > 0 else None,
        "Accuracy per 1K Tokens": round(accuracy_per_1k_tokens, 2)
    })

    # 📊 에러 분포 시각화
    plt.figure(figsize=(8, 4))
    sns.countplot(x='error_type', data=df, order=error_counts.index)
    plt.title(f"[{mode}] Error Type Distribution")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{CHART_DIR}/error_type_distribution_{mode}.png")
    plt.close()

    # 📊 토큰 vs 정확도 시각화
    plt.figure(figsize=(8, 4))
    sns.boxplot(x='is_correct', y='total_tokens', data=df)
    plt.title(f"[{mode}] Total Tokens vs Correctness")
    plt.xlabel("Correct")
    plt.ylabel("Total Tokens")
    plt.tight_layout()
    plt.savefig(f"{CHART_DIR}/tokens_vs_correctness_{mode}.png")
    plt.close()

# 🧾 summary.csv 저장
summary_df = pd.DataFrame(summaries)
summary_df.to_csv(SUMMARY_CSV, index=False)

print("✅ 분석 완료!")
print(f"📊 요약 파일: {SUMMARY_CSV}")
print(f"📈 시각화 파일: {CHART_DIR}/")