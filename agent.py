import os
import json
from typing import Dict, Any
import openai


class SpreadsheetAgent:
    def __init__(self, llm_model: str = "gpt-4o"):
        """
        GPT-4o 기반 에이전트 초기화. 환경변수에서 API 키를 불러옴.
        """
        self.model = llm_model
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        openai.api_key = api_key

    def analyze_context(self, context_json: str) -> Dict[str, Any]:
        """
        셀의 현재 상태 및 주변 문맥을 받아, LLM을 통해 작업 추론 및 자동화 제안
        """
        try:
            context = json.loads(context_json)
        except json.JSONDecodeError:
            return {"error": "Invalid JSON input."}

        prompt = self._build_prompt(context)

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "너는 지능적인 스프레드시트 도우미야. 사용자가 어떤 셀을 편집하고 있고 "
                            "그 주변에 어떤 값들이 있는지를 보고, 사용자가 하려는 작업을 추론해서 "
                            "셀의 좌표와 추천 수식/값을 JSON으로 제시해줘."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3
            )

            content = response['choices'][0]['message']['content']
            return json.loads(content)

        except json.JSONDecodeError:
            return {
                "error": "LLM 응답이 JSON 형식이 아닙니다.",
                "raw_response": content
            }
        except Exception as e:
            return {"error": str(e)}

    def _build_prompt(self, context: Dict[str, Any]) -> str:
        """
        프롬프트 생성: 현재 셀 및 주변 셀 문맥을 기반으로 LLM에게 설명
        """
        return f"""
다음은 사용자가 선택하거나 편집한 셀과 주변 셀의 값들입니다.
이 정보를 기반으로 사용자의 의도를 추론하고,
어떤 셀에 어떤 수식 또는 값을 입력해야 할지 다음 형식으로 JSON으로 답해주세요.

출력 예시:
{{
  "action": "sum_column",
  "target_cells": [[5, 2]],
  "value": "=SUM(B2:B4)",
  "reasoning": "사용자는 B열 숫자들의 합계를 구하려는 의도입니다."
}}

아래는 입력 데이터입니다:

{json.dumps(context, ensure_ascii=False, indent=2)}
"""