import os
import json
import re
from typing import Dict, Any, List, Tuple
import openai

from pyspread.model import CodeArray
from pyspread.context import get_main_window_instance


class SpreadsheetAgent:
    def __init__(self, llm_model: str = "gpt-4o"):
        self.model = llm_model
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        openai.api_key = api_key

    def analyze_context(self, context_json: str) -> Dict[str, Any]:
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
                            "너는 똑똑한 스프레드시트 비서야. 사용자가 편집한 셀과 주변 셀의 내용을 바탕으로 "
                            "사용자의 의도를 정확히 파악하고, 필요한 셀에 값을 자동으로 채워줘.\n\n"
                            "만약 사용자의 의도가 명확히 보이고 값 채우기나 계산이 필요하다면 다음 JSON 형식으로 응답해:\n"
                            "{\n"
                            '  "action": "fill",\n'
                            '  "target_cells": [[행, 열], [행, 열], ...],\n'
                            '  "value": "값 또는 수식 (여러 값일 경우 콤마로 구분)",\n'
                            '  "reasoning": "왜 이런 작업을 추천하는지 간단히 설명"\n'
                            "}\n\n"
                            "단, 정말 아무 작업도 필요 없다면 다음처럼만 응답해:\n"
                            '{ "action": "none", "reasoning": "행동이 필요하지 않습니다." }\n\n'
                            "JSON 이외의 설명은 절대 포함하지 마."
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )

            content = response['choices'][0]['message']['content']
            content = self._strip_json_block(content)
            return json.loads(content)

        except json.JSONDecodeError:
            return {"error": "LLM 응답이 JSON 형식이 아닙니다.", "raw_response": content}
        except Exception as e:
            return {"error": str(e)}

    def _build_prompt(self, context: Dict[str, Any]) -> str:
        return f"""
아래는 사용자가 방금 편집한 셀과 그 주변 셀들의 값입니다.

이 정보를 바탕으로 사용자의 의도를 파악하고,
자동으로 채워야 할 셀이 있다면 JSON으로 명확히 응답해주세요.

입력:
{json.dumps(context, ensure_ascii=False, indent=2)}
"""

    def _strip_json_block(self, text: str) -> str:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        return match.group(1) if match else text.strip()


def validate_llm_output(data: Dict[str, Any]) -> Tuple[List[Tuple[int, int, int]], List[str]]:
    if data.get("action") == "none":
        print("[Agent] LLM 응답: 행동 필요 없음")
        return [], []

    if not all(k in data for k in ("target_cells", "value")):
        raise ValueError("LLM 결과에 target_cells 또는 value가 없습니다.")

    cells = []
    for cell in data["target_cells"]:
        if not (isinstance(cell, list) and len(cell) == 2):
            raise ValueError(f"셀 좌표 형식이 잘못되었습니다: {cell}")
        row, col = cell
        cells.append((row, col, 0))

    value = data["value"].strip()
    values = [v.strip() for v in value.split(",")] if "," in value else [value] * len(cells)
    if len(values) < len(cells):
        values += [''] * (len(cells) - len(values))
    elif len(values) > len(cells):
        values = values[:len(cells)]

    values = [v if v.startswith("=") or v.startswith("'") else f"'{v}'" for v in values]

    if len(values) != len(cells):
        raise ValueError("셀 개수와 값 개수가 일치하지 않습니다.")

    return cells, values


def apply_cell_changes(cells: List[Tuple[int, int, int]], values: List[str]):
    print("[Agent] 셀 적용 시작")
    if not cells:
        print("[Agent] 셀이 비어 있음. 적용 건너뜀.")
        return

    main_window = get_main_window_instance()
    if main_window is None:
        print("[LLM Error] main_window_instance가 아직 초기화되지 않았습니다.")
        return
    code_array: CodeArray = main_window.grid.model.code_array

    for (row, col, tab), value in zip(cells, values):
        print(f"[Apply] ({row},{col}) ← {value}")
        code_array[row, col, tab] = value

    main_window.grid.model.layoutChanged.emit()
    main_window.grid.viewport().update()
    print("[Agent] 셀 업데이트 완료")


def handle_agent_response(context_json: str):
    agent = SpreadsheetAgent()
    result = agent.analyze_context(context_json)

    print("[LLM OUTPUT]", json.dumps(result, ensure_ascii=False, indent=2))

    if "error" in result:
        print("[Error]", result["error"])
        return

    try:
        cells, values = validate_llm_output(result)
        apply_cell_changes(cells, values)
    except Exception as e:
        print("[Agent Exception]", e)