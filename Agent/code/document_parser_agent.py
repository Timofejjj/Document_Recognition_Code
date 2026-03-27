import base64
import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional, TypedDict

def _running_in_venv() -> bool:
    return bool(os.getenv("VIRTUAL_ENV")) or getattr(sys, "base_prefix", sys.prefix) != sys.prefix


def _maybe_reexec_into_local_venv() -> None:
    """
    Если пользователь запускает системным Python (например, /usr/local/bin/python3.12),
    пробуем автоматически перезапустить скрипт в локальном .venv, чтобы зависимости
    (requests/pydantic/langgraph) гарантированно были доступны.
    """
    if _running_in_venv():
        return
    venv_python = os.path.join(os.path.dirname(__file__), ".venv", "bin", "python")
    if os.path.isfile(venv_python) and os.access(venv_python, os.X_OK):
        os.execv(venv_python, [venv_python, *sys.argv])


_maybe_reexec_into_local_venv()

try:
    from groq import Groq
except ModuleNotFoundError:
    print("❌ Не найден модуль 'groq'.")
    print("Установите зависимости в .venv:")
    print("  .venv/bin/pip install groq")
    raise
from pydantic import BaseModel, Field
try:
    from langchain_core.output_parsers import JsonOutputParser  # type: ignore[reportMissingImports]
except Exception:
    JsonOutputParser = None

try:
    from langgraph.graph import END, START, StateGraph  # type: ignore[reportMissingImports]
except Exception:
    END = START = StateGraph = None

try:
    import tkinter as tk
    from tkinter import filedialog
except Exception:
    tk = None
    filedialog = None

# ==========================================
# КОНФИГУРАЦИЯ GROQ
# ==========================================
def _load_api_key_from_dotenv(var_name: str) -> Optional[str]:
    dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.isfile(dotenv_path):
        return None
    try:
        with open(dotenv_path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() != var_name:
                    continue
                val = v.strip().strip('"').strip("'").strip()
                return val or None
    except Exception:
        return None
    return None


def _get_groq_api_key() -> Optional[str]:
    key = os.getenv(
        "GROQ_API_KEY",
        "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",
    )
    if key:
        return key.strip()
    key = _load_api_key_from_dotenv("GROQ_API_KEY")
    if key:
        return key.strip()
    return None


GROQ_API_KEY = _get_groq_api_key()
MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"


# ==========================================
# 1. ОПРЕДЕЛЕНИЕ СХЕМЫ ДАННЫХ (Pydantic)
# ==========================================
class InvoiceItem(BaseModel):
    name: str = Field(description="Наименование товара или услуги")
    quantity: float = Field(description="Количество", default=1.0)
    price: float = Field(description="Цена за единицу (без НДС)", default=0.0)


class DocumentSchema(BaseModel):
    document_type: str = Field(description="Тип документа (Накладная, Чек, Акт, Неизвестно)")
    date: str = Field(description="Дата документа в формате YYYY-MM-DD", default="")
    supplier: str = Field(description="Название компании-поставщика (кто выдал документ)", default="")
    total_amount: float = Field(description="Итоговая сумма по документу", default=0.0)
    items: List[InvoiceItem] = Field(description="Список позиций (товаров) в документе", default=[])
    confidence_score: float = Field(description="Уверенность ИИ в распознавании (от 0.0 до 1.0)", default=0.5)


# ==========================================
# 2. ОПРЕДЕЛЕНИЕ СОСТОЯНИЯ ГРАФА (State)
# ==========================================
class AgentState(TypedDict):
    image_bytes: str
    parsed_json: Optional[Dict[str, Any]]
    error: Optional[str]


def build_format_instructions() -> str:
    if JsonOutputParser is not None:
        parser = JsonOutputParser(pydantic_object=DocumentSchema)
        return parser.get_format_instructions()

    # Fallback-инструкции, если langchain_core не установлен.
    return (
        'Верни строго JSON без markdown и комментариев со следующей структурой: '
        '{"document_type":"", "date":"YYYY-MM-DD", "supplier":"", '
        '"total_amount":0.0, "items":[{"name":"", "quantity":1.0, "price":0.0}], '
        '"confidence_score":1.0}'
    )


def parse_ai_json(ai_message: str) -> Dict[str, Any]:
    if JsonOutputParser is not None:
        parser = JsonOutputParser(pydantic_object=DocumentSchema)
        parsed_data = parser.parse(ai_message)
        validated = DocumentSchema.model_validate(parsed_data)
        return validated.model_dump()

    text = ai_message.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    decoder = json.JSONDecoder()
    parsed_data = None

    # 1) Пытаемся распарсить строку целиком.
    try:
        parsed_data = json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2) Если в ответе есть лишний текст, вытаскиваем первый JSON-объект.
    if parsed_data is None:
        obj_start = text.find("{")
        if obj_start == -1:
            raise ValueError("Ответ модели не содержит JSON-объект.")
        parsed_data, _ = decoder.raw_decode(text[obj_start:])

    validated = DocumentSchema.model_validate(parsed_data)
    return validated.model_dump()


# ==========================================
# 3. ЛОГИКА УЗЛА ПАРСИНГА (Node)
# ==========================================
def parse_document_node(state: AgentState):
    print("⏳ Узел парсинга запущен. Отправка запроса в Groq...")

    format_instructions = build_format_instructions()

    system_prompt = (
        "Ты — эксперт-бухгалтер и система OCR. Твоя задача — извлечь данные из предоставленного "
        "изображения документа (накладная, чек, счет).\n"
        "ВАЖНО: Ты должен вернуть ТОЛЬКО валидный JSON.\n"
        f"{format_instructions}"
    )

    try:
        if not GROQ_API_KEY:
            raise RuntimeError("Не задан GROQ_API_KEY (переменная окружения).")

        client = Groq(api_key=GROQ_API_KEY)

        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Распознай этот документ и верни данные строго в формате JSON.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{state['image_bytes']}"},
                        },
                    ],
                },
            ],
            temperature=0,
            top_p=1,
            max_completion_tokens=1024,
            stream=True,
        )

        full_text_parts: List[str] = []
        for chunk in completion:
            content = chunk.choices[0].delta.content or ""
            if content:
                sys.stdout.write(content)
                sys.stdout.flush()
                full_text_parts.append(content)

        ai_message = "".join(full_text_parts).strip()
        if not ai_message:
            raise RuntimeError("Groq stream завершился без content.")

        parsed_data = parse_ai_json(ai_message)

        print("✅ Парсинг успешно завершен.")
        return {"parsed_json": parsed_data, "error": None}

    except Exception as e:
        print(f"❌ Ошибка в узле парсинга: {e}")
        return {"error": str(e), "parsed_json": None}


# ==========================================
# 4. СБОРКА ГРАФА
# ==========================================
agent_app = None
if StateGraph is not None:
    workflow = StateGraph(AgentState)
    workflow.add_node("parser_node", parse_document_node)
    workflow.add_edge(START, "parser_node")
    workflow.add_edge("parser_node", END)
    agent_app = workflow.compile()


# ==========================================
# 5. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==========================================
def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def pick_image_file() -> Optional[str]:
    # macOS: если Tkinter недоступен (часто у Homebrew Python), используем нативный диалог
    if (tk is None or filedialog is None) and sys.platform == "darwin":
        try:
            script = (
                'set theFile to choose file with prompt "Выберите фотографию документа"\n'
                "POSIX path of theFile"
            )
            res = subprocess.run(
                ["osascript", "-e", script],
                check=False,
                capture_output=True,
                text=True,
            )
            if res.returncode != 0:
                return None  # отмена или ошибка диалога
            path = res.stdout.strip()
            return path or None
        except Exception:
            return None

    if tk is None or filedialog is None:
        return None

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    file_path = filedialog.askopenfilename(
        title="Выберите фотографию документа",
        filetypes=[
            ("Изображения", "*.jpg *.jpeg *.png *.webp *.bmp"),
            ("Все файлы", "*.*"),
        ],
    )

    root.destroy()
    return file_path or None


def prompt_image_path_stdin() -> Optional[str]:
    """Если нет Tk (часто у Homebrew Python), запрашиваем путь в терминале."""
    if not sys.stdin.isatty():
        return None
    try:
        line = input(
            "Путь к изображению (перетащите файл в окно терминала или вставьте путь): "
        ).strip()
    except EOFError:
        return None

    def normalize_path(s: str) -> str:
        s = s.strip().strip('"').strip("'").strip()
        if s.startswith("file://"):
            s = s[len("file://") :]
        s = s.replace("\\ ", " ")
        return s

    candidates: List[str] = []
    if line:
        candidates.append(line)
        if "''" in line:
            candidates.extend([p for p in line.split("''") if p.strip()])
        if "'" in line:
            candidates.extend([p for p in line.split("'") if p.strip()])

    # Попытка вытащить путь к файлу по расширению, если терминал «склеил» ввод
    lower = line.lower()
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
        idx = lower.find(ext)
        if idx != -1:
            candidates.append(line[: idx + len(ext)])

    for raw in candidates:
        candidate = normalize_path(raw)
        if candidate and os.path.isfile(candidate):
            return candidate

    cleaned = normalize_path(line)
    if cleaned:
        print(f"⚠️ Файл не найден: {cleaned}")
    return None


def resolve_image_path() -> Optional[str]:
    # 1) Путь из аргумента командной строки: python script.py path/to/image.jpg
    if len(sys.argv) > 1:
        candidate = sys.argv[1].strip('"').strip("'")
        if os.path.isfile(candidate):
            return candidate
        print(f"⚠️ Файл из аргумента не найден: {candidate}")

    # 2) Выбор через окно проводника
    selected = pick_image_file()
    if selected and os.path.isfile(selected):
        return selected

    if tk is None or filedialog is None:
        print(
            "ℹ️ Tkinter недоступен (нет оконного выбора файла). "
            "Установите python-tk для Python 3.12 или укажите путь ниже."
        )

    # 3) Интерактивный ввод пути (без Tk)
    return prompt_image_path_stdin()


def build_output_json_path(image_path: str) -> str:
    base_dir = os.path.dirname(os.path.abspath(image_path)) or os.getcwd()
    stem = os.path.splitext(os.path.basename(image_path))[0] or "parsed_result"
    candidate = os.path.join(base_dir, f"{stem}.json")
    if not os.path.exists(candidate):
        return candidate
    i = 1
    while True:
        candidate_i = os.path.join(base_dir, f"{stem}_{i}.json")
        if not os.path.exists(candidate_i):
            return candidate_i
        i += 1


def save_json_result(image_path: str, parsed_json: Dict[str, Any]) -> str:
    output_path = build_output_json_path(image_path)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(parsed_json, f, ensure_ascii=False, indent=4)
    return output_path


if __name__ == "__main__":
    image_path = resolve_image_path()

    if not image_path:
        print("\n⚠️ ОШИБКА: Изображение не выбрано.")
        print("Запуск возможен так:")
        print("1) python document_parser_agent.py C:\\path\\to\\photo.jpg")
        print("2) или python document_parser_agent.py и выбрать файл в окне (если есть Tk).")
        print("3) или ввести путь к файлу, когда скрипт попросит (без Tk).")
        raise SystemExit(1)

    try:
        base64_image = encode_image_to_base64(image_path)
        print(f"🖼 Изображение '{image_path}' успешно загружено.")

        initial_state: AgentState = {
            "image_bytes": base64_image,
            "parsed_json": None,
            "error": None,
        }

        if agent_app is not None:
            print("🚀 Запуск ИИ-Агента (LangGraph)...")
            final_state = agent_app.invoke(initial_state)
        else:
            print("🚀 Запуск ИИ-Агента (без LangGraph)...")
            final_state = parse_document_node(initial_state)

        if final_state.get("error"):
            print("\n❌ Произошла ошибка при выполнении графа:")
            print(final_state["error"])
        else:
            print("\n🎉 Успешный результат парсинга (JSON):")
            print(json.dumps(final_state["parsed_json"], indent=4, ensure_ascii=False))
            output_path = save_json_result(image_path, final_state["parsed_json"])
            print(f"\n💾 JSON сохранён в файл: {output_path}")

    except FileNotFoundError:
        print(f"\n⚠️ ОШИБКА: Файл не найден: {image_path}")
