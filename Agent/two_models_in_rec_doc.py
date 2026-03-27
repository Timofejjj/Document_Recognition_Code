"""
Двухузловой LangGraph-агент для распознавания документов.

Граф:
  START
    │
    ▼
  ocr_node          ← llama-4-scout (vision): читает текст с изображения
    │
    ▼
  json_node         ← gpt-oss-120b (reasoning): строит валидный JSON по схеме
    │
    ▼
  END

Результат сохраняется в <имя_фото>.json рядом с изображением.
"""

import base64
import json
import os
import subprocess
import sys
import io
from typing import Any, Dict, List, Optional, TypedDict

# ──────────────────────────────────────────────
# Авто-перезапуск в .venv (если нужно)
# ──────────────────────────────────────────────
def _running_in_venv() -> bool:
    return bool(os.getenv("VIRTUAL_ENV")) or getattr(sys, "base_prefix", sys.prefix) != sys.prefix


def _maybe_reexec_into_local_venv() -> None:
    if _running_in_venv():
        return
    venv_python = os.path.join(os.path.dirname(__file__), ".venv", "bin", "python")
    if os.path.isfile(venv_python) and os.access(venv_python, os.X_OK):
        os.execv(venv_python, [venv_python, *sys.argv])


_maybe_reexec_into_local_venv()

from groq import Groq                                            # noqa: E402
from pydantic import BaseModel, Field, field_validator            # noqa: E402
from langgraph.graph import END, START, StateGraph               # noqa: E402
from PIL import Image, ImageOps                                  # noqa: E402

# ──────────────────────────────────────────────
# Конфигурация
# ──────────────────────────────────────────────
GROQ_API_KEY: str = os.getenv(
    "GROQ_API_KEY",
    "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",
)

OCR_MODEL  = "meta-llama/llama-4-scout-17b-16e-instruct"   # узел 1: vision OCR
JSON_MODEL = "openai/gpt-oss-120b"                          # узел 2: JSON-формирователь

# ──────────────────────────────────────────────
# 1. Pydantic-схемы результата
# ──────────────────────────────────────────────
def _parse_float(value: Any) -> Optional[float]:
    """
    Нормализует числа из OCR/LLM:
    - "1 250,50" / "1\u00a0250,50" -> 1250.50
    - "1,250.50" -> 1250.50
    - "1250" / 1250 -> 1250.0
    - "" / None -> None
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, bool):
        return float(int(value))
    if not isinstance(value, str):
        return None

    s = value.strip()
    if not s:
        return None

    # remove common noise
    s = s.replace("\u00a0", " ")  # NBSP
    s = s.replace(" ", "")
    s = s.replace("\u202f", "")  # narrow no-break space

    # keep digits, separators and minus only
    allowed = set("0123456789.,-")
    s = "".join(ch for ch in s if ch in allowed)
    if not s or s in {"-", ",", "."}:
        return None

    # Decide decimal separator:
    # - if both ',' and '.' exist: last occurrence is decimal; the other is thousands
    # - else: if only ',' exists -> decimal ',', else '.' or none
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "")
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        s = s.replace(",", ".")

    try:
        return float(s)
    except Exception:
        return None


class InvoiceItem(BaseModel):
    name: str = Field(description="Наименование товара или услуги")
    unit_of_measurement: Optional[str] = Field(description="Единица измерения (шт, упак, кг и т.д.)", default=None)
    quantity: float = Field(description="Количество", default=1.0)
    price: float = Field(description="Цена за единицу (без НДС)", default=0.0)
    amount_without_vat: Optional[float] = Field(description="Стоимость без НДС", default=None)
    vat_rate: Optional[str] = Field(description="Ставка НДС (например, '20%', '10%', 'Без НДС')", default=None)
    vat_amount: Optional[float] = Field(description="Сумма НДС", default=None)
    amount_with_vat: Optional[float] = Field(description="Стоимость с НДС", default=None)
    notes: Optional[str] = Field(description="Примечание", default=None)

    @field_validator("quantity", "price", mode="before")
    @classmethod
    def _coerce_required_floats(cls, v: Any) -> Any:
        parsed = _parse_float(v)
        return v if parsed is None else parsed

    @field_validator("amount_without_vat", "vat_amount", "amount_with_vat", mode="before")
    @classmethod
    def _coerce_optional_floats(cls, v: Any) -> Any:
        parsed = _parse_float(v)
        return v if parsed is None else parsed


class PeriodRecord(BaseModel):
    document_number: Optional[str] = Field(description="Номер документа", default=None)
    document_date: Optional[str] = Field(description="Дата документа в формате YYYY-MM-DD", default=None)
    quantity: Optional[float] = Field(description="Количество по данному документу", default=None)

    @field_validator("quantity", mode="before")
    @classmethod
    def _coerce_quantity(cls, v: Any) -> Any:
        parsed = _parse_float(v)
        return v if parsed is None else parsed


class DocumentSchema(BaseModel):
    document_type: str = Field(description="Тип документа (ТТН, ТН, Чек, Акт, Отчет, Неизвестно)")
    document_number: Optional[str] = Field(description="Серия и номер документа (например, 'ЯБ 11')", default=None)
    date: str = Field(description="Дата документа в формате YYYY-MM-DD", default="")
    supplier: str = Field(description="Название компании-поставщика / Грузоотправитель", default="")
    buyer: Optional[str] = Field(description="Название компании-покупателя / Грузополучатель", default=None)
    total_amount: float = Field(description="Итоговая сумма по документу (с НДС)", default=0.0)
    total_vat_amount: Optional[float] = Field(description="Всего сумма НДС по документу", default=None)
    items: List[InvoiceItem] = Field(description="Список позиций (товаров/услуг)", default=[])
    purchase_form: Optional[str] = Field(description="Форма выкупа", default=None)
    opening_balance: Optional[float] = Field(description="Остаток на начало месяца / периода", default=None)
    expense: Optional[float] = Field(description="Расход", default=None)
    closing_balance: Optional[float] = Field(description="Остаток на конец месяца / периода", default=None)
    periods: List[PeriodRecord] = Field(description="Записи за период (для складских/торговых отчётов)", default=[])
    confidence_score: float = Field(description="Уверенность ИИ (0.0–1.0)", default=0.5)

    @field_validator(
        "total_amount",
        mode="before",
    )
    @classmethod
    def _coerce_total_amount(cls, v: Any) -> Any:
        parsed = _parse_float(v)
        return v if parsed is None else parsed

    @field_validator(
        "total_vat_amount",
        "opening_balance",
        "expense",
        "closing_balance",
        mode="before",
    )
    @classmethod
    def _coerce_optional_amounts(cls, v: Any) -> Any:
        parsed = _parse_float(v)
        return v if parsed is None else parsed


JSON_SCHEMA_EXAMPLE = json.dumps(
    {
        "document_type": "ТТН | ТН | Чек | Акт | Отчет | Неизвестно",
        "document_number": "ЯБ 11",
        "date": "YYYY-MM-DD",
        "supplier": "Грузоотправитель / Поставщик",
        "buyer": "Грузополучатель / Покупатель",
        "total_amount": 0.0,
        "total_vat_amount": 0.0,
        "items": [
            {
                "name": "string",
                "unit_of_measurement": "шт",
                "quantity": 1.0,
                "price": 0.0,
                "amount_without_vat": 0.0,
                "vat_rate": "20%",
                "vat_amount": 0.0,
                "amount_with_vat": 0.0,
                "notes": None,
            }
        ],
        "purchase_form": None,
        "opening_balance": None,
        "expense": None,
        "closing_balance": None,
        "periods": [
            {"document_number": "string", "document_date": "YYYY-MM-DD", "quantity": 0.0}
        ],
        "confidence_score": 1.0,
    },
    ensure_ascii=False,
)

# ──────────────────────────────────────────────
# 2. Состояние LangGraph-графа
# ──────────────────────────────────────────────
class AgentState(TypedDict):
    image_bytes:  str                       # base64-строка изображения
    ocr_text:     Optional[str]             # сырой текст с документа (узел 1)
    raw_json_text: Optional[str]            # сырой ответ JSON-модели (до валидации)
    parsed_json:  Optional[Dict[str, Any]]  # итоговый JSON (после validate_node)
    validation_error: Optional[str]         # текст ошибки парсинга/валидации для ретрая
    attempts: int                           # количество попыток исправления JSON
    error:        Optional[str]


# ──────────────────────────────────────────────
# Вспомогательные утилиты
# ──────────────────────────────────────────────
def _stream_groq(
    model: str,
    messages: list,
    temperature: float = 1.0,
    max_completion_tokens: int = 1024,
    top_p: float = 1.0,
    **extra_kwargs: Any,
) -> str:
    """Запускает Groq-completion в режиме stream, печатает чанки и возвращает полный текст."""
    client = Groq(api_key=GROQ_API_KEY)
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        top_p=top_p,
        stream=True,
        stop=None,
        **extra_kwargs,
    )
    parts: List[str] = []
    for chunk in completion:
        content = chunk.choices[0].delta.content or ""
        if content:
            sys.stdout.write(content)
            sys.stdout.flush()
            parts.append(content)
    sys.stdout.write("\n")
    sys.stdout.flush()
    return "".join(parts).strip()


def _call_groq(
    model: str,
    messages: list,
    temperature: float = 1.0,
    max_completion_tokens: int = 1024,
    top_p: float = 1.0,
    **extra_kwargs: Any,
) -> str:
    """Не-стриминговый вызов (нужен для response_format=json_object)."""
    client = Groq(api_key=GROQ_API_KEY)
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        top_p=top_p,
        stream=False,
        stop=None,
        **extra_kwargs,
    )
    content = completion.choices[0].message.content or ""
    return content.strip()


def _extract_json(text: str) -> Dict[str, Any]:
    """Вытаскивает первый валидный JSON-объект из текста и валидирует по схеме."""
    clean = text.strip()
    if clean.startswith("```"):
        lines = clean.splitlines()
        lines = lines[1:] if lines[0].startswith("```") else lines
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        clean = "\n".join(lines).strip()

    decoder = json.JSONDecoder()
    parsed: Optional[Dict[str, Any]] = None

    try:
        parsed = json.loads(clean)
    except json.JSONDecodeError:
        pass

    if parsed is None:
        start = clean.find("{")
        if start == -1:
            raise ValueError("Ответ модели не содержит JSON-объект.")
        i, last_err = start, None
        while i != -1:
            try:
                parsed, _ = decoder.raw_decode(clean[i:])
                break
            except Exception as e:
                last_err = e
                i = clean.find("{", i + 1)
        if parsed is None:
            raise ValueError(f"Не удалось извлечь JSON: {last_err}")

    validated = DocumentSchema.model_validate(parsed)
    return validated.model_dump()


# ──────────────────────────────────────────────
# 3. Узел 1 — OCR (vision-модель)
# ──────────────────────────────────────────────
def ocr_node(state: AgentState) -> AgentState:
    """
    llama-4-scout-17b-16e-instruct (vision) читает изображение
    и возвращает подробный сырой текст со всеми данными документа.
    """
    print("\n🔍 [Узел 1 / OCR] Отправка изображения в llama-4-scout...")
    try:
        ocr_text = _stream_groq(
            model=OCR_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Ты — система OCR. Прочитай текст с изображения документа "
                                "максимально подробно и дословно. Документ может быть: "
                                "ТТН (товарно-транспортная накладная), ТН (товарная накладная), "
                                "чек, акт, счёт, складской/торговый отчёт, ведомость.\n\n"
                                "Обязательно извлеки:\n"
                                "• Тип, серию и номер документа\n"
                                "• Дату\n"
                                "• Грузоотправитель / Поставщик\n"
                                "• Грузополучатель / Покупатель\n"
                                "• ВСЕ позиции таблицы: наименование, ед. измерения, "
                                "количество, цену, сумму без НДС, ставку НДС, сумму НДС, "
                                "сумму с НДС, примечания\n"
                                "• Итоговую сумму, итоговый НДС\n"
                                "• Для отчётов/ведомостей: форму выкупа, остатки на начало/конец, "
                                "расход, записи за период (номер документа, дата, количество)\n\n"
                                "Если видишь таблицу, ОБЯЗАТЕЛЬНО сохраняй её структуру, используя формат "
                                "Markdown-таблиц. Не пропускай ни одной строки и ни одной колонки. "
                                "Если колонка пустая, ставь пробел.\n"
                                "Если текст повёрнут — мысленно поверни и прочитай.\n"
                                "Верни ТОЛЬКО прочитанный текст, без пояснений."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{state['image_bytes']}"
                            },
                        },
                    ],
                }
            ],
            temperature=0.1,
            max_completion_tokens=4096,
            top_p=1,
        )
        if not ocr_text:
            raise RuntimeError("OCR-модель вернула пустой ответ.")
        print("✅ [Узел 1 / OCR] Текст получен.")
        return {**state, "ocr_text": ocr_text, "error": None}
    except Exception as e:
        print(f"❌ [Узел 1 / OCR] Ошибка: {e}")
        return {**state, "ocr_text": None, "error": str(e)}


# ──────────────────────────────────────────────
# 4. Узел 2 — JSON-формирователь (reasoning-модель)
# ──────────────────────────────────────────────
def json_node(state: AgentState) -> AgentState:
    """
    gpt-oss-120b получает сырой OCR-текст и строит
    строгий JSON по бизнес-схеме DocumentSchema.
    """
    if state.get("error") or not state.get("ocr_text"):
        return {**state, "raw_json_text": None, "parsed_json": None}

    print("\n🧠 [Узел 2 / JSON] Формирование JSON через gpt-oss-120b...")
    try:
        system_prompt = (
            "Ты — эксперт-бухгалтер и парсер документов (РБ/РФ). "
            "Тебе дадут распознанный текст документа. "
            "Твоя задача — вернуть ТОЛЬКО валидный JSON без markdown, без пояснений.\n\n"
            f"Схема (пример типов полей):\n{JSON_SCHEMA_EXAMPLE}\n\n"
            "Перед формированием JSON, проверь математику строк:\n"
            "• quantity * price = amount_without_vat\n"
            "• amount_without_vat + vat_amount = amount_with_vat\n"
            "Если распознанные цифры не сходятся, используй логику, чтобы понять, где опечатка OCR, "
            "опираясь на итоговые суммы документа.\n\n"
            "Правила заполнения:\n"
            "• document_type — одно из: ТТН, ТН, Чек, Акт, Отчет, Неизвестно\n"
            "• document_number — серия и номер (например 'ЯБ 11'), если есть\n"
            "• date — формат YYYY-MM-DD; если не найдено — пустая строка\n"
            "• supplier — грузоотправитель / поставщик\n"
            "• buyer — грузополучатель / покупатель (null если нет)\n"
            "• total_amount — итоговая сумма С НДС (число)\n"
            "• total_vat_amount — итоговая сумма НДС (null если нет)\n"
            "• items — АБСОЛЮТНО ВСЕ позиции из таблицы документа, без пропусков и сокращений.\n"
            "  ВАЖНО: Если в OCR-тексте 7 строк таблицы — в items ДОЛЖНО быть ровно 7 объектов.\n"
            "  НЕ СОКРАЩАЙ, НЕ ОБЪЕДИНЯЙ, НЕ ПРОПУСКАЙ ни одной строки.\n"
            "  Каждый объект item содержит:\n"
            "  - name, unit_of_measurement (шт/упак/кг/...)\n"
            "  - quantity, price (за единицу без НДС)\n"
            "  - amount_without_vat, vat_rate ('20%'/'10%'/'Без НДС'), vat_amount, amount_with_vat\n"
            "  - notes (примечание, null если нет)\n"
            "  Если позиций нет — пустой список []\n"
            "• purchase_form — форма выкупа (для отчётов, иначе null)\n"
            "• opening_balance — остаток на начало периода (null если нет)\n"
            "• expense — расход (null если нет)\n"
            "• closing_balance — остаток на конец периода (null если нет)\n"
            "• periods — записи за период [{document_number, document_date, quantity}]\n"
            "  Пустой список [] если документ не является отчётом/ведомостью\n"
            "• confidence_score — твоя уверенность от 0.0 до 1.0\n\n"
            "Ответ должен начинаться с '{' и заканчиваться '}'."
        )

        retry_hint = ""

        # Если в предыдущей попытке JSON был невалидный
        if state.get("validation_error"):
            retry_hint = (
                "\n\nВ предыдущей попытке JSON был невалидный. "
                "Исправь JSON по схеме. Ошибка валидации/парсинга:\n"
                f"{state['validation_error']}\n"
                "Верни только исправленный JSON."
            )

        messages = [
            
            {   "role": "system", 
                "content": system_prompt + retry_hint
            },

            {
                "role": "user",
                "content": (
                    "Вот распознанный текст документа:\n\n"
                    f"{state['ocr_text']}\n\n"
                    "Верни строго JSON по схеме."
                    f"{retry_hint}"
                ),
            },
        ]

        # Prefer: JSON Mode / Structured Outputs. If unsupported, fall back to streaming + extraction.
        try:
            raw_json_text = _call_groq(
                model=JSON_MODEL,
                messages=messages,
                temperature=0,
                max_completion_tokens=16384,
                top_p=1,
                reasoning_effort="medium",
                response_format={"type": "json_object"},
            )
        except TypeError:
            raw_json_text = _stream_groq(
                model=JSON_MODEL,
                messages=messages,
                temperature=0,
                max_completion_tokens=16384,
                top_p=1,
                reasoning_effort="medium",
            )
        print("✅ [Узел 2 / JSON] Ответ получен (перед валидацией).")
        return {
            **state,
            "raw_json_text": raw_json_text,
            "parsed_json": None,
            "validation_error": None,
            "attempts": int(state.get("attempts", 0)) + 1,
            "error": None,
        }
    except Exception as e:
        print(f"❌ [Узел 2 / JSON] Ошибка: {e}")
        return {**state, "raw_json_text": None, "parsed_json": None, "error": str(e)}


def _count_ocr_table_rows(ocr_text: str) -> int:
    """Подсчитывает количество строк данных в markdown-таблице OCR (без заголовка/разделителя)."""
    count = 0
    for line in ocr_text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        # пропускаем разделитель (|---|---|...)
        inner = stripped.strip("|").strip()
        if inner and all(ch in "-| " for ch in inner):
            continue
        # пропускаем заголовок (первая строка таблицы с текстовыми названиями колонок)
        lower = inner.lower()
        if "наименование" in lower or "ед." in lower or "кол-во" in lower:
            continue
        count += 1
    return count

#---------------------------------------------------------
# Перепроверка возвращенного JSON-моделью JSON на корректность
#---------------------------------------------------------
def validate_node(state: AgentState) -> AgentState:
    """Парсит/валидирует JSON. При ошибке записывает validation_error для ретрая json_node."""
    if state.get("error"):
        return state
    raw = state.get("raw_json_text")
    if not raw:
        return {**state, "validation_error": "Пустой ответ от JSON-модели.", "parsed_json": None}
    try:
        try:
            obj = json.loads(raw)
            parsed = DocumentSchema.model_validate(obj).model_dump()
        except Exception:
            parsed = _extract_json(raw)

        errors: List[str] = []

        # Проверка: количество items vs количество строк в OCR-таблице
        ocr_text = state.get("ocr_text") or ""
        expected_rows = _count_ocr_table_rows(ocr_text)
        actual_items = len(parsed.get("items", []))
        if expected_rows > 0 and actual_items < expected_rows:
            errors.append(
                f"В OCR-тексте {expected_rows} строк таблицы, но в items только {actual_items}. "
                f"Нужно включить ВСЕ {expected_rows} позиций без пропусков."
            )

        # Проверка: сумма amount_with_vat по позициям vs total_amount
        items = parsed.get("items", [])
        if items and parsed.get("total_amount", 0) > 0:
            items_total = sum(
                (it.get("amount_with_vat") or 0.0) for it in items
            )
            doc_total = parsed["total_amount"]
            if items_total > 0 and abs(items_total - doc_total) > 1.0:
                errors.append(
                    f"Сумма amount_with_vat по всем items = {items_total:.2f}, "
                    f"но total_amount = {doc_total:.2f}. Разница {abs(items_total - doc_total):.2f}. "
                    f"Проверь, что ВСЕ позиции включены и суммы корректны."
                )

        if errors:
            err_text = "\n".join(errors)
            print(f"⚠️ [Валидация] Найдены проблемы:\n{err_text}")
            return {**state, "parsed_json": None, "validation_error": err_text, "error": None}

        return {**state, "parsed_json": parsed, "validation_error": None, "error": None}
    except Exception as e:
        return {**state, "parsed_json": None, "validation_error": str(e), "error": None}


# ──────────────────────────────────────────────
# 5. Сборка LangGraph-графа
# ──────────────────────────────────────────────
MAX_JSON_FIX_ATTEMPTS = 3


def _route_after_validate(state: AgentState) -> str:
    if state.get("parsed_json") is not None and not state.get("validation_error"):
        return "done"
    if int(state.get("attempts", 0)) >= MAX_JSON_FIX_ATTEMPTS:
        return "done"  # выходим, чтобы не зациклиться; ошибка останется в validation_error
    return "retry"


workflow = StateGraph(AgentState)
workflow.add_node("ocr_node", ocr_node)
workflow.add_node("json_node", json_node)
workflow.add_node("validate_node", validate_node)

workflow.add_edge(START, "ocr_node")
workflow.add_edge("ocr_node", "json_node")
workflow.add_edge("json_node", "validate_node")
workflow.add_conditional_edges(
    "validate_node",
    _route_after_validate,
    {
        "retry": "json_node",
        "done": END,
    },
)

agent_app = workflow.compile()


# ──────────────────────────────────────────────
# 6. Вспомогательные функции (выбор файла, сохранение)
# ──────────────────────────────────────────────
def encode_image_to_base64(image_path: str) -> str:
    """
    Препроцессинг перед отправкой в API:
    - авто-поворот по EXIF
    - уменьшение до max 2048px по длинной стороне
    - конвертация в JPEG quality=85 (обычно <2–3MB)
    """
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img)

    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    elif img.mode == "L":
        img = img.convert("RGB")

    max_side = 2048
    w, h = img.size
    long_side = max(w, h)
    if long_side > max_side:
        scale = max_side / float(long_side)
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        img = img.resize(new_size, Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85, optimize=True)
    data = buf.getvalue()
    return base64.b64encode(data).decode("utf-8")


def pick_image_file() -> Optional[str]:
    if sys.platform == "darwin":
        try:
            script = (
                'set theFile to choose file with prompt "Выберите фотографию документа"\n'
                "POSIX path of theFile"
            )
            res = subprocess.run(
                ["osascript", "-e", script],
                check=False, capture_output=True, text=True,
            )
            if res.returncode == 0:
                return res.stdout.strip() or None
        except Exception:
            pass
    return None


def normalize_path(s: str) -> str:
    s = s.strip().strip('"').strip("'").strip()
    if s.startswith("file://"):
        s = s[len("file://"):]
    return s.replace("\\ ", " ")


def resolve_image_path() -> Optional[str]:
    if len(sys.argv) > 1:
        candidate = normalize_path(sys.argv[1])
        if os.path.isfile(candidate):
            return candidate
        print(f"⚠️ Файл из аргумента не найден: {candidate}")

    selected = pick_image_file()
    if selected and os.path.isfile(selected):
        return selected

    if sys.stdin.isatty():
        try:
            raw = input("Путь к изображению: ").strip()
        except EOFError:
            return None
        candidate = normalize_path(raw)
        if os.path.isfile(candidate):
            return candidate
        if candidate:
            print(f"⚠️ Файл не найден: {candidate}")
    return None


def build_output_json_path(image_path: str) -> str:
    base_dir = os.path.dirname(os.path.abspath(image_path)) or os.getcwd()
    stem = os.path.splitext(os.path.basename(image_path))[0] or "parsed_result"
    candidate = os.path.join(base_dir, f"{stem}.json")
    if not os.path.exists(candidate):
        return candidate
    i = 1
    while True:
        path_i = os.path.join(base_dir, f"{stem}_{i}.json")
        if not os.path.exists(path_i):
            return path_i
        i += 1


def save_json_result(image_path: str, parsed_json: Dict[str, Any]) -> str:
    output_path = build_output_json_path(image_path)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(parsed_json, f, ensure_ascii=False, indent=4)
    return output_path


# ──────────────────────────────────────────────
# 7. Точка входа
# ──────────────────────────────────────────────
if __name__ == "__main__":
    image_path = resolve_image_path()

    if not image_path:
        print("\n⚠️ ОШИБКА: Изображение не выбрано.")
        print("Запуск: python two_models_in_rec_doc.py /path/to/photo.jpg")
        raise SystemExit(1)

    base64_image = encode_image_to_base64(image_path)
    print(f"🖼 Изображение '{image_path}' загружено.")

    initial_state: AgentState = {
        "image_bytes": base64_image,
        "ocr_text":    None,
        "raw_json_text": None,
        "parsed_json": None,
        "validation_error": None,
        "attempts": 0,
        "error":       None,
    }

    print("🚀 Запуск LangGraph-агента (2 узла)...")
    final_state = agent_app.invoke(initial_state)

    if final_state.get("error"):
        print(f"\n❌ Ошибка при выполнении графа:\n{final_state['error']}")
        raise SystemExit(1)

    if final_state.get("parsed_json") is None:
        err = final_state.get("validation_error") or "Не удалось получить валидный JSON."
        print(f"\n❌ Ошибка валидации JSON:\n{err}")
        raise SystemExit(1)

    print("\n🎉 Результат (JSON):")
    print(json.dumps(final_state["parsed_json"], indent=4, ensure_ascii=False))

    output_path = save_json_result(image_path, final_state["parsed_json"])
    print(f"\n💾 JSON сохранён: {output_path}")
