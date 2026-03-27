# Document Recognition (LangGraph + Groq)

`two_models_in_rec_doc.py` — open-source пример пайплайна распознавания документов (чеки, акты, ТН/ТТН, складские отчёты/ведомости) с **самопроверкой** и **самоисправлением**.

## Быстрый старт

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
export GROQ_API_KEY="your_key"
.venv/bin/python two_models_in_rec_doc.py
```

## Что внутри

### Архитектура LangGraph

Граф устроен как цикл:

`START → ocr_node → json_node → validate_node → (retry → json_node) / (done → END)`

- **`ocr_node`** (Vision OCR): модель `meta-llama/llama-4-scout-17b-16e-instruct` читает изображение и возвращает текст.  
  Если видит таблицу — просится сохранять структуру в **Markdown-таблице** без пропусков строк/колонок.
- **`json_node`** (JSON builder): модель `openai/gpt-oss-120b` строит **строгий JSON** по Pydantic‑схеме.
  - Сначала пытается включить JSON Mode: `response_format={"type":"json_object"}`
  - Если недоступно — fallback на streaming + извлечение JSON
- **`validate_node`** (Quality gate): кодом проверяется:
  - JSON парсится
  - проходит Pydantic‑валидацию по `DocumentSchema`
  - items не “обрезаны” (сравнение `len(items)` с количеством строк в OCR markdown‑таблице)
  - сумма `amount_with_vat` по items согласуется с `total_amount` (с допуском)

Если валидация не прошла, ошибка записывается в `validation_error` и отправляется обратно в `json_node` для исправления.  
Количество попыток ограничено `MAX_JSON_FIX_ATTEMPTS`.

### Препроцессинг изображения (Pillow)

Перед отправкой в модели изображение:
- авто‑поворачивается по EXIF (`ImageOps.exif_transpose`)
- уменьшается до **2048px** по длинной стороне
- сохраняется как JPEG `quality=85` (обычно <2–3MB)

### Схема данных (Pydantic)

Корневой объект — `DocumentSchema`:
- общие поля документа: `document_type`, `document_number`, `date`, `supplier`, `buyer`, итоги
- список позиций `items: List[InvoiceItem]`
- поля для отчётов/ведомостей: `purchase_form`, остатки, `periods: List[PeriodRecord]`

### Починка “человеческих” чисел

В модели добавлены `@field_validator(mode="before")`, которые превращают строки вида:
- `"1 250,50"`, `"10 000,00"`, `"1,250.50"`

в нормальные `float`, чтобы Pydantic не падал на форматах OCR.

## Запуск

```bash
.venv/bin/python two_models_in_rec_doc.py
```

или с явным путём:

```bash
.venv/bin/python two_models_in_rec_doc.py /path/to/photo.jpg
```

Скрипт сохранит результат рядом с изображением в файл `<имя_фото>.json`.

## Конфигурация

Обязательно задайте ключ:

- через переменную окружения:

```bash
export GROQ_API_KEY="your_key"
```

## Где менять поведение

- **OCR**: промпт внутри `ocr_node`
- **JSON**: промпт внутри `json_node`
- **Проверки качества**: `validate_node`
- **Лимиты ретраев**: `MAX_JSON_FIX_ATTEMPTS`
- **Сжатие/ресайз**: `encode_image_to_base64()`

