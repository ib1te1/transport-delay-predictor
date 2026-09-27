# Сквозные тесты

Проверяют систему целиком на запущенном `docker compose`: `migrate` и
`seed`, проигрывание CSV (сценарий А), шины Redis, `matcher`,
`predictor` за `api`, REST и WebSocket дашборда, приём NDTP по TCP
(сценарий Б), отдачу `web` и сброс демо через `scripts/reset-demo`.

Данные — маленький синтетический датасет в `data/validate/`: три автобуса
на своих улицах, остановка каждые 120 с, у каждого автобуса постоянное
отклонение от плана, плюс терминал без расписания. Отсюда тесты знают
точный ответ: какие остановки пройдены и с какой задержкой. Датасет
пересобирается командой `python tests/e2e/synthetic.py`.

`docker-compose.e2e.yml` подменяет только `/data` и `config/system.yaml`
(копия основного с `replay.speedup: 240`), пороги
`config/assumptions.yaml` остаются настоящими. NDTP `ingest` принимает
только в режиме `ingest.mode: emulator`, поэтому тесты сценария Б
пересоздают его с конфигом `config/ingest-ndtp.yaml` (переменная
`E2E_INGEST_CONFIG`) уже после проигрывания и возвращают обратно в
`replay`: два источника одновременно не работают, как и на демо.

## Запуск

Команды — из корня репозитория, в Git Bash или на Linux/macOS. Нужны
Docker и Python 3.13.

```bash
export COMPOSE_FILE=docker-compose.yml:tests/e2e/docker-compose.e2e.yml
docker compose up --build --wait --wait-timeout 300
python -m pip install -r requirements-dev.txt
python -m pytest tests/e2e
docker compose --profile '*' down -v
```

В Git Bash на Windows разделитель списка файлов compose — `;`, а не `:`:

```bash
export COMPOSE_PATH_SEPARATOR=';'
export COMPOSE_FILE='docker-compose.yml;tests/e2e/docker-compose.e2e.yml'
```

Прогон занимает полторы-две минуты. Состояние от прошлого прогона тесты
сбрасывают сами через `scripts/reset-demo`, поэтому набор можно
запускать повторно на том же стеке. Последний тест тоже делает сброс и
проигрывает датасет заново: после него стек снова заполнен.

`docker compose down -v` удаляет том Postgres. Если стек поднимался и
для работы с настоящим датасетом, после тестов нужен обычный
`docker compose up --build` без `COMPOSE_FILE`: `seed` загрузит
справочники периода заново.

## Настройки

| Переменная | По умолчанию | Что задаёт |
| --- | --- | --- |
| `E2E_REQUIRED` | — | `1`: недоступный стек — ошибка, а не пропуск (так в CI) |
| `E2E_API_URL` | `http://127.0.0.1:8000` | адрес `api`; WebSocket — `E2E_WS_URL` |
| `E2E_INGEST_URL`, `E2E_MATCHER_URL`, `E2E_PREDICTOR_URL`, `E2E_WEB_URL` | порты 8001, 8002, 8003, 5173 | адреса остальных сервисов |
| `E2E_NDTP_HOST`, `E2E_NDTP_PORT` | `127.0.0.1`, `9201` | порт NDTP у `ingest` |
| `E2E_COMPOSE_FILES` | основной файл и `docker-compose.e2e.yml` | файлы compose через `os.pathsep` |
| `E2E_READY_TIMEOUT`, `E2E_SETTLE_TIMEOUT` | 240, 120 | секунды ожидания готовности и обработки проигрывания |

Без запущенного стека все тесты пропускаются с причиной. Тесты с
пометкой `xfail` описывают поведение из спек, которого ещё нет в коде
(`/metrics`, алерты, проверка прогнозов по факту); когда оно появится,
они начнут проходить и пометку надо снять.
