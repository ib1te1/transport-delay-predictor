# Предиктор изменений в графике движения наземного транспорта

Прогноз отклонения от расписания городского наземного транспорта за 10–15
минут до события и дашборд диспетчера с уровнем риска.

Система из трёх модулей: ML-ядро (`predictor`), бэкенд (`ingest`,
`matcher`, `api`) и дашборд (`web`). Устройство — в
`docs/specs/system-design.md`, условия задачи — в `docs/case-brief.md`.

## Запуск

1. Распаковать раздачу организаторов как есть в `data/dataset/`, чтобы
   получилось `data/dataset/train/`, `data/dataset/test/`,
   `data/dataset/validate/` и т. д. Без датасета система тоже поднимется,
   но без расписания и прогнозов.
2. При необходимости: `cp .env.example .env` и поправить порты.
3. Поднять:

   ```bash
   docker compose up --build
   ```

Порядок старта: Postgres и Redis → `migrate` (миграции) → `seed`
(расписание периода из `config/system.yaml`) → сервисы.

| Что | Адрес |
| --- | --- |
| Дашборд | <http://localhost:5173> |
| API и Swagger | <http://localhost:8000/docs> |
| Swagger ML-ядра | <http://localhost:8003/docs> |
| Время ответа ML-ядра | <http://localhost:8003/metrics> |
| Метрики цикла прогноза, живой MAE, алерты | <http://localhost:8000/metrics> |

## Настройки

`config/system.yaml` — по секции на сервис; сервис читает только свою.
Переменные окружения (порты, адреса) — в `.env.example`.

## Производительность

Задержка прогноза, пропускная способность, отказы и холодный старт —
`docs/performance.md`. Замер повторяется одной командой, но начинается с
`docker compose down -v` и стирает состояние демо:

```bash
python scripts/measure-performance.py
```

## Разработка

Python 3.13, Node 24. Команды ниже — для Git Bash на Windows; на
Linux/macOS вместо `py -3.13` и `.venv/Scripts/` — `python3.13` и
`.venv/bin/`.

```bash
py -3.13 -m venv .venv
.venv/Scripts/python -m pip install -r services/api/requirements.txt -r migrations/requirements.txt -r requirements-dev.txt
.venv/Scripts/python -m pip install --no-deps -e packages/contracts -e packages/common
docker compose up -d postgres redis
.venv/Scripts/alembic -c migrations/alembic.ini upgrade head
export DATABASE_URL=postgresql://delay_predictor:delay_predictor@localhost:5432/delay_predictor REDIS_URL=redis://localhost:6379/0
cd services/api && ../../.venv/Scripts/python -m pytest
```

Тесты не читают `.env`: `DATABASE_URL` и `REDIS_URL` нужно экспортировать
в шелле, как показано выше (порты — из `.env.example`), иначе тесты с
базой и Redis пропускаются. Тесты пакета или сервиса запускаются из его
папки.

### Документация по коду

Sphinx строит справочник Python API из docstring и сигнатур всех сервисов,
общих пакетов и ML-ядра. После создания виртуального окружения установите
зависимости и запустите сборку из корня репозитория:

```bash
.venv/Scripts/python -m pip install -r requirements-dev.txt -r services/api/requirements.txt -r ml/requirements.txt
.venv/Scripts/python scripts/build-code-docs.py
```

На Linux/macOS замените `.venv/Scripts/python` на `.venv/bin/python`.
Готовый справочник откройте из `data/docs/code/index.html`. Для сборки не
нужны запущенные Postgres, Redis или сервисы. Исходники документации — в
`docs/code/`; сборка использует отдельный процесс для каждого сервиса,
поскольку во всех сервисах пакет называется `app`.

Сквозные тесты всей системы идут на запущенном стеке с синтетическим
датасетом, подробности — в `tests/e2e/README.md`:

```bash
export COMPOSE_PATH_SEPARATOR=';' COMPOSE_FILE='docker-compose.yml;tests/e2e/docker-compose.e2e.yml'
docker compose up --build --wait --wait-timeout 300
.venv/Scripts/python -m pip install -r requirements-dev.txt
.venv/Scripts/python -m pytest tests/e2e
docker compose --profile '*' down -v
```

На Linux/macOS первая строка — без `COMPOSE_PATH_SEPARATOR` и с `:`
между файлами.

После смены периода в `config/system.yaml` (секция `seed`) справочники
перезагружаются так:

```bash
docker compose run --rm seed
```

## Подача телеметрии

Обычный `docker compose up --build` поднимает сервис `ingest`, но не
запускает проигрывание CSV. При `ingest.mode: replay` после готовности
сервисов отдельно выполните:

```bash
docker compose --profile replay run --rm replay
```

Проигрыватель читает `data/dataset/<replay.period>/traffic.csv`, сортирует
точки по времени и сохраняет их в Postgres перед публикацией в Redis Stream
`telemetry`. `replay.period` должен совпадать с `seed.period`. `speedup`,
включительные границы `start_at` и `end_at` задаются в `config/system.yaml`.
Прерванный прогон возобновляется той же командой после отправки накопившихся
записей. Завершённый прогон команда не повторяет: она ничего не публикует,
пишет в лог подсказку и завершается с кодом 3. Для нового демо-прогона
выполните `scripts/reset-demo.sh` (в PowerShell — `scripts/reset-demo.ps1`):
скрипт очищает состояние всех сервисов и печатает команду запуска
проигрывателя.

Проверить сохранённые строки и очередь отправки можно так:

```bash
docker compose exec postgres psql -U delay_predictor -d delay_predictor -c "SELECT count(*) AS total, count(*) FILTER (WHERE published_at IS NULL) AS pending FROM telemetry"
docker compose exec redis redis-cli XLEN telemetry
```

Для проверки TCP/NDTP переключите `ingest.mode` в `emulator` и задайте
`ingest.ndtp.dataset_anchor` временем выбранного периода в UTC, например
`2026-01-06T12:30:00Z` для тестовой раздачи. После смены режима перезапустите
стек. Образ эмулятора поставляется в раздаче и загружается отдельно:

```bash
docker compose down -v
docker load -i data/dataset/ndtp-telemetry-emulator.tar
docker compose --profile emulator up --build -d
curl http://localhost:18080/api/cells
```

Эмулятор подключается к `ingest:9201` внутри сети Compose. Его конфиг
хранится в памяти: после каждого запуска отправьте его снова. Например,
для одного терминала из тестового периода:

```bash
curl -X POST http://localhost:18080/api/config \
  -H 'Content-Type: application/json' \
  -d '{"targetHost":"ingest","targetPort":9201,"units":[{"unitId":664030,"intervalMs":5000,"autoGenerate":true,"cells":[]}]}'
```

Остановить передачу можно конфигом с `"units":[]` или командой
`docker compose --profile emulator down`. При смене периода или режима
используйте `docker compose down -v` перед новым запуском: это удаляет
локальный том Postgres и временный Redis Stream. Исходный датасет не
удаляется.
