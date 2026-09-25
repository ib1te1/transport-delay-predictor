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

## Настройки

`config/system.yaml` — по секции на сервис; сервис читает только свою.
Переменные окружения (порты, адреса) — в `.env.example`.

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

После смены периода в `config/system.yaml` (секция `seed`) справочники
перезагружаются так:

```bash
docker compose run --rm seed
```
