# Матчер

Матчер читает телеметрию из Redis Stream `telemetry`, сопоставляет её с
`stops_plan`, сохраняет состояние и пишет события в Redis Stream
`stop_events` по общему контракту `StopEvent`. Детали алгоритма и границы
решения — в [спецификации](../../docs/specs/matcher-design.md).

После `docker compose up --build` проверьте
`http://127.0.0.1:8002/ready`: ответ `ready` означает доступность БД,
Redis и обоих фоновых обработчиков. `input_failing` — не проходит
обработка телеметрии, `output_failing` — отправка событий в Redis; оба
повторяются с нарастающей паузой до 10 с, причина в
`docker compose logs matcher`.
Демо-телеметрию запускает профиль `replay` у ингеста.

`GET /quality`:

- `stream_cursor` — id последней обработанной записи потока (`0-0` —
  ещё ничего), `stream_last_id` — последняя запись в потоке (`null`, если
  поток пуст);
- `unread` — в потоке есть записи после `stream_cursor`;
- `lag_ms` — на сколько миллисекунд (по времени добавления в поток)
  обработанная запись старше последней. `null`, если сравнивать не с
  чем: ничего не обработано, поток пуст или кончается раньше
  `stream_cursor`. Последние два случая при непустом `stream_cursor`
  значат, что в базе прошлый прогон: нужен `scripts/reset-demo`;
- `planned_visits`, `stop_events`, `outbox_pending` — плановые посещения,
  найденные события и события, ещё не отправленные в Redis;
- `rejected_late_ticks` и `skipped` (`malformed`, `unknown_vehicle`,
  `unplanned`, `invalid_location`) — отброшенные точки с момента запуска.
  У точки без валидной координаты детектор учитывает только время.

Пороги детектора лежат в `config/assumptions.yaml` и сохраняются вместе с
позицией. Если их изменить после обработки телеметрии, матчер не
стартует: контейнер в состоянии `Exited`, в `docker compose logs matcher`
видно сохранённые и заданные значения. Верните прежние значения или
начните демо заново: `scripts/reset-demo.sh` (`scripts/reset-demo.ps1`).
Новый прогон тоже начинается только через этот скрипт.

Для проверки в локальном окружении с Python 3.13 установите зависимости
из `services/matcher/requirements.txt`, `migrations/requirements.txt` и
`requirements-dev.txt`, а общие пакеты — editable. Тестам нужны Postgres с
применёнными миграциями (`alembic -c migrations/alembic.ini upgrade head`)
и Redis: задайте `DATABASE_URL` и `REDIS_URL` и из `services/matcher`
выполните `pytest`. При `REQUIRE_INFRA=1` без этих переменных
инфраструктурные тесты завершатся ошибкой. Тесты используют временные
схемы Postgres и отдельные потоки Redis.
