# Матчер

Матчер читает опубликованную телеметрию из Postgres, сопоставляет её с
`stops_plan`, сохраняет состояние и пишет события в Redis Stream
`stop_events` по общему контракту `StopEvent`. Детали алгоритма и границы
решения — в [спецификации](../../docs/specs/matcher-design.md).

После `docker compose up --build` проверьте
`http://127.0.0.1:8002/ready`: ответ `ready` означает доступность БД,
Redis и обоих фоновых обработчиков. `GET /quality` показывает позицию
обработки, число плановых посещений и событий, остаток очереди отправки.
Демо-телеметрию запускает профиль `replay` у ингеста.

Для проверки в локальном окружении с Python 3.13 установите зависимости
из `services/matcher/requirements.txt`, `migrations/requirements.txt` и
`requirements-dev.txt`, а общие пакеты — editable. Затем примените миграции
`alembic -c migrations/alembic.ini upgrade head` и из `services/matcher`
выполните `pytest`. При `REQUIRE_INFRA=1` должны быть заданы
`DATABASE_URL` и `REDIS_URL`, иначе инфраструктурные тесты завершатся
ошибкой. Тесты используют временные схемы Postgres и отдельный stream Redis.

Пороги детектора лежат в `config/assumptions.yaml`; изменение порогов после
обработки телеметрии требует нового воспроизведения данных.
