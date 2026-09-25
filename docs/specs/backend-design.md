# Бэкенд: оркестрация, API, инфраструктура

Спека трека «Бэкенд». Общие границы и контракты — в
`docs/specs/system-design.md`; здесь они не пересказываются, а
используются.

## 1. Зона ответственности

- сервис `api`: оркестрация прогнозов, риск, алерты, REST и WebSocket для
  дашборда, метрики, Swagger;
- `seed`: загрузка справочников;
- инфраструктура: compose, миграции, `packages/common`, CI;
- заглушка `predictor`, пока ML не заменил её моделью;
- сборка документации, README и инструкция для жюри,
  `docs/performance.md`.

## 2. Состояние `api`

`api` читает две шины и держит в памяти:

- из `telemetry` — последнюю запись каждого ТС и **часы датасета**
  (максимальный `event_time`);
- из `stop_events` — последний `StopEvent` каждого ТС, то есть актуальный
  `cur_dev_s`.

Всё, что нужно после перезапуска, восстанавливается из Postgres: последние
позиции — из `telemetry`, последние прибытия — из `stop_events`, открытые
алерты — из `alerts`. Состояние в памяти — кэш, а не источник правды.

## 3. Цикл прогноза

Раз в `api.scoring_period_sec` **по часам датасета** для каждого
активного ТС (есть `tr_id`, последняя запись не старше
`api.drop_after_sec`):

1. `T` — текущее время датасета.
2. **Целевая остановка** — первая плановая остановка этого ТС с
   `time_plan` в `(T + 10 мин, T + 15 мин]`: левая граница не входит,
   правая входит. Если такой нет, прогноза в этот тик нет.
3. **Сборка `PredictRequest`:**
   - телеметрия ТС за `[T − api.request.telemetry_window_sec, T]`;
   - остановки ТС с `time_plan` в
     `[T − api.request.schedule_back_sec, target_time_begin]`, факт — только
     из `stop_events` с `time_fact ≤ T`;
   - `cur_dev_s` — `delay_s` последнего `StopEvent` с `time_fact ≤ T`,
     иначе `None`;
   - `sample_id = f"{tr_id}_{int(T.timestamp())}"`.
4. Все запросы тика — **одной пачкой** в `POST /predict` с таймаутом
   `api.predict_timeout_ms`.
5. Ответ пишется в `predictions`, по нему пересчитываются риск и алерты,
   изменения публикуются в шину `predictions`.

**Если `predictor` не ответил** (ошибка, таймаут, неверный ответ) —
для каждого запроса пачки `prediction_s = cur_dev_s` (или `0`, если
`cur_dev_s` нет), `reasons = [unknown]`, `model_version = "fallback"`,
`degraded = true`, `degraded_reason = "predictor_unavailable"`. Следующий
тик снова пробует `predictor`.

**ТС устарело** (запись старше `api.stale_after_sec`, но не старше
`api.drop_after_sec`) — прогноз строится как обычно по последнему
состоянию, с `degraded = true`, `degraded_reason = "stale_telemetry"` и
причиной `stale_telemetry` в `reasons`. Старше `api.drop_after_sec` — ТС
выводится из цикла и на карте показывается серым.

## 4. Проверка прогнозов по факту

Когда приходит `StopEvent`, всем записям `predictions` с этим
`target_stop_id` проставляются `actual_delay_s = delay_s` и
`abs_error_s = |prediction_s − delay_s|`. Живой MAE — среднее
`abs_error_s` по проверенным прогнозам; отдаётся в `/metrics` и в снимке
для дашборда.

## 5. Риск

Уровень риска ТС — по `prediction_s` текущего прогноза. Пороги — в
конфиге (`api.risk`), по умолчанию из классов организаторов:

| Уровень | Условие |
| --- | --- |
| `green` | `−60 ≤ prediction_s ≤ 120` |
| `yellow` | `120 < prediction_s ≤ 300` или `prediction_s < −60` |
| `red` | `prediction_s > 300` |

Опережение — жёлтое: ранний уход ломает интервал так же, как опоздание.
`p_late` показывается в карточке, но на цвет не влияет — у ТС один
источник цвета.

**Риск маршрута** — худший уровень среди ТС, чья текущая целевая
остановка лежит на этом маршруте. Маршруты — из `route_shapes`; пока
таблица пуста, `/api/routes` отдаёт пустой список.

## 6. Алерты

- **Открытие.** Прогноз по паре (ТС, целевая остановка) стал `red`, а
  открытого алерта по этой паре нет. Открытый алерт на пару — не больше
  одного; гарантирует уникальный частичный индекс.
- **Обновление.** Новые прогнозы по той же паре обновляют
  `predicted_delay_s` и `reasons`; `opened_at` не меняется.
- **Подтверждение.** Пришёл `StopEvent` по целевой остановке:
  `status = confirmed`, записываются `actual_delay_s` и
  `lead_time_s = time_fact − opened_at`.
- **Отмена.** Прогноз по паре опустился ниже `red`: `status = cancelled`.
  Если по той же паре снова `red` — открывается новый алерт.

Участок алерта — от последней пройденной остановки ТС
(`segment_from_stop_id`) до целевой.

Алерт не бывает задним числом по построению: `opened_at = T`, а плановое
прибытие не раньше `T + 10 мин`. Среднее `lead_time_s` по подтверждённым
алертам — в `/metrics` и на дашборде.

## 7. API для дашборда

Схемы — в `services/api/app/schemas.py`; фронт генерирует TS-типы из
`openapi.json`.

| Эндпоинт | Ответ |
| --- | --- |
| `GET /health` | живость |
| `GET /api/state` | снимок: часы датасета; все ТС — позиция, курс, скорость, `risk_level`, текущий прогноз (целевая остановка, адрес, план, `prediction_s`, `p_late`, `reasons`), флаги `stale` / `degraded`; сводка по уровням; открытые алерты; живой MAE |
| `GET /api/vehicles/{tr_id}` | карточка: текущий прогноз, последние прогнозы, трек за `api.card_track_sec`, остановки ТС вокруг текущего момента с планом и фактом |
| `GET /api/alerts?status=` | алерты по статусу, новые сверху |
| `GET /api/routes` | маршруты с линией и уровнем риска |
| `GET /api/stops` | остановки периода для карты |
| `GET /metrics` | см. §9 |

**WebSocket `/ws`** — только изменения, после того как клиент взял снимок:

| `type` | Когда | Содержимое |
| --- | --- | --- |
| `clock` | раз в секунду реального времени | время датасета |
| `vehicles` | пачкой, не чаще раза в секунду | позиции и флаги изменившихся ТС |
| `prediction` | новый прогноз | прогноз и `risk_level` ТС |
| `alert` | открытие, обновление, закрытие | алерт целиком |

## 8. Таблицы

```
predictions
  sample_id          text PK
  tr_id              bigint
  t                  timestamptz
  target_stop_id     bigint
  target_time_begin  timestamptz
  cur_dev_s          double precision NULL
  prediction_s       double precision
  p_late             double precision NULL
  reasons            jsonb
  risk_level         text          -- green | yellow | red
  degraded           boolean
  degraded_reason    text NULL
  model_version      text
  actual_delay_s     double precision NULL
  abs_error_s        double precision NULL
  index (target_stop_id), index (tr_id, t desc)

alerts
  id                   bigserial PK
  tr_id                bigint
  target_stop_id       bigint
  segment_from_stop_id bigint NULL
  status               text        -- open | confirmed | cancelled
  opened_at            timestamptz
  closed_at            timestamptz NULL
  predicted_delay_s    double precision
  reasons              jsonb
  actual_delay_s       double precision NULL
  lead_time_s          double precision NULL
  unique (tr_id, target_stop_id) where status = 'open'

vehicles      tr_id bigint PK, unit_id bigint unique
stops_plan    stop_id bigint PK, tr_id bigint, time_plan timestamptz,
              lat double precision, lon double precision, address text
              index (tr_id, time_plan)
```

`telemetry`, `stop_events`, `route_shapes` — в спеках своих треков.

## 9. Метрики

`GET /metrics` отдаёт JSON:

- `predict_latency_ms` — p50 / p95 / max вызова `/predict` за последние
  5 минут реального времени;
- `scoring_tick_ms` — длительность тика целиком;
- `stream_lag_s` — отставание обработки: реальное время минус время
  получения последней записи из шины;
- `vehicles_active`, `predictions_per_min`;
- `degraded_share` — доля прогнозов с `degraded`;
- `alerts` — открыто / подтверждено / отменено, среднее `lead_time_s`;
- `live_mae_s` и число проверенных прогнозов.

## 10. `seed`

Команда в образе `api` (`python -m app.seed`), в compose — одноразовый
сервис после `migrate`. Читает из `/data` файлы периода `seed.period`
(`test` по умолчанию — у него есть разметка для сравнения):

- расписание → `stops_plan`, **без `time_fact_begin`**;
- `traffic.csv` → пары `tr_id`, `unit_id` → `vehicles`.

Время переводится из наивного в UTC. Повторный запуск идемпотентен:
upsert по ключам.

## 11. Конфиг

```yaml
api:
  scoring_period_sec: 60
  stale_after_sec: 120
  drop_after_sec: 900
  predict_timeout_ms: 1000
  card_track_sec: 1800
  request:
    telemetry_window_sec: 1800
    schedule_back_sec: 3600
  risk:
    green: [-60, 120]
    red_above: 300

seed:
  period: test
  source_timezone: UTC
```

## 12. Тесты

Каждое правило, которое легко сломать незаметно, — отдельный тест:

- целевая остановка: ровно `T + 10 мин` не выбирается, ровно
  `T + 15 мин` — выбирается; нет остановки в окне — нет запроса;
- анти-утечка: в `PredictRequest` нет телеметрии после `T` и нет факта
  остановок после `T`;
- отказ и таймаут `predictor` → fallback с `degraded`;
- устаревшее ТС → `stale_telemetry`; очень старое — выводится из цикла;
- пороги риска, включая опережение;
- жизнь алерта: одно открытие на пару, подтверждение с `lead_time_s`,
  отмена, повторное открытие;
- проверка по факту: `abs_error_s` проставляется всем прогнозам на
  остановку;
- `seed` не загружает факт;
- снимок `/api/state` и доставка `prediction` по WS.

`predictor` в тестах подменяется моком HTTP (`respx`). Тесты с базой и
Redis — на service containers в CI, как уже устроено.

## 13. Порядок работ

1. Скелет: контракты, `system.yaml`, `seed`, таблицы бэкенда, заглушка
   `predictor`, compose с монтированием `/data`, CI.
2. Цикл прогноза с fallback и запись `predictions`.
3. `/api/state`, `/api/vehicles/{tr_id}`, WS — фронт с этого момента
   работает на живых данных.
4. Риск, алерты, проверка по факту, `/metrics`.
5. `/api/routes`, когда `matcher` даст `route_shapes`.
6. README, инструкция для жюри, Sphinx, замер в `docs/performance.md`.
