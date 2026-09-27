Как читать код
==============

Поток данных начинается в ``ingest``: сервис принимает или проигрывает
телеметрию, сохраняет её в Postgres и публикует в Redis Stream. ``matcher``
привязывает точки к рейсам и формирует текущее состояние. ``api`` собирает
данные для дашборда и запрашивает прогноз у ``predictor``. ``predictor``
использует пакет ``busdelay`` с обученной моделью или базовым прогнозом.
Общие схемы сообщений лежат в ``contracts``; доступ к конфигурации и
инфраструктуре — в ``common``.

У каждого сервиса собственный пакет ``app``. Поэтому сборщик запускает Sphinx
отдельно для каждого сервиса: ссылки на модули ``app.*`` в его справочнике
всегда указывают на нужную реализацию.

Точки входа и основные модули
-----------------------------

* `ingest.app.main <ingest/app-main.html>`_ запускает приём телеметрии;
  `ingest.app.ndtp <ingest/app-ndtp.html>`_ разбирает протокол, а
  `ingest.app.replay <ingest/app-replay.html>`_ воспроизводит CSV.
* `matcher.app.main <matcher/app-main.html>`_ запускает обработчик потока;
  `matcher.app.matching <matcher/app-matching.html>`_ обнаруживает прибытие
  на остановку, а `matcher.app.store <matcher/app-store.html>`_ сохраняет
  состояние и события.
* `api.app.main <api/app-main.html>`_ содержит HTTP- и WebSocket-точки;
  `api.app.loop <api/app-loop.html>`_ организует прогнозы, а
  `api.app.dashboard <api/app-dashboard.html>`_ готовит состояние дашборда.
* `predictor.app.main <predictor/app-main.html>`_ предоставляет ``/predict``;
  `predictor.app.predictor <predictor/app-predictor.html>`_ выбирает модель
  или базовый прогноз. `busdelay.features <ml/busdelay-features.html>`_
  вычисляет признаки, `busdelay.model <ml/busdelay-model.html>`_ обучает
  модель, `busdelay.inference <ml/busdelay-inference.html>`_ выполняет прогноз.
* `contracts <contracts/index.html>`_ задаёт структуры обмена; `common
  <common/index.html>`_ содержит доступ к конфигурации, базе и шине.

Для устройства системы и проектных решений см. ``docs/specs/system-design.md``
в репозитории. Для настройки среды и запуска см. ``README.md``.
