Документация по коду
====================

Справочник описывает Python-пакеты системы и связи между ними. Страницы API
создаются из docstring и сигнатур исходного кода при сборке Sphinx.

.. toctree::
   :maxdepth: 2

   architecture

Пакеты и сервисы
----------------

* `API <api/index.html>`_ — состояние транспорта, прогнозы и WebSocket дашборда.
* `Ingest <ingest/index.html>`_ — приём и публикация телеметрии.
* `Matcher <matcher/index.html>`_ — сопоставление транспорта с рейсами.
* `Predictor <predictor/index.html>`_ — HTTP-сервис прогноза задержки.
* `Contracts <contracts/index.html>`_ — общие схемы обмена между сервисами.
* `Common <common/index.html>`_ — конфигурация, Postgres и Redis.
* `ML <ml/index.html>`_ — подготовка признаков, обучение и инференс модели.

Исходники фронтенда находятся в ``web/src``. HTTP-контракты работающих
сервисов также доступны в Swagger по адресам ``/docs`` каждого сервиса.
