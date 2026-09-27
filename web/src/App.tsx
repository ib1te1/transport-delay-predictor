import { useLayoutEffect, useMemo, useState } from 'react';
import { Details } from './components/Details';
import { FleetMap } from './components/FleetMap';
import { Icon } from './components/Icon';
import { MapBoundary } from './components/MapBoundary';
import { date, delay, freshnessLabels, riskLabels, time } from './format';
import { useDashboard } from './useDashboard';
import type { Risk, Vehicle } from './types';
import './styles.css';

type Filter = 'all' | Risk | 'none';
const filters: { id: Filter; label: string; short: string }[] = [
  { id: 'all', label: 'Весь парк', short: 'Все' },
  { id: 'red', label: 'Высокий риск', short: 'Риск' },
  { id: 'yellow', label: 'Отклонение', short: 'Отклонение' },
  { id: 'green', label: 'В пределах нормы', short: 'Норма' },
  { id: 'none', label: 'Без прогноза', short: 'Нет прогноза' },
];
const priority: Record<Risk | 'none', number> = { red: 0, yellow: 1, green: 2, none: 3 };
const risk = (vehicle: Vehicle): Risk | 'none' => vehicle.prediction?.risk_level ?? 'none';
const connectionLabels = {
  connecting: 'Подключение', syncing: 'Сверка данных', live: 'Поток подключён',
  quiet: 'Поток задерживается', reconnecting: 'Связь потеряна',
};
const connectionShortLabels = {
  connecting: 'Связь…', syncing: 'Сверка', live: 'В сети',
  quiet: 'Задержка', reconnecting: 'Нет связи',
};

export default function App() {
  const [theme, setTheme] = useState<'dark' | 'light'>(() => {
    try { return localStorage.getItem('dashboard-theme') === 'light' ? 'light' : 'dark'; }
    catch { return 'dark'; }
  });
  useLayoutEffect(() => {
    document.documentElement.dataset.theme = theme;
    try { localStorage.setItem('dashboard-theme', theme); } catch { /* Storage may be unavailable. */ }
  }, [theme]);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [filter, setFilter] = useState<Filter>('all');
  const [query, setQuery] = useState('');
  const { snapshot, connection, card, cardLoading, cardError, refreshCard } = useDashboard(selectedId);
  const vehicles = snapshot?.vehicles ?? [];
  const results = useMemo(() => vehicles.filter(vehicle =>
    (filter === 'all' || risk(vehicle) === filter) &&
    (!query.trim() || `${vehicle.tr_id}`.includes(query.trim()) || `${vehicle.unit_id}`.includes(query.trim()))
  ).sort((a, b) => priority[risk(a)] - priority[risk(b)] || a.tr_id - b.tr_id), [vehicles, filter, query]);
  const selectedVehicle = vehicles.find(vehicle => vehicle.tr_id === selectedId) ?? null;
  const count = (id: Filter) => id === 'all' ? vehicles.length : snapshot?.summary.risk[id] ?? 0;

  return <div className="app-shell">
    <header className="topbar">
      <div className="brand"><div className="brand-mark" aria-hidden="true"><span /><span /><span /></div><div><span className="brand-name">КОНТУР</span><span className="brand-caption">Диспетчерская транспорта</span></div></div>
      <div className="topbar-meta"><div className={`connection-status ${connection}`} role="status" aria-live="polite"><span className="connection-dot" /><span className="connection-full">{connectionLabels[connection]}</span><span className="connection-compact">{connectionShortLabels[connection]}</span></div><button className="theme-toggle" type="button" onClick={() => setTheme(value => value === 'dark' ? 'light' : 'dark')} aria-label={theme === 'dark' ? 'Включить светлую тему' : 'Включить тёмную тему'} title={theme === 'dark' ? 'Светлая тема' : 'Тёмная тема'}><Icon name={theme === 'dark' ? 'sun' : 'moon'} size={18} /><span>{theme === 'dark' ? 'Светлая' : 'Тёмная'}</span></button><div className="data-clock"><Icon name="clock" size={18} /><div><span>{date(snapshot?.clock ?? null)}</span><strong>{time(snapshot?.clock ?? null, true)} <small>UTC</small></strong></div></div></div>
    </header>
    <div className="workspace-heading"><div><span className="eyebrow">МОНИТОРИНГ ДВИЖЕНИЯ</span><h1>Оперативная обстановка</h1><p>Положение транспорта и прогноз отклонения от расписания</p></div><span className="snapshot-label">{snapshot ? `Снимок № ${snapshot.seq}` : 'Ожидание данных'}</span></div>
    <nav className="summary-bar" aria-label="Фильтр по риску">{filters.map(item => <button className={`summary-filter ${filter === item.id ? 'active' : ''} ${item.id}`} type="button" key={item.id} onClick={() => setFilter(item.id)} aria-pressed={filter === item.id}><span className="filter-title"><span className="filter-dot" />{item.label}</span><strong>{snapshot ? count(item.id) : '—'}</strong><span className="filter-short">{item.short}</span></button>)}</nav>
    <main className="dashboard-grid">
      <section className="fleet-panel" aria-label="Список транспорта"><div className="panel-header"><span className="eyebrow">ТРАНСПОРТ</span><span className="panel-number">01 / 03</span></div><div className="fleet-controls"><label className="search-box"><Icon name="search" size={18} /><span className="sr-only">Поиск по ID транспорта или борта</span><input type="search" value={query} onChange={event => setQuery(event.target.value)} placeholder="Поиск по ID транспорта" /></label><div className="result-count">{snapshot ? `${results.length} из ${vehicles.length} ТС` : 'Ожидание списка'}<span>По приоритету риска</span></div></div>
        {results.length ? <ul className="fleet-list">{results.map(vehicle => { const level = risk(vehicle); const active = selectedId === vehicle.tr_id; return <li key={vehicle.tr_id}><button type="button" className={`fleet-item ${active ? 'selected' : ''}`} onClick={() => setSelectedId(vehicle.tr_id)} aria-pressed={active}><span className={`fleet-risk ${level} ${vehicle.freshness === 'offline' ? 'offline' : ''}`} /><span className="fleet-item-body"><span className="fleet-item-main"><strong>ТС {vehicle.tr_id}</strong><span className={`risk-text ${level}`}>{riskLabels[level]}</span></span><span className="fleet-item-detail"><span>{freshnessLabels[vehicle.freshness]}</span><span>{vehicle.prediction ? delay(vehicle.prediction.prediction_s) : '—'}</span></span><span className="fleet-item-time">Последние данные: {time(vehicle.last_seen, true)} UTC</span></span><Icon name="arrow" size={16} /></button></li>; })}</ul> : <div className="list-empty"><Icon name={snapshot ? 'search' : 'activity'} size={28} /><strong>{!snapshot ? connection === 'reconnecting' ? 'API недоступен' : 'Загружаем состояние' : vehicles.length === 0 ? 'Транспорт пока не поступил' : 'Ничего не найдено'}</strong><p>{!snapshot ? 'Проверьте соединение с API. Повторная попытка выполняется автоматически.' : vehicles.length === 0 ? 'Список заполнится после поступления телеметрии.' : 'Измените запрос или выберите другой фильтр.'}</p></div>}
      </section>
      <MapBoundary><FleetMap vehicles={vehicles} selectedId={selectedId} card={card} onSelect={setSelectedId} theme={theme} /></MapBoundary>
      <Details id={selectedId} vehicle={selectedVehicle} card={card} loading={cardLoading} error={cardError} onClose={() => setSelectedId(null)} onRetry={refreshCard} snapshot={snapshot} />
    </main>
    <footer className="app-footer"><span>Данные отображаются по времени источника · UTC</span><span>Прогноз относится к целевой остановке на горизонте 10–15 минут</span></footer>
  </div>;
}
