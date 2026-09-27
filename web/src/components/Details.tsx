import { useEffect, useRef } from 'react';
import { date, degradedLabels, delay, freshnessLabels, reasonLabels, riskLabels, time } from '../format';
import { hasPosition, incidents } from '../protocol';
import type { Alert, NetworkStop, Prediction, Snapshot, Vehicle, VehicleCard } from '../types';
import { Icon } from './Icon';

function quality(snapshot: Snapshot | null) {
  return <div className="quality-panel">
    <div className="quality-title"><Icon name="activity" size={18} />Качество данных</div>
    <div className="quality-row"><span>Телеметрия актуальна</span><strong>{snapshot?.summary.freshness.active ?? '—'}</strong></div>
    <div className="quality-row"><span>Устарела</span><strong>{snapshot?.summary.freshness.stale ?? '—'}</strong></div>
    <div className="quality-row"><span>Нет связи с ТС</span><strong>{snapshot?.summary.freshness.offline ?? '—'}</strong></div>
    <div className="quality-row"><span>Проверка прогнозов · MAE</span><strong>{snapshot?.live_mae_s == null ? 'Ещё нет проверок' : `${(snapshot.live_mae_s / 60).toLocaleString('ru-RU', { maximumFractionDigits: 1 })} мин`}</strong></div>
  </div>;
}

function predictionType(seconds: number) {
  return seconds < 0 ? 'Опережение' : seconds > 0 ? 'Задержка' : 'По расписанию';
}

function probability(value: number | null) {
  return value === null ? 'Нет оценки' : `${(value * 100).toLocaleString('ru-RU', { maximumFractionDigits: 1 })} %`;
}

function CurrentPrediction({ prediction }: { prediction: Prediction }) {
  return <section className="prediction-panel" aria-label="Текущий прогноз">
    <div className="prediction-topline"><span>ТЕКУЩИЙ ПРОГНОЗ</span><span className={`risk-badge ${prediction.risk_level}`}>{riskLabels[prediction.risk_level]}</span></div>
    <div className="prediction-value"><strong>{delay(prediction.prediction_s)}</strong><span>{predictionType(prediction.prediction_s)}</span></div>
    <p className="prediction-target">К целевой остановке <strong>№ {prediction.target_stop_id}</strong>{prediction.target_address ? ` · ${prediction.target_address}` : ''}</p>
    <div className="prediction-facts">
      <div><span>Плановое прибытие</span><strong>{time(prediction.target_time_begin)} UTC</strong></div>
      <div><span>Вероятность опоздания &gt; 2 мин</span><strong>{probability(prediction.p_late)}</strong></div>
      <div><span>Время прогноза</span><strong>{time(prediction.t, true)} UTC</strong></div>
    </div>
    {prediction.degraded && <div className="degraded-note" role="status"><Icon name="info" size={18} /><span>Резервный расчёт{prediction.degraded_reason ? `: ${degradedLabels[prediction.degraded_reason]}` : ''}</span></div>}
    {prediction.reasons.length > 0 && <div className="reason-group"><h4>Причины прогноза</h4><div className="reason-list">{prediction.reasons.map(reason => <span className="reason-tag" key={reason}>{reasonLabels[reason]}</span>)}</div></div>}
    <p className="model-caption">Модель: {prediction.model_version}</p>
  </section>;
}

function stopName(id: number | null, stops: Map<number, NetworkStop>) {
  if (id === null) return null;
  return stops.get(id)?.address || `Остановка № ${id}`;
}

function segment(alert: Alert, stops: Map<number, NetworkStop>) {
  const from = stopName(alert.segment_from_stop_id, stops);
  const to = stopName(alert.target_stop_id, stops);
  return from ? `${from} → ${to}` : `до ${to}`;
}

function IncidentFacts({ alert, stops }: { alert: Alert; stops: Map<number, NetworkStop> }) {
  const planned = stops.get(alert.target_stop_id)?.time_plan ?? null;
  return <>
    <span className="incident-segment">Участок: {segment(alert, stops)}</span>
    <span className="incident-meta">Алерт с {time(alert.opened_at)} UTC{planned ? ` · плановое прибытие ${time(planned)} UTC` : ''}</span>
    {alert.reasons.length > 0 && <span className="reason-list">{alert.reasons.map(reason => <span className="reason-tag" key={reason}>{reasonLabels[reason]}</span>)}</span>}
  </>;
}

function IncidentList({ snapshot, stops, onSelect }: { snapshot: Snapshot | null; stops: Map<number, NetworkStop>; onSelect: (id: number) => void }) {
  const items = incidents(snapshot?.alerts ?? []);
  return <section className="incident-panel" aria-label="Инциденты">
    <div className="section-title"><h3>Инциденты: риск опоздания</h3><span>{items.length}</span></div>
    {items.length ? <ol className="incident-list">{items.map(alert => <li key={alert.id}><button type="button" className="incident-item" onClick={() => onSelect(alert.tr_id)}>
      <span className="incident-head"><strong>ТС {alert.tr_id}</strong><span className="risk-badge red">Опоздание {delay(alert.predicted_delay_s)}</span></span>
      <IncidentFacts alert={alert} stops={stops} />
    </button></li>)}</ol> : <p className="incident-empty">Открытых инцидентов нет. Здесь появятся ТС с высоким риском опоздания.</p>}
  </section>;
}

interface Props {
  id: number | null;
  vehicle: Vehicle | null;
  card: VehicleCard | null;
  loading: boolean;
  error: string;
  snapshot: Snapshot | null;
  stops: Map<number, NetworkStop>;
  onSelect: (id: number) => void;
  onClose: () => void;
  onRetry: () => void;
}

export function Details({ id, vehicle, card, loading, error, snapshot, stops, onSelect, onClose, onRetry }: Props) {
  const heading = useRef<HTMLHeadingElement>(null);
  useEffect(() => {
    if (id !== null && window.matchMedia('(max-width: 900px)').matches) heading.current?.focus();
  }, [id]);

  if (id === null) return <aside className="detail-panel welcome-panel" aria-label="Карточка транспорта">
    <div className="panel-header"><span className="eyebrow">КАРТОЧКА ТРАНСПОРТА</span><span className="panel-number">03 / 03</span></div>
    <div className="detail-scroll">
      <IncidentList snapshot={snapshot} stops={stops} onSelect={onSelect} />
      <p className="incident-hint">Выберите ТС в списке, на карте или среди инцидентов, чтобы увидеть прогноз, трек и остановки.</p>
    </div>
    {quality(snapshot)}
  </aside>;

  const current = vehicle ?? card?.vehicle ?? null;
  const prediction = current?.prediction ?? null;
  const incident = incidents(snapshot?.alerts ?? []).find(alert => alert.tr_id === id) ?? null;
  return <aside className="detail-panel" aria-label={`Карточка ТС ${id}`}>
    <div className="panel-header"><span className="eyebrow">КАРТОЧКА ТРАНСПОРТА</span><button className="icon-button" type="button" onClick={onClose} aria-label="Закрыть карточку"><Icon name="close" size={19} /></button></div>
    <div className="detail-scroll">
      <div className="detail-title-row"><div><h2 ref={heading} tabIndex={-1}>ТС {id}</h2><span className="detail-subtitle">{current ? `Бортовой ID ${current.unit_id}` : 'Данные о транспорте'}</span></div>{current && <span className={`freshness-pill ${current.freshness}`}>{freshnessLabels[current.freshness]}</span>}</div>
      {current && <div className="vehicle-metrics"><div><span>Последняя телеметрия</span><strong>{time(current.last_seen, true)} UTC</strong></div><div><span>Скорость</span><strong>{current.speed_kmh === null ? 'Нет данных' : `${Math.round(current.speed_kmh)} км/ч`}</strong></div><div><span>Положение</span><strong>{hasPosition(current) ? 'На карте' : 'Без координат'}</strong></div></div>}
      {error && <div className="inline-error" role="alert"><span>{error}</span><button type="button" onClick={onRetry}>Повторить</button></div>}
      {loading && <p className="loading-note" role="status">Обновляем трек и остановки…</p>}
      {incident && <section className="incident-card" role="status" aria-label="Инцидент"><span className="incident-head"><strong>Инцидент</strong><span className="risk-badge red">Опоздание {delay(incident.predicted_delay_s)}</span></span><IncidentFacts alert={incident} stops={stops} /></section>}
      {prediction ? <CurrentPrediction prediction={prediction} /> : <div className="no-prediction"><Icon name="clock" size={22} /><div><strong>Активного прогноза пока нет</strong><p>Прогноз появится, когда для ТС будет определена целевая остановка.</p></div></div>}
      <section className="detail-section"><div className="section-title"><h3>Последний фактический трек</h3><span>{card?.track.length ?? 0} точек</span></div>{card?.track.length ? <p>От {time(card.track[0].event_time, true)} до {time(card.track[card.track.length - 1].event_time, true)} UTC. Линия показана на карте.</p> : <p>Точки с координатами пока не поступили.</p>}</section>
      <section className="detail-section"><div className="section-title"><h3>Остановки</h3><span>{card?.stops.length ?? 0}</span></div>{card?.stops.length ? <ol className="stop-list">{card.stops.map(stop => <li key={`${stop.stop_id}-${stop.time_plan}`}><span className={`stop-node ${stop.time_fact ? 'passed' : ''}`} /><div><strong>{stop.address || `Остановка № ${stop.stop_id}`}</strong><small>План {time(stop.time_plan)} UTC{stop.time_fact ? ` · Факт ${time(stop.time_fact)} UTC` : ' · Факт не получен'}</small>{stop.delay_s !== null && <small>{predictionType(stop.delay_s)} {delay(stop.delay_s)}</small>}</div></li>)}</ol> : <p>Остановок в текущем окне данных нет.</p>}</section>
      {card?.predictions.length ? <section className="detail-section"><div className="section-title"><h3>История прогнозов</h3><span>{card.predictions.length}</span></div><ol className="history-list">{card.predictions.slice(0, 8).map(item => <li key={item.sample_id}><span>{time(item.t, true)} UTC · остановка № {item.target_stop_id}</span><strong className={item.risk_level}>{delay(item.prediction_s)}</strong></li>)}</ol></section> : null}
      <p className="detail-date">Время данных: {date(card?.clock ?? snapshot?.clock ?? null)} · UTC</p>
    </div>
  </aside>;
}
