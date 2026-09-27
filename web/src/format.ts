import type { Reason } from './types.ts';

export const riskLabels = { red: 'Высокий риск', yellow: 'Отклонение', green: 'В пределах нормы', none: 'Без прогноза' };
export const freshnessLabels = { active: 'Телеметрия актуальна', stale: 'Телеметрия устарела', offline: 'Нет связи с ТС' };
export const reasonLabels: Record<Reason, string> = {
  accumulated_delay: 'Накопленное отставание', slow_approach: 'Замедление перед остановкой',
  long_dwell: 'Длительная стоянка', low_speed_segment: 'Низкая скорость на участке',
  stale_telemetry: 'Устаревшая телеметрия', unknown: 'Причина не определена',
};
export const degradedLabels = {
  predictor_unavailable: 'Модель временно недоступна', stale_telemetry: 'Недостаточно свежих данных', warming_up: 'Модель накапливает данные',
};
export function time(value: string | null, seconds = false): string {
  if (!value) return '—';
  return new Intl.DateTimeFormat('ru-RU', { timeZone: 'UTC', hour: '2-digit', minute: '2-digit', ...(seconds ? { second: '2-digit' } : {}) }).format(new Date(value));
}
export function date(value: string | null): string {
  return value ? new Intl.DateTimeFormat('ru-RU', { timeZone: 'UTC', day: 'numeric', month: 'long', year: 'numeric' }).format(new Date(value)) : 'Ожидание времени данных';
}
export function delay(seconds: number): string {
  const value = Math.abs(seconds) / 60;
  return `${seconds > 0 ? '+' : seconds < 0 ? '−' : ''}${value.toLocaleString('ru-RU', { maximumFractionDigits: 1 })} мин`;
}
