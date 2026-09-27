import type { Message, Prediction, Snapshot, Summary, Vehicle, VehicleCard } from './types.ts';

export function summarize(vehicles: Vehicle[]): Summary {
  const summary: Summary = { risk: { green: 0, yellow: 0, red: 0, none: 0 }, freshness: { active: 0, stale: 0, offline: 0 } };
  for (const vehicle of vehicles) {
    summary.risk[vehicle.prediction?.risk_level ?? 'none']++;
    summary.freshness[vehicle.freshness]++;
  }
  return summary;
}

export class SequenceGap extends Error {}

export function applyMessage(state: Snapshot, message: Message): Snapshot {
  if (message.data.seq <= state.seq) return state;
  if (message.data.seq !== state.seq + 1) throw new SequenceGap('Missing stream message');
  const next = { ...state, seq: message.data.seq };
  switch (message.type) {
    case 'clock': return { ...next, clock: message.data.clock };
    case 'vehicles': {
      const vehicles = new Map(state.vehicles.map(vehicle => [vehicle.tr_id, vehicle]));
      for (const id of message.data.removed) vehicles.delete(id);
      for (const vehicle of message.data.vehicles) vehicles.set(vehicle.tr_id, vehicle);
      const all = [...vehicles.values()];
      return { ...next, clock: message.data.clock, vehicles: all, summary: summarize(all) };
    }
    case 'alert': return { ...next, alerts: [...state.alerts.filter(alert => alert.id !== message.data.alert.id), message.data.alert] };
    case 'prediction': return next;
  }
}

export function reconcile(snapshot: Snapshot, buffered: Message[]): Snapshot {
  return buffered.reduce(applyMessage, snapshot);
}

export function mergeHistory(card: VehicleCard, trId: number, messages: Message[]): VehicleCard {
  const predictions = new Map<string, Prediction>(card.predictions.map(prediction => [prediction.sample_id, prediction]));
  for (const message of messages) {
    if (message.type === 'prediction' && message.data.tr_id === trId && message.data.seq > card.seq) {
      predictions.set(message.data.prediction.sample_id, message.data.prediction);
    }
  }
  return { ...card, predictions: [...predictions.values()].sort((a, b) => Date.parse(b.t) - Date.parse(a.t)).slice(0, 240) };
}

export function hasPosition(vehicle: Pick<Vehicle, 'lat' | 'lon'>): boolean {
  return vehicle.lat !== null && vehicle.lon !== null && Number.isFinite(vehicle.lat) && Number.isFinite(vehicle.lon)
    && Math.abs(vehicle.lat) <= 90 && Math.abs(vehicle.lon) <= 180;
}
