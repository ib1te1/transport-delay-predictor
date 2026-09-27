import assert from 'node:assert/strict';
import test from 'node:test';
import { delay } from '../src/format.ts';
import { nearestPlannedStopKm, networkLines, trackSections } from '../src/network.ts';
import { applyMessage, incidents, mergeHistory, reconcile, SequenceGap, summarize } from '../src/protocol.ts';
import type { Alert, Message, Prediction, Snapshot, Vehicle } from '../src/types.ts';

const prediction: Prediction = {
  sample_id: 'sample-1', t: '2026-09-27T10:00:00Z', target_stop_id: 8,
  target_address: null, target_time_begin: '2026-09-27T10:12:00Z',
  prediction_s: -90, p_late: null, reasons: [], risk_level: 'yellow',
  degraded: false, degraded_reason: null, model_version: 'v1',
};
const vehicle: Vehicle = {
  tr_id: 7, unit_id: 40, lat: 51.5, lon: 46.0, heading_deg: null,
  speed_kmh: 22, last_seen: '2026-09-27T10:00:00Z', freshness: 'active', prediction,
};
const snapshot: Snapshot = {
  seq: 3, clock: '2026-09-27T10:00:00Z', vehicles: [vehicle],
  summary: summarize([vehicle]), alerts: [], live_mae_s: null, checked_predictions: 0,
};

test('snapshot buffer skips old messages and applies new sequence', () => {
  const messages: Message[] = [
    { type: 'clock', data: { seq: 2, clock: '2026-09-27T09:59:59Z' } },
    { type: 'clock', data: { seq: 4, clock: '2026-09-27T10:00:01Z' } },
  ];
  const state = reconcile(snapshot, messages);
  assert.equal(state.seq, 4);
  assert.equal(state.clock, '2026-09-27T10:00:01Z');
  assert.equal(applyMessage(state, messages[1]), state);
});

test('gap requires a new snapshot, including after server sequence reset', () => {
  assert.throws(() => applyMessage(snapshot, { type: 'clock', data: { seq: 5, clock: snapshot.clock! } }), SequenceGap);
  const restarted = { ...snapshot, seq: 0 };
  assert.equal(applyMessage(restarted, { type: 'clock', data: { seq: 1, clock: snapshot.clock! } }).seq, 1);
});

test('vehicle diff updates counters, removes entries, and leaves predictions out of the map', () => {
  const update: Message = { type: 'vehicles', data: {
    seq: 4, clock: snapshot.clock!, removed: [7], vehicles: [{ ...vehicle, tr_id: 9, freshness: 'offline', prediction: null }],
  } };
  const state = applyMessage(snapshot, update);
  assert.deepEqual(state.vehicles.map(item => item.tr_id), [9]);
  assert.equal(state.summary.freshness.offline, 1);
  assert.equal(state.summary.risk.none, 1);
  const changed = applyMessage(state, { type: 'prediction', data: { seq: 5, tr_id: 9, prediction } });
  assert.equal(changed.vehicles[0].prediction, null);
});

test('negative prediction means early arrival and keeps server risk', () => {
  assert.equal(delay(prediction.prediction_s), '−1,5 мин');
  assert.equal(summarize([vehicle]).risk.yellow, 1);
});

test('card history applies only newer selected-vehicle events, deduplicated by sample', () => {
  const card = { seq: 7, clock: snapshot.clock, vehicle, predictions: [prediction], track: [], stops: [] };
  const next = { ...prediction, sample_id: 'sample-2', t: '2026-09-27T10:01:00Z' };
  const messages: Message[] = [
    { type: 'prediction', data: { seq: 7, tr_id: 7, prediction: next } },
    { type: 'prediction', data: { seq: 8, tr_id: 9, prediction: next } },
    { type: 'prediction', data: { seq: 9, tr_id: 7, prediction: next } },
    { type: 'prediction', data: { seq: 10, tr_id: 7, prediction: next } },
  ];
  assert.deepEqual(mergeHistory(card, 7, messages).predictions.map(item => item.sample_id), ['sample-2', 'sample-1']);
});

test('network lines stay within each run and follow stop order', () => {
  const stop = (route_id: number, stop_order: number, lon: number) => ({
    route_id, stop_order, stop_id: route_id * 10 + stop_order,
    address: null, lat: 55, lon, time_plan: '2026-01-06T08:00:00Z',
  });
  const lines = networkLines([stop(7, 2, 2), stop(8, 1, 8), stop(7, 1, 1), stop(7, 3, 3)]);
  assert.equal(lines.features.length, 1);
  assert.equal(lines.features[0].properties?.route_id, 7);
  assert.deepEqual(lines.features[0].geometry.coordinates, [[1, 55], [2, 55], [3, 55]]);
});

test('planned lines stop at a layover instead of bridging separate trips', () => {
  const stop = (stop_order: number, minute: number, lon: number) => ({
    route_id: 122658, stop_order, stop_id: stop_order,
    address: null, lat: 55.8, lon,
    time_plan: `2026-01-06T10:${String(minute).padStart(2, '0')}:00Z`,
  });
  const route = [stop(1, 0, 37.36), stop(2, 2, 37.37), stop(3, 20, 37.45), stop(4, 22, 37.46)];
  const lines = networkLines(route);
  assert.deepEqual(lines.features.map(line => line.geometry.coordinates), [
    [[37.36, 55.8], [37.37, 55.8]],
    [[37.45, 55.8], [37.46, 55.8]],
  ]);
  assert.ok(nearestPlannedStopKm(55.975, 37.43, route)! > 18);
});

test('GPS away from planned stops is shown separately', () => {
  const route = [{ route_id: 122658, stop_order: 1, stop_id: 1, address: null,
    lat: 55.8, lon: 37.4, time_plan: '2026-01-06T10:00:00Z' }];
  const track = [
    { event_time: '2026-01-06T10:00:00Z', lat: 55.8, lon: 37.4, speed_kmh: 0 },
    { event_time: '2026-01-06T10:01:00Z', lat: 55.801, lon: 37.4, speed_kmh: 5 },
    { event_time: '2026-01-06T10:02:00Z', lat: 55.975, lon: 37.43, speed_kmh: 0 },
  ];
  const sections = trackSections(track, route);
  assert.equal(sections.onRoute.features.length, 1);
  assert.equal(sections.away.features.length, 1);
  assert.deepEqual(sections.away.features[0].geometry.coordinates[1], [37.43, 55.975]);
});

test('GPS track does not connect positions across missing telemetry', () => {
  const track = [
    { event_time: '2026-01-06T12:38:57Z', lat: 55.964363, lon: 37.43, speed_kmh: 99 },
    { event_time: '2026-01-06T12:46:14Z', lat: 55.802868, lon: 37.43, speed_kmh: 12 },
  ];
  const sections = trackSections(track, []);
  assert.equal(sections.onRoute.features.length, 0);
  assert.equal(sections.away.features.length, 0);
});

const alert = (overrides: Partial<Alert>): Alert => ({
  id: 1, tr_id: 7, target_stop_id: 8, segment_from_stop_id: 5, status: 'open',
  opened_at: '2026-09-27T10:00:00Z', closed_at: null, predicted_delay_s: 400,
  reasons: ['accumulated_delay'], actual_delay_s: null, lead_time_s: null, ...overrides,
});

test('alert messages keep only open alerts in the snapshot', () => {
  const opened = applyMessage(snapshot, { type: 'alert', data: { seq: 4, alert: alert({}) } });
  assert.deepEqual(opened.alerts.map(item => item.id), [1]);
  const updated = applyMessage(opened, { type: 'alert', data: { seq: 5, alert: alert({ predicted_delay_s: 500 }) } });
  assert.deepEqual(updated.alerts.map(item => item.predicted_delay_s), [500]);
  const closed = applyMessage(updated, { type: 'alert', data: { seq: 6, alert: alert({ status: 'cancelled', closed_at: '2026-09-27T10:05:00Z' }) } });
  assert.deepEqual(closed.alerts, []);
});

test('incidents take the newest open alert of each vehicle, largest delay first', () => {
  const list = incidents([
    alert({ id: 1, tr_id: 7, opened_at: '2026-09-27T10:00:00Z', predicted_delay_s: 900 }),
    alert({ id: 2, tr_id: 7, opened_at: '2026-09-27T10:01:00Z', predicted_delay_s: 350 }),
    alert({ id: 3, tr_id: 9, opened_at: '2026-09-27T10:00:00Z', predicted_delay_s: 600 }),
    alert({ id: 4, tr_id: 11, status: 'confirmed', predicted_delay_s: 1200 }),
  ]);
  assert.deepEqual(list.map(item => item.id), [3, 2]);
});
