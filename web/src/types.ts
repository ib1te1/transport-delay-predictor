// Dashboard wire models mirror services/api/app/schemas.py.
export type Risk = 'green' | 'yellow' | 'red';
export type Freshness = 'active' | 'stale' | 'offline';
export type Reason = 'accumulated_delay' | 'slow_approach' | 'long_dwell' | 'low_speed_segment' | 'stale_telemetry' | 'unknown';
export interface Prediction {
  sample_id: string; t: string; target_stop_id: number; target_address: string | null;
  target_time_begin: string; prediction_s: number; p_late: number | null;
  reasons: Reason[]; risk_level: Risk; degraded: boolean;
  degraded_reason: 'predictor_unavailable' | 'stale_telemetry' | 'warming_up' | null;
  model_version: string;
}
export interface Vehicle {
  tr_id: number; unit_id: number; lat: number | null; lon: number | null;
  heading_deg: number | null; speed_kmh: number | null; last_seen: string;
  freshness: Freshness; prediction: Prediction | null;
}
export interface Alert {
  id: number; tr_id: number; target_stop_id: number; segment_from_stop_id: number | null;
  status: 'open' | 'confirmed' | 'cancelled'; opened_at: string; closed_at: string | null;
  predicted_delay_s: number; reasons: string[]; actual_delay_s: number | null; lead_time_s: number | null;
}
export interface Summary {
  risk: Record<Risk | 'none', number>;
  freshness: Record<Freshness, number>;
}
export interface Snapshot {
  seq: number; clock: string | null; vehicles: Vehicle[]; summary: Summary;
  alerts: Alert[]; live_mae_s: number | null; checked_predictions: number;
}
export interface TrackPoint { event_time: string; lat: number; lon: number; speed_kmh: number | null }
export interface CardStop {
  stop_id: number; address: string | null; lat: number; lon: number;
  time_plan: string; time_fact: string | null; delay_s: number | null;
}
export interface NetworkStop {
  route_id: number; stop_order: number; stop_id: number;
  address: string | null; lat: number; lon: number; time_plan: string;
}
export interface VehicleCard {
  seq: number; clock: string | null; vehicle: Vehicle | null;
  predictions: Prediction[]; track: TrackPoint[]; stops: CardStop[];
}
export type Message =
  | { type: 'clock'; data: { seq: number; clock: string } }
  | { type: 'vehicles'; data: { seq: number; clock: string; vehicles: Vehicle[]; removed: number[] } }
  | { type: 'prediction'; data: { seq: number; tr_id: number; prediction: Prediction } }
  | { type: 'alert'; data: { seq: number; alert: Alert } };
