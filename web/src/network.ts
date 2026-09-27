import type { NetworkStop, TrackPoint } from './types.ts';

// The timetable treats a pause of seven minutes as the start of another trip.
const TRIP_GAP_MS = 7 * 60 * 1000;
// The API considers telemetry stale after two minutes; do not invent a path over a longer gap.
const TRACK_GAP_MS = 2 * 60 * 1000;
// Visual distinction for GPS far from the planned stops in the demo dataset.
const AWAY_FROM_PLAN_KM = 3;

export function nearestPlannedStopKm(lat: number, lon: number, stops: NetworkStop[]): number | null {
  if (!stops.length) return null;
  const radians = Math.PI / 180;
  return Math.min(...stops.map(stop => {
    const latitudeDelta = (stop.lat - lat) * radians;
    const longitudeDelta = (stop.lon - lon) * radians;
    const halfChord = Math.sin(latitudeDelta / 2) ** 2
      + Math.cos(lat * radians) * Math.cos(stop.lat * radians) * Math.sin(longitudeDelta / 2) ** 2;
    return 12742 * Math.asin(Math.sqrt(halfChord));
  }));
}

export function trackSections(track: TrackPoint[], route: NetworkStop[]) {
  const onRoute: GeoJSON.Feature<GeoJSON.LineString>[] = [];
  const away: GeoJSON.Feature<GeoJSON.LineString>[] = [];
  const distant = route.length ? track.map(point =>
    (nearestPlannedStopKm(point.lat, point.lon, route) ?? 0) > AWAY_FROM_PLAN_KM) : [];
  for (let index = 1; index < track.length; index++) {
    const previous = track[index - 1];
    const current = track[index];
    if (Date.parse(current.event_time) - Date.parse(previous.event_time) > TRACK_GAP_MS) continue;
    const line: GeoJSON.Feature<GeoJSON.LineString> = {
      type: 'Feature', properties: {}, geometry: {
        type: 'LineString', coordinates: [[previous.lon, previous.lat], [current.lon, current.lat]],
      },
    };
    (distant[index - 1] || distant[index] ? away : onRoute).push(line);
  }
  return {
    onRoute: { type: 'FeatureCollection' as const, features: onRoute },
    away: { type: 'FeatureCollection' as const, features: away },
  };
}

export function networkLines(stops: NetworkStop[]): GeoJSON.FeatureCollection<GeoJSON.LineString> {
  const routes = new Map<number, NetworkStop[]>();
  for (const stop of stops) {
    const route = routes.get(stop.route_id) ?? [];
    route.push(stop);
    routes.set(stop.route_id, route);
  }
  return {
    type: 'FeatureCollection',
    features: Array.from(routes, ([routeId, route]) => {
      route.sort((a, b) => a.stop_order - b.stop_order || a.stop_id - b.stop_id);
      const trips: NetworkStop[][] = [];
      let trip: NetworkStop[] = [];
      for (const stop of route) {
        const previous = trip.at(-1);
        if (previous && Date.parse(stop.time_plan) - Date.parse(previous.time_plan) >= TRIP_GAP_MS) {
          trips.push(trip);
          trip = [];
        }
        trip.push(stop);
      }
      trips.push(trip);
      return trips.filter(stopsInTrip => stopsInTrip.length >= 2).map(stopsInTrip => ({
        type: 'Feature' as const,
        properties: { route_id: routeId },
        geometry: {
          type: 'LineString' as const,
          coordinates: stopsInTrip.map(stop => [stop.lon, stop.lat]),
        },
      }));
    }).flat(),
  };
}
