import type { NetworkStop } from './types.ts';

export function networkLines(stops: NetworkStop[]): GeoJSON.FeatureCollection<GeoJSON.LineString> {
  const routes = new Map<number, NetworkStop[]>();
  for (const stop of stops) {
    const route = routes.get(stop.route_id) ?? [];
    route.push(stop);
    routes.set(stop.route_id, route);
  }
  return {
    type: 'FeatureCollection',
    features: Array.from(routes, ([routeId, route]) => ({ routeId, route }))
      .filter(({ route }) => route.length >= 2)
      .map(({ routeId, route }) => ({
        type: 'Feature' as const,
        properties: { route_id: routeId },
        geometry: {
          type: 'LineString' as const,
          coordinates: route.sort((a, b) => a.stop_order - b.stop_order || a.stop_id - b.stop_id)
            .map(stop => [stop.lon, stop.lat]),
        },
      })),
  };
}
