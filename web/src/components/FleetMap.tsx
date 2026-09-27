import { useEffect, useRef, useState } from 'react';
import maplibregl from 'maplibre-gl';
import type { GeoJSONSource, Map as MapInstance, Marker } from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import { hasPosition } from '../protocol';
import { riskLabels } from '../format';
import type { Vehicle, VehicleCard } from '../types';
import { Icon } from './Icon';

interface Props { vehicles: Vehicle[]; selectedId: number | null; card: VehicleCard | null; onSelect: (id: number) => void }
export function FleetMap({ vehicles, selectedId, card, onSelect }: Props) {
  const container = useRef<HTMLDivElement>(null);
  const map = useRef<MapInstance | null>(null);
  const markers = useRef<Map<number, Marker>>(new Map());
  const select = useRef(onSelect); select.current = onSelect;
  const fitted = useRef(false);
  const [ready, setReady] = useState(false);
  const [mapError, setMapError] = useState(false);
  const [tileError, setTileError] = useState(false);
  const [basemap, setBasemap] = useState(false);
  const located = vehicles.filter(hasPosition);
  const fit = () => {
    if (!map.current || !located.length) return;
    const bounds = new maplibregl.LngLatBounds();
    located.forEach(vehicle => bounds.extend([vehicle.lon!, vehicle.lat!]));
    map.current.fitBounds(bounds, { padding: 85, maxZoom: 14, duration: 0 });
  };
  useEffect(() => {
    if (!container.current) return;
    let instance: MapInstance;
    try {
      instance = new maplibregl.Map({ container: container.current, center: [0, 0], zoom: 1,
        attributionControl: false,
        style: { version: 8, sources: {}, layers: [] },
      });
      map.current = instance;
      instance.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'bottom-right');
      instance.addControl(new maplibregl.AttributionControl({ compact: false }), 'bottom-left');
      instance.on('load', () => {
        instance.addSource('track', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
        instance.addLayer({ id: 'track-line', type: 'line', source: 'track', paint: { 'line-color': '#16857a', 'line-width': 4, 'line-opacity': 0.7 } });
        instance.addSource('stops', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
        instance.addLayer({ id: 'stop-points', type: 'circle', source: 'stops', paint: { 'circle-radius': 5, 'circle-color': '#ffffff', 'circle-stroke-width': 2, 'circle-stroke-color': '#16857a' } });
        setReady(true);
      });
      instance.on('error', event => { if ('sourceId' in event && event.sourceId === 'osm') setTileError(true); });
    } catch { setMapError(true); return; }
    const observer = new ResizeObserver(() => instance.resize());
    observer.observe(container.current);
    const currentMarkers = markers.current;
    return () => { observer.disconnect(); currentMarkers.forEach(marker => marker.remove()); currentMarkers.clear(); instance.remove(); map.current = null; };
  }, []);
  useEffect(() => {
    const instance = map.current;
    if (!ready || !instance) return;
    for (const [id, marker] of markers.current) {
      if (!vehicles.some(vehicle => vehicle.tr_id === id && hasPosition(vehicle))) { marker.remove(); markers.current.delete(id); }
    }
    for (const vehicle of vehicles.filter(hasPosition)) {
      let marker = markers.current.get(vehicle.tr_id);
      if (!marker) {
        const button = document.createElement('button');
        button.type = 'button';
        button.addEventListener('click', () => select.current(vehicle.tr_id));
        marker = new maplibregl.Marker({ element: button }).setLngLat([vehicle.lon!, vehicle.lat!]).addTo(instance);
        markers.current.set(vehicle.tr_id, marker);
      }
      const element = marker.getElement();
      const risk = vehicle.freshness === 'offline' ? 'none' : vehicle.prediction?.risk_level ?? 'none';
      element.className = `vehicle-marker ${risk} ${selectedId === vehicle.tr_id ? 'selected' : ''}`;
      element.textContent = `${vehicle.tr_id}`;
      element.setAttribute('aria-label', `ТС ${vehicle.tr_id}: ${riskLabels[risk]}`);
      element.setAttribute('aria-pressed', String(selectedId === vehicle.tr_id));
      marker.setLngLat([vehicle.lon!, vehicle.lat!]);
    }
    if (!fitted.current && vehicles.some(hasPosition)) { fit(); fitted.current = true; }
  }, [vehicles, selectedId, ready]);
  useEffect(() => {
    const vehicle = vehicles.find(item => item.tr_id === selectedId);
    if (ready && vehicle && hasPosition(vehicle)) map.current?.easeTo({ center: [vehicle.lon!, vehicle.lat!], zoom: Math.max(map.current.getZoom(), 13), duration: 0 });
  }, [selectedId, ready]);
  useEffect(() => {
    if (!ready || !map.current) return;
    (map.current.getSource('track') as GeoJSONSource).setData({ type: 'FeatureCollection', features: card && card.track.length > 1 ? [{ type: 'Feature', properties: {}, geometry: { type: 'LineString', coordinates: card.track.map(point => [point.lon, point.lat]) } }] : [] });
    (map.current.getSource('stops') as GeoJSONSource).setData({ type: 'FeatureCollection', features: (card?.stops ?? []).map(stop => ({ type: 'Feature', properties: {}, geometry: { type: 'Point', coordinates: [stop.lon, stop.lat] } })) });
  }, [card, ready]);
  useEffect(() => {
    const instance = map.current;
    if (!ready || !instance) return;
    if (basemap && !instance.getSource('osm')) {
      instance.addSource('osm', { type: 'raster', tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'], tileSize: 256, attribution: '© <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noreferrer">OpenStreetMap</a> contributors', maxzoom: 19 });
      instance.addLayer({ id: 'osm', type: 'raster', source: 'osm', paint: { 'raster-saturation': -0.75, 'raster-opacity': 0.65 } }, 'track-line');
    }
    if (instance.getLayer('osm')) instance.setLayoutProperty('osm', 'visibility', basemap ? 'visible' : 'none');
  }, [basemap, ready]);
  return <section className="map-panel" aria-label="Карта транспорта">
    <div className="map-canvas" ref={container} />
    <div className="map-heading"><span className="eyebrow">ОПЕРАТИВНАЯ КАРТА</span><span>{located.length} ТС с координатами</span></div>
    <div className="map-actions">
      <button className={`map-button ${basemap ? 'active' : ''}`} aria-pressed={basemap} onClick={() => { setBasemap(value => !value); setTileError(false); }} disabled={mapError}><Icon name="layers" />Подложка</button>
      <button className="map-button icon-only" title="Показать весь транспорт" aria-label="Показать весь транспорт" onClick={fit} disabled={!located.length || mapError}><Icon name="locate" /></button>
    </div>
    {!located.length && <div className="map-empty"><div className="map-empty-symbol"><Icon name="map" size={36} /></div><h3>Город в поле зрения</h3><p>Положения транспорта появятся здесь,<br />когда поступит телеметрия.</p></div>}
    {mapError && <div className="map-warning" role="status">Карта недоступна в этом браузере. Выберите транспорт в списке.</div>}
    {tileError && basemap && <div className="map-warning" role="status">Подложка недоступна. Положения транспорта и трек продолжают обновляться.</div>}
    <div className="map-footnote"><span className="coordinate-cross">⌖</span>{basemap ? 'Подложка OpenStreetMap' : 'Координатное поле · без подложки'}{vehicles.length > located.length && <span> · Без координат: {vehicles.length - located.length}</span>}</div>
    {card?.track.length ? <div className="track-key"><span />Последний трек · {card.track.length} точек</div> : null}
  </section>;
}
