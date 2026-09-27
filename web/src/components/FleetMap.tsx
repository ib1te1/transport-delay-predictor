import { useEffect, useRef, useState } from 'react';
import maplibregl from 'maplibre-gl';
import type { GeoJSONSource, Map as MapInstance, Marker } from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import { hasPosition } from '../protocol';
import { riskLabels } from '../format';
import { nearestPlannedStopKm, networkLines, trackSections } from '../network';
import type { NetworkStop, Vehicle, VehicleCard } from '../types';
import { Icon } from './Icon';

interface Props {
  vehicles: Vehicle[]; selectedId: number | null; card: VehicleCard | null; onSelect: (id: number) => void;
  network: NetworkStop[] | null; networkError: boolean; theme: 'dark' | 'light';
}
const mapPadding = { top: 75, right: 70, bottom: 125, left: 70 };
export function FleetMap({ vehicles, selectedId, card, onSelect, network, networkError, theme }: Props) {
  const container = useRef<HTMLDivElement>(null);
  const map = useRef<MapInstance | null>(null);
  const markers = useRef<Map<number, Marker>>(new Map());
  const select = useRef(onSelect); select.current = onSelect;
  const fitted = useRef(false);
  const fittedIds = useRef<Set<number>>(new Set());
  const autoFit = useRef(true);
  const followSelected = useRef(true);
  const [ready, setReady] = useState(false);
  const [mapError, setMapError] = useState(false);
  const [tileError, setTileError] = useState(false);
  const [basemap, setBasemap] = useState(false);
  const located = vehicles.filter(hasPosition);
  const selectedRoute = network?.filter(stop => stop.route_id === selectedId) ?? [];
  const selectedVehicle = vehicles.find(vehicle => vehicle.tr_id === selectedId);
  const nearestStopKm = selectedVehicle && hasPosition(selectedVehicle)
    ? nearestPlannedStopKm(selectedVehicle.lat!, selectedVehicle.lon!, selectedRoute) : null;
  const selectedTrack = card?.vehicle?.tr_id === selectedId ? card.track : [];
  const sections = trackSections(selectedTrack, selectedRoute);
  const fit = () => {
    if (!map.current || (!located.length && !network?.length)) return;
    map.current.resize();
    const bounds = new maplibregl.LngLatBounds();
    if (located.length) located.forEach(vehicle => bounds.extend([vehicle.lon!, vehicle.lat!]));
    else network?.forEach(stop => bounds.extend([stop.lon, stop.lat]));
    map.current.fitBounds(bounds, { padding: mapPadding, maxZoom: 14, duration: 0 });
  };
  const fitSelected = (vehicle: Vehicle) => {
    const instance = map.current;
    if (!instance || !hasPosition(vehicle)) return;
    instance.resize();
    const route = network?.filter(stop => stop.route_id === vehicle.tr_id) ?? [];
    const trace = card?.vehicle?.tr_id === vehicle.tr_id ? card.track : [];
    if (!route.length && !trace.length) {
      instance.easeTo({ center: [vehicle.lon!, vehicle.lat!], zoom: Math.max(instance.getZoom(), 13), duration: 0 });
      return;
    }
    const bounds = new maplibregl.LngLatBounds();
    bounds.extend([vehicle.lon!, vehicle.lat!]);
    route.forEach(stop => bounds.extend([stop.lon, stop.lat]));
    trace.forEach(point => bounds.extend([point.lon, point.lat]));
    instance.fitBounds(bounds, { padding: mapPadding, maxZoom: 13, duration: 0 });
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
      instance.on('movestart', event => {
        if (event.originalEvent) { autoFit.current = false; followSelected.current = false; }
      });
      instance.on('load', () => {
        instance.addSource('network', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
        instance.addLayer({ id: 'network-lines', type: 'line', source: 'network', paint: { 'line-color': '#789ba6', 'line-width': 2, 'line-opacity': 0.65 } });
        instance.addLayer({ id: 'selected-route', type: 'line', source: 'network', filter: ['==', ['get', 'route_id'], -1], paint: { 'line-color': '#16857a', 'line-width': 4, 'line-opacity': 0.95, 'line-dasharray': [2, 1] } });
        instance.addSource('track', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
        instance.addLayer({ id: 'track-line', type: 'line', source: 'track', paint: { 'line-color': '#16857a', 'line-width': 4, 'line-opacity': 0.7 } });
        instance.addSource('track-away', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
        instance.addLayer({ id: 'track-away-line', type: 'line', source: 'track-away', paint: { 'line-color': '#c79738', 'line-width': 4, 'line-opacity': 0.9, 'line-dasharray': [2, 1] } });
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
    if (!ready || !map.current || network === null) return;
    (map.current.getSource('network') as GeoJSONSource | undefined)?.setData(networkLines(network));
    if (!fitted.current && network.length && !located.length) fit();
  }, [network, ready]);
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
        button.className = 'vehicle-marker';
        const label = document.createElement('span');
        label.className = 'vehicle-marker-label';
        const icon = document.createElement('span');
        icon.className = 'vehicle-marker-icon';
        const image = document.createElement('img');
        image.src = '/bus-marker.png';
        image.alt = '';
        image.width = 38;
        image.height = 38;
        image.draggable = false;
        icon.append(image);
        button.append(label, icon);
        button.addEventListener('click', () => select.current(vehicle.tr_id));
        marker = new maplibregl.Marker({ element: button }).setLngLat([vehicle.lon!, vehicle.lat!]).addTo(instance);
        markers.current.set(vehicle.tr_id, marker);
      }
      const element = marker.getElement();
      const risk = vehicle.freshness === 'offline' ? 'none' : vehicle.prediction?.risk_level ?? 'none';
      for (const level of ['red', 'yellow', 'green']) element.classList.toggle(level, level === risk);
      element.classList.toggle('selected', selectedId === vehicle.tr_id);
      element.querySelector('.vehicle-marker-label')!.textContent = `${vehicle.tr_id}`;
      element.setAttribute('aria-label', `ТС ${vehicle.tr_id}: ${vehicle.freshness === 'offline' ? 'нет связи с ТС' : riskLabels[risk]}`);
      element.setAttribute('aria-pressed', String(selectedId === vehicle.tr_id));
      marker.setLngLat([vehicle.lon!, vehicle.lat!]);
    }
    if (selectedId === null && autoFit.current && located.some(vehicle => !fittedIds.current.has(vehicle.tr_id))) {
      fit();
      located.forEach(vehicle => fittedIds.current.add(vehicle.tr_id));
      fitted.current = true;
    }
    const selected = located.find(vehicle => vehicle.tr_id === selectedId);
    if (selected && followSelected.current) {
      const point = instance.project([selected.lon!, selected.lat!]);
      if (point.x < 45 || point.x > instance.getContainer().clientWidth - 45
        || point.y < 55 || point.y > instance.getContainer().clientHeight - 65) fitSelected(selected);
    }
  }, [vehicles, selectedId, ready]);
  useEffect(() => {
    const vehicle = vehicles.find(item => item.tr_id === selectedId);
    const instance = map.current;
    if (!ready || !instance) return;
    instance.setFilter('selected-route', ['==', ['get', 'route_id'], selectedId ?? -1]);
    if (!vehicle || !hasPosition(vehicle)) return;
    followSelected.current = true;
    fitSelected(vehicle);
  }, [selectedId, ready, network]);
  useEffect(() => {
    if (!ready || !map.current) return;
    const track = map.current.getSource('track') as GeoJSONSource | undefined;
    const away = map.current.getSource('track-away') as GeoJSONSource | undefined;
    const stops = map.current.getSource('stops') as GeoJSONSource | undefined;
    if (!track || !away || !stops) return;
    track.setData(sections.onRoute);
    away.setData(sections.away);
    stops.setData({ type: 'FeatureCollection', features: (card?.stops ?? []).map(stop => ({ type: 'Feature', properties: {}, geometry: { type: 'Point', coordinates: [stop.lon, stop.lat] } })) });
    const selected = vehicles.find(vehicle => vehicle.tr_id === selectedId);
    if (selected && selectedTrack.length && followSelected.current) {
      const width = map.current.getContainer().clientWidth;
      const height = map.current.getContainer().clientHeight;
      if (selectedTrack.some(point => {
        const pixel = map.current!.project([point.lon, point.lat]);
        return pixel.x < 20 || pixel.x > width - 20 || pixel.y < 30 || pixel.y > height - 30;
      })) fitSelected(selected);
    }
  }, [card, ready, network, selectedId]);
  useEffect(() => {
    const instance = map.current;
    if (!ready || !instance || !instance.isStyleLoaded()) return;
    if (basemap && !instance.getSource('osm')) {
      instance.addSource('osm', { type: 'raster', tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'], tileSize: 256, attribution: '© <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noreferrer">OpenStreetMap</a> contributors', maxzoom: 19 });
      instance.addLayer({ id: 'osm', type: 'raster', source: 'osm', paint: { 'raster-saturation': -0.75, 'raster-opacity': 0.65 } }, 'network-lines');
    }
    if (instance.getLayer('osm')) instance.setLayoutProperty('osm', 'visibility', basemap ? 'visible' : 'none');
  }, [basemap, ready]);
  useEffect(() => {
    const instance = map.current;
    if (!ready || !instance) return;
    instance.setPaintProperty('network-lines', 'line-color', theme === 'dark' ? '#83aeb8' : '#789ba6');
    instance.setPaintProperty('selected-route', 'line-color', theme === 'dark' ? '#75dfc7' : '#16857a');
    instance.setPaintProperty('track-line', 'line-color', theme === 'dark' ? '#75dfc7' : '#16857a');
    instance.setPaintProperty('track-away-line', 'line-color', theme === 'dark' ? '#f2b862' : '#c79738');
    instance.setPaintProperty('stop-points', 'circle-color', theme === 'dark' ? '#172b39' : '#ffffff');
    instance.setPaintProperty('stop-points', 'circle-stroke-color', theme === 'dark' ? '#75dfc7' : '#16857a');
    if (instance.getLayer('osm')) instance.setPaintProperty('osm', 'raster-brightness-max', theme === 'dark' ? 0.55 : 1);
  }, [theme, ready, basemap]);
  return <section className="map-panel" aria-label="Карта транспорта">
    <div className="map-canvas" ref={container} />
    <div className="map-heading"><span className="eyebrow">ОПЕРАТИВНАЯ КАРТА</span><span>{located.length} из {vehicles.length} ТС с координатами</span></div>
    <div className="map-actions">
      <button className={`map-button ${basemap ? 'active' : ''}`} aria-pressed={basemap} onClick={() => { setBasemap(value => !value); setTileError(false); }} disabled={mapError}><Icon name="layers" />Подложка</button>
      <button className="map-button icon-only" title="Показать транспорт или сеть" aria-label="Показать транспорт или сеть" onClick={() => {
        autoFit.current = true; followSelected.current = true;
        if (selectedVehicle && hasPosition(selectedVehicle)) fitSelected(selectedVehicle); else fit();
      }} disabled={(!located.length && !network?.length) || mapError}><Icon name="locate" /></button>
    </div>
    {!located.length && <div className="map-empty"><div className="map-empty-symbol"><Icon name="map" size={36} /></div><h3>Город в поле зрения</h3><p>Положения транспорта появятся здесь,<br />когда поступит телеметрия.</p></div>}
    {mapError && <div className="map-warning" role="status">Карта недоступна в этом браузере. Выберите транспорт в списке.</div>}
    {tileError && basemap && <div className="map-warning" role="status">Подложка недоступна. Положения транспорта и трек продолжают обновляться.</div>}
    {networkError && <div className="network-warning" role="status">Сеть остановок пока недоступна. Повторяем запрос.</div>}
    <div className="map-footnote"><span className="coordinate-cross">⌖</span>{nearestStopKm !== null ? `До ближайшей остановки: ${nearestStopKm < 1 ? `${Math.round(nearestStopKm * 1000)} м` : `${nearestStopKm.toLocaleString('ru-RU', { maximumFractionDigits: 1 })} км`}` : basemap ? 'Подложка OpenStreetMap' : 'Без подложки'}</div>
    {(selectedRoute.length > 1 || selectedTrack.length) ? <div className="track-key">{selectedRoute.length > 1 && <><span className="planned-route" />План</>}{selectedTrack.length > 1 && <><span className="gps-track" />GPS</>}{sections.away.features.length > 0 && <><span className="gps-away" />Вне плана</>}</div> : null}
  </section>;
}
