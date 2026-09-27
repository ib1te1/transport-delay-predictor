export function Icon({ name, size = 20 }: { name: 'bus' | 'map' | 'search' | 'arrow' | 'close' | 'layers' | 'locate' | 'activity' | 'clock' | 'info' | 'refresh' | 'sun' | 'moon'; size?: number }) {
  const paths = {
    bus: <><rect x="5" y="3" width="14" height="16" rx="3" /><path d="M5 11h14M8 7h8M8 19v2m8-2v2" /><path d="M8 15h1m6 0h1" /></>,
    map: <><path d="m3 6 6-3 6 3 6-3v15l-6 3-6-3-6 3V6Z" /><path d="M9 3v15m6-12v15" /></>,
    search: <><circle cx="10.5" cy="10.5" r="6.5" /><path d="m16 16 4 4" /></>,
    arrow: <path d="m9 5 7 7-7 7" />,
    close: <path d="m6 6 12 12M6 18 18 6" />,
    layers: <><path d="m12 3 10 5-10 5L2 8l10-5Zm-10 9 10 5 10-5M2 16l10 5 10-5" /></>,
    locate: <><circle cx="12" cy="12" r="6" /><circle cx="12" cy="12" r="2" /><path d="M12 2v4m0 12v4M2 12h4m12 0h4" /></>,
    activity: <path d="M2 12h5l3-8 4 16 3-8h5" />,
    clock: <><circle cx="12" cy="12" r="9" /><path d="M12 7v5l3 2" /></>,
    info: <><circle cx="12" cy="12" r="9" /><path d="M12 11v6m0-10v1" /></>,
    refresh: <><path d="M20 7v5h-5M4 17v-5h5" /><path d="M6 7a7 7 0 0 1 12-1l2 6M4 12l2 6a7 7 0 0 0 12-1" /></>,
    sun: <><circle cx="12" cy="12" r="4" /><path d="M12 2v2m0 16v2M4.93 4.93l1.42 1.42m11.3 11.3 1.42 1.42M2 12h2m16 0h2M4.93 19.07l1.42-1.42m11.3-11.3 1.42-1.42" /></>,
    moon: <path d="M20.5 14.2A8.5 8.5 0 0 1 9.8 3.5 8.5 8.5 0 1 0 20.5 14.2Z" />,
  };
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[name]}</svg>;
}
