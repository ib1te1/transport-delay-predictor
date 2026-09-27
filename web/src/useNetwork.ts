import { useEffect, useMemo, useState } from 'react';
import { getJson } from './client';
import type { NetworkStop } from './types';

// Planned network stops, loaded once and retried until the API has them.
export function useNetwork() {
  const [network, setNetwork] = useState<NetworkStop[] | null>(null);
  const [networkError, setNetworkError] = useState(false);
  useEffect(() => {
    const controller = new AbortController();
    let retry: ReturnType<typeof setTimeout>;
    const load = async () => {
      try {
        const stops = await getJson<NetworkStop[]>('/api/stops', controller.signal);
        if (!controller.signal.aborted) { setNetwork(stops); setNetworkError(false); }
      } catch {
        if (!controller.signal.aborted) {
          setNetworkError(true);
          retry = setTimeout(() => void load(), 15000);
        }
      }
    };
    void load();
    return () => { controller.abort(); clearTimeout(retry); };
  }, []);
  const stops = useMemo(() => new Map((network ?? []).map(stop => [stop.stop_id, stop])), [network]);
  return { network, networkError, stops };
}
