import { useEffect, useRef, useState } from 'react';
import { connectDashboard, getJson } from './client';
import type { Connection } from './client';
import { mergeHistory } from './protocol';
import type { Message, Snapshot, VehicleCard } from './types';

export function useDashboard(selectedId: number | null) {
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [connection, setConnection] = useState<Connection>('connecting');
  const [revision, setRevision] = useState(0);
  const [card, setCard] = useState<VehicleCard | null>(null);
  const [cardLoading, setCardLoading] = useState(false);
  const [cardError, setCardError] = useState('');
  const messages = useRef<Message[]>([]);
  const selectedRef = useRef(selectedId);
  selectedRef.current = selectedId;
  useEffect(() => connectDashboard({
    state: setSnapshot, connection: setConnection,
    reset: () => { messages.current = []; setRevision(value => value + 1); },
    message: message => {
      if (message.type !== 'prediction' || message.data.tr_id !== selectedRef.current) return;
      messages.current = [...messages.current.slice(-999), message];
      setCard(value => value ? mergeHistory(value, message.data.tr_id, [message]) : value);
    },
  }), []);
  useEffect(() => { setCard(null); messages.current = []; }, [selectedId]);
  useEffect(() => {
    if (selectedId === null) { setCardError(''); setCardLoading(false); return; }
    const controller = new AbortController();
    let busy = false;
    const refresh = async () => {
      if (busy) return;
      busy = true;
      setCardLoading(true);
      try {
        const value = await getJson<VehicleCard>(`/api/vehicles/${selectedId}`, controller.signal);
        if (controller.signal.aborted) return;
        setCard(mergeHistory(value, selectedId, messages.current));
        setCardError('');
      } catch (error) {
        if (!controller.signal.aborted) setCardError(error instanceof Error ? error.message : 'Не удалось загрузить карточку.');
      } finally {
        busy = false;
        if (!controller.signal.aborted) setCardLoading(false);
      }
    };
    void refresh();
    const timer = setInterval(() => void refresh(), 15000);
    return () => { controller.abort(); clearInterval(timer); };
  }, [selectedId, revision]);
  return { snapshot, connection, card, cardLoading, cardError, refreshCard: () => setRevision(value => value + 1) };
}
