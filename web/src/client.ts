import { applyMessage, reconcile } from './protocol.ts';
import type { Message, Snapshot } from './types.ts';

export type Connection = 'connecting' | 'syncing' | 'live' | 'quiet' | 'reconnecting';
export const API_URL = (import.meta.env.VITE_API_URL || `${location.protocol}//${location.hostname}:8000`).replace(/\/$/, '');
export const apiPath = (path: string) => `${API_URL}${path}`;
export function socketUrl(base: string): string {
  const url = new URL(`${base}/ws`);
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
  return url.toString();
}

export async function getJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(apiPath(path), { signal: signal ? AbortSignal.any([signal, AbortSignal.timeout(10000)]) : AbortSignal.timeout(10000) });
  if (!response.ok) throw new Error(response.status === 404 ? 'Транспорт больше не доступен в текущем прогоне.' : response.status === 503 ? 'Сервис ещё готовит данные. Повторите через несколько секунд.' : `Не удалось получить данные (${response.status}).`);
  return response.json() as Promise<T>;
}

interface Callbacks {
  state: (snapshot: Snapshot) => void;
  connection: (connection: Connection) => void;
  message: (message: Message) => void;
  reset: () => void;
}

export function connectDashboard(callbacks: Callbacks): () => void {
  let stopped = false;
  let socket: WebSocket | null = null;
  let request: AbortController | null = null;
  let retry: ReturnType<typeof setTimeout>;
  let attempt = 0;
  let generation = 0;
  let lastMessage = Date.now();
  let current: Snapshot | null = null;
  let buffered: Message[] = [];
  let syncing = false;
  let status: Connection = 'connecting';
  const setStatus = (value: Connection) => {
    if (status !== value) { status = value; callbacks.connection(value); }
  };
  const sync = async () => {
    if (syncing || stopped) return;
    syncing = true;
    setStatus('syncing');
    const ownGeneration = generation;
    request = new AbortController();
    try {
      const snapshot = await getJson<Snapshot>('/api/state', request.signal);
      if (stopped || ownGeneration !== generation) return;
      current = reconcile(snapshot, buffered);
      buffered = [];
      syncing = false;
      attempt = 0;
      callbacks.state(current);
      callbacks.reset();
      setStatus(Date.now() - lastMessage > 10000 ? 'quiet' : 'live');
    } catch {
      if (stopped || ownGeneration !== generation) return;
      syncing = false;
      socket?.close();
    }
  };
  const open = () => {
    if (stopped) return;
    generation++;
    current = null;
    buffered = [];
    syncing = false;
    setStatus(attempt ? 'reconnecting' : 'connecting');
    const active = new WebSocket(socketUrl(API_URL));
    socket = active;
    const handshakeTimeout = setTimeout(() => active.close(), 10000);
    active.onopen = () => { clearTimeout(handshakeTimeout); lastMessage = Date.now(); void sync(); };
    active.onmessage = event => {
      if (stopped || socket !== active) return;
      try {
        const message = JSON.parse(event.data as string) as Message;
        if (!['clock', 'vehicles', 'prediction', 'alert'].includes(message.type) || !Number.isSafeInteger(message.data.seq)) throw new Error('Invalid stream message');
        lastMessage = Date.now();
        callbacks.message(message);
        if (syncing || !current) {
          buffered.push(message);
          if (buffered.length > 5000) active.close();
          return;
        }
        try {
          current = applyMessage(current, message);
          callbacks.state(current);
          setStatus('live');
        } catch {
          buffered = [message];
          void sync();
        }
      } catch { active.close(); }
    };
    active.onerror = () => active.close();
    active.onclose = () => {
      clearTimeout(handshakeTimeout);
      if (stopped || socket !== active) return;
      generation++;
      request?.abort();
      syncing = false;
      setStatus('reconnecting');
      retry = setTimeout(open, Math.min(1000 * 2 ** attempt++, 15000));
    };
  };
  const watchdog = setInterval(() => {
    if (socket?.readyState === WebSocket.OPEN && !syncing && Date.now() - lastMessage > 10000) setStatus('quiet');
  }, 1000);
  open();
  return () => { stopped = true; generation++; clearTimeout(retry); clearInterval(watchdog); request?.abort(); socket?.close(); };
}
