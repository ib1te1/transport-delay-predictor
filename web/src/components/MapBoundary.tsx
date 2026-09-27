import { Component } from 'react';
import type { ReactNode } from 'react';
import { Icon } from './Icon';

interface Props { children: ReactNode }
interface State { failed: boolean }

export class MapBoundary extends Component<Props, State> {
  state: State = { failed: false };

  static getDerivedStateFromError(): State {
    return { failed: true };
  }

  render() {
    if (this.state.failed) return <section className="map-panel map-fallback" aria-label="Карта транспорта"><Icon name="map" size={36} /><h2>Карта недоступна</h2><p>Список и карточка транспорта продолжают работать. Выберите ТС в списке.</p></section>;
    return this.props.children;
  }
}
