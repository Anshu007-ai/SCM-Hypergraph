import React from 'react';

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: React.ErrorInfo | null;
}

export class ErrorBoundary extends React.Component<
  { children: React.ReactNode },
  State
> {
  constructor(props: { children: React.ReactNode }) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  static getDerivedStateFromError(error: Error): Partial<State> {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    this.setState({ errorInfo });
    console.error('ErrorBoundary caught:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div
          style={{
            padding: 40,
            fontFamily: 'monospace',
            color: '#ff6b6b',
            backgroundColor: '#1a1a2e',
            minHeight: '100vh',
          }}
        >
          <h1 style={{ fontSize: 24, marginBottom: 16 }}>
            Something went wrong
          </h1>
          <pre style={{ whiteSpace: 'pre-wrap', fontSize: 14, color: '#fff' }}>
            {this.state.error?.toString()}
          </pre>
          <h2 style={{ fontSize: 18, marginTop: 24, marginBottom: 8 }}>
            Component Stack
          </h2>
          <pre style={{ whiteSpace: 'pre-wrap', fontSize: 12, color: '#aaa' }}>
            {this.state.errorInfo?.componentStack}
          </pre>
        </div>
      );
    }
    return this.props.children;
  }
}
