import React, { useState, useEffect } from 'react';
import { Play, Pause, SkipForward, AlertCircle, Zap, TrendingDown, TrendingUp } from 'lucide-react';

interface DataStreamEvent {
  nodeId: string;
  timestamp: string;
  priceChange: number;
  percentChange: number;
  reason: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
}

interface RealTimeStreamProps {
  onPriceDisruption?: (event: DataStreamEvent) => void;
}

export const RealTimeStream: React.FC<RealTimeStreamProps> = ({ onPriceDisruption }) => {
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamEvents, setStreamEvents] = useState<DataStreamEvent[]>([]);
  const [currentEventIndex, setCurrentEventIndex] = useState(0);

  // Generate mock supply chain events
  const generateStreamEvents = (): DataStreamEvent[] => {
    const nodeIds = ['N0001', 'N0002', 'N0003', 'N0004', 'N0005'];
    const reasons = [
      'Commodity price spike',
      'Supply shortage',
      'Currency fluctuation',
      'Supplier delay',
      'Market demand surge',
      'Logistics cost increase',
      'Quality issue',
      'Supplier bankruptcy risk',
    ];

    const severities: ('low' | 'medium' | 'high' | 'critical')[] = ['low', 'medium', 'high', 'critical'];

    return Array.from({ length: 12 }, (_, i) => ({
      nodeId: nodeIds[Math.floor(Math.random() * nodeIds.length)],
      timestamp: new Date(Date.now() + i * 1000).toLocaleTimeString(),
      priceChange: Math.random() * 200 - 50,
      percentChange: Math.random() * 30 - 5,
      reason: reasons[Math.floor(Math.random() * reasons.length)],
      severity: severities[Math.floor(Math.random() * severities.length)],
    }));
  };

  useEffect(() => {
    setStreamEvents(generateStreamEvents());
  }, []);

  // Auto-stream events
  useEffect(() => {
    if (!isStreaming || currentEventIndex >= streamEvents.length) return;

    const timer = setTimeout(() => {
      const event = streamEvents[currentEventIndex];
      onPriceDisruption?.(event);
      setCurrentEventIndex((prev) => prev + 1);
    }, 1500);

    return () => clearTimeout(timer);
  }, [isStreaming, currentEventIndex, streamEvents, onPriceDisruption]);

  const getSeverityStyles = (severity: string) => {
    switch (severity) {
      case 'critical':
        return 'bg-red-50 border-red-200 text-red-900';
      case 'high':
        return 'bg-orange-50 border-orange-200 text-orange-900';
      case 'medium':
        return 'bg-yellow-50 border-yellow-200 text-yellow-900';
      default:
        return 'bg-green-50 border-green-200 text-green-900';
    }
  };

  const getSeverityBadge = (severity: string) => {
    const badges: Record<string, { bg: string; text: string }> = {
      critical: { bg: 'bg-red-100', text: 'text-red-700' },
      high: { bg: 'bg-orange-100', text: 'text-orange-700' },
      medium: { bg: 'bg-yellow-100', text: 'text-yellow-700' },
      low: { bg: 'bg-green-100', text: 'text-green-700' },
    };
    return badges[severity];
  };

  const currentEvent = streamEvents[currentEventIndex];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-blue-100">
            <Zap className="w-5 h-5 text-blue-600" />
          </div>
          <h3 className="text-2xl font-bold text-gray-900">Real-Time Data Stream</h3>
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => setIsStreaming(!isStreaming)}
            className="px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-700 text-white flex items-center gap-2 transition-all duration-200 font-medium"
          >
            {isStreaming ? (
              <>
                <Pause className="w-4 h-4" />
                <span className="hidden sm:inline">Pause</span>
              </>
            ) : (
              <>
                <Play className="w-4 h-4" />
                <span className="hidden sm:inline">Stream</span>
              </>
            )}
          </button>
          <button
            onClick={() => setCurrentEventIndex((prev) => Math.min(prev + 1, streamEvents.length - 1))}
            disabled={currentEventIndex >= streamEvents.length - 1}
            className="px-4 py-2 rounded-lg bg-gray-200 hover:bg-gray-300 text-gray-700 flex items-center gap-2 transition-colors disabled:opacity-50 disabled:cursor-not-allowed font-medium"
          >
            <SkipForward className="w-4 h-4" />
            <span className="hidden sm:inline">Next</span>
          </button>
        </div>
      </div>

      {currentEvent && (
        <div className={`relative overflow-hidden border rounded-lg p-6 transition-all duration-300 ${getSeverityStyles(currentEvent.severity)}`}>
          <div className="absolute top-0 right-0 w-32 h-32 opacity-10 rounded-full -mr-8 -mt-8" />
          <div className="relative z-10">
            <div className="flex items-start justify-between mb-4">
              <div>
                <p className="font-semibold text-lg">Node {currentEvent.nodeId}</p>
                <p className="text-sm opacity-75 mt-1">{currentEvent.timestamp}</p>
              </div>
              <div className={`px-3 py-1 rounded-full text-xs font-semibold uppercase tracking-wide ${getSeverityBadge(currentEvent.severity).bg} ${getSeverityBadge(currentEvent.severity).text}`}>
                {currentEvent.severity}
              </div>
            </div>
            <div className="flex items-baseline gap-2 mb-4">
              <span className="text-3xl font-bold">
                {currentEvent.priceChange > 0 ? '+' : ''}${currentEvent.priceChange.toFixed(2)}
              </span>
              <span className="text-lg font-semibold opacity-75">
                ({currentEvent.percentChange > 0 ? '+' : ''}{currentEvent.percentChange.toFixed(1)}%)
              </span>
            </div>
            <p className="text-sm font-medium flex items-center gap-2">
              <AlertCircle className="w-4 h-4" />
              {currentEvent.reason}
            </p>
          </div>
        </div>
      )}

      {/* Event timeline */}
      <div className="bg-white border border-gray-200 rounded-lg p-4">
        <p className="text-sm font-semibold text-gray-900 mb-4">Event Timeline</p>
        <div className="space-y-2 max-h-64 overflow-y-auto">
          {streamEvents.map((event, idx) => (
            <div
              key={idx}
              className={`p-3 rounded-lg border transition-all duration-200 ${
                idx === currentEventIndex
                  ? 'bg-blue-50 border-blue-300 shadow-md'
                  : idx < currentEventIndex
                    ? 'bg-gray-50 border-gray-200'
                    : 'bg-white border-gray-200'
              }`}
            >
              <div className="flex items-center justify-between mb-2">
                <span className="font-semibold text-sm text-gray-900">{event.nodeId}</span>
                <div className="flex items-center gap-1">
                  {event.priceChange > 0 ? (
                    <TrendingUp className="w-4 h-4 text-red-500" />
                  ) : (
                    <TrendingDown className="w-4 h-4 text-green-500" />
                  )}
                  <span className={`font-mono font-semibold ${event.priceChange > 0 ? 'text-red-600' : 'text-green-600'}`}>
                    {event.priceChange > 0 ? '+' : ''}${event.priceChange.toFixed(0)}
                  </span>
                </div>
              </div>
              <p className="text-xs text-gray-600">{event.reason}</p>
            </div>
          ))}
        </div>
      </div>

      <div className="flex items-center justify-between text-xs text-gray-600">
        <span>Event {currentEventIndex + 1} of {streamEvents.length}</span>
        <span>{Math.round((((currentEventIndex + 1) / streamEvents.length) * 100))}% Complete</span>
      </div>
    </div>
  );
};
