import { useEffect, useRef, useState, useCallback } from 'react';
import { io } from 'socket.io-client';
import { useStore } from '../store/useStore';

const SOCKET_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';

export function useSocket() {
    const [isConnected, setIsConnected] = useState(false);
    const socketRef = useRef(null);
    // Zustand dispatcher
    const updateLiveQuote = useStore(state => state.updateLiveQuote);
    const setLiveQuoteTime = useStore(state => state.setLiveQuoteTime);
    const setTickerQuote = useStore(state => state.setTickerQuote);

    useEffect(() => {
        // Initialize socket connection
        socketRef.current = io(SOCKET_URL, {
            transports: ['websocket', 'polling'], // Fallback to polling if websocket fails
            reconnectionAttempts: 5,
            reconnectionDelay: 1000,
        });

        socketRef.current.on('connect', () => {
            console.log('🟢 Connected to WebSocket server');
            setIsConnected(true);
        });

        socketRef.current.on('disconnect', () => {
            console.log('🔴 Disconnected from WebSocket server');
            setIsConnected(false);
        });

        socketRef.current.on('connect_error', (err) => {
            console.error('WebSocket connection error:', err);
        });

        // Directly dispatch to Zustand store
        socketRef.current.on('quote_update', (data) => {
            if (data && data.ticker) {
                updateLiveQuote(data.ticker, data);
                setLiveQuoteTime(new Date().toLocaleTimeString());
                
                // For tickerQuote, AuthenticatedApp will handle selectedTicker check,
                // but if we want to move it completely we need the selectedTicker context.
                // For now, let AuthenticatedApp's onQuoteUpdate listener handle it, OR
                // we can export a way to get state.
            }
        });

        return () => {
            if (socketRef.current) {
                socketRef.current.disconnect();
            }
        };
    }, [updateLiveQuote, setLiveQuoteTime]);

    const subscribeToTicker = useCallback((ticker) => {
        if (socketRef.current && isConnected) {
            socketRef.current.emit('subscribe', { ticker });
        }
    }, [isConnected]);

    const unsubscribeFromTicker = useCallback((ticker) => {
        if (socketRef.current && isConnected) {
            socketRef.current.emit('unsubscribe', { ticker });
        }
    }, [isConnected]);

    const onQuoteUpdate = useCallback((callback) => {
        if (socketRef.current) {
            socketRef.current.on('quote_update', callback);
        }
    }, []);

    const offQuoteUpdate = useCallback((callback) => {
        if (socketRef.current) {
            socketRef.current.off('quote_update', callback);
        }
    }, []);

    return {
        isConnected,
        subscribeToTicker,
        unsubscribeFromTicker,
        onQuoteUpdate,
        offQuoteUpdate
    };
}
