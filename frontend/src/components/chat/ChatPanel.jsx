import React, { useState, useEffect, useRef } from 'react';
import { MessageSquare, X, Send, Trash2 } from 'lucide-react';
import { streamAgentChat, clearAgentSession } from '../../services/api';
import Plot from 'react-plotly.js';
import { Rnd } from 'react-rnd';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

const ChartRenderer = ({ ticker }) => {
    const [chartData, setChartData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [showModal, setShowModal] = useState(false);

    useEffect(() => {
        setLoading(true);
        fetch(`/api/chart/pattern/${ticker}`)
            .then(res => res.json())
            .then(data => {
                if (data.error) throw new Error(data.error);
                setChartData(data);
                setLoading(false);
            })
            .catch(err => {
                console.error("Error loading chart:", err);
                setError(err.message);
                setLoading(false);
            });
    }, [ticker]);

    if (loading) return <div style={{ padding: '10px', fontStyle: 'italic', opacity: 0.7 }}>Generating chart for {ticker}...</div>;
    if (error) return <div style={{ color: '#ff6b6b', padding: '10px' }}>Failed to load chart: {error}</div>;
    if (!chartData) return null;

    return (
        <div style={{ margin: '10px 0' }}>
            <button 
                onClick={() => setShowModal(true)}
                style={{ 
                    padding: '8px 12px', 
                    background: '#4facfe', 
                    color: '#fff', 
                    border: 'none', 
                    borderRadius: '4px', 
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px',
                    fontWeight: 'bold'
                }}
            >
                📈 View Full Interactive Chart ({ticker})
            </button>
            
            {showModal && (
                <div style={{
                    position: 'fixed',
                    top: 0, left: 0, right: 0, bottom: 0,
                    backgroundColor: 'rgba(0,0,0,0.85)',
                    zIndex: 9999,
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                    padding: '40px'
                }}>
                    <div style={{
                        width: '90vw',
                        height: '85vh',
                        backgroundColor: '#1e1e24',
                        borderRadius: '12px',
                        position: 'relative',
                        display: 'flex',
                        flexDirection: 'column',
                        boxShadow: '0 10px 40px rgba(0,0,0,0.8)'
                    }}>
                        <div style={{ padding: '15px 20px', borderBottom: '1px solid #333', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                            <h3 style={{ margin: 0, color: '#fff', fontSize: '1.2rem' }}>{ticker} Pattern Analysis</h3>
                            <button 
                                onClick={() => setShowModal(false)}
                                style={{ background: 'transparent', border: 'none', color: '#fff', cursor: 'pointer', display: 'flex', alignItems: 'center', padding: '4px' }}
                                title="Close Chart"
                            >
                                <X size={24} />
                            </button>
                        </div>
                        <div style={{ flex: 1, padding: '15px', minHeight: 0 }}>
                            <Plot
                                data={chartData.data}
                                layout={{
                                    ...chartData.layout, 
                                    autosize: true, 
                                    margin: { l: 40, r: 40, t: 30, b: 40 },
                                    dragmode: 'zoom' // Allow users to draw a box to zoom
                                }}
                                useResizeHandler={true}
                                style={{ width: "100%", height: "100%" }}
                                config={{ 
                                    responsive: true, 
                                    displayModeBar: true, 
                                    scrollZoom: true // Allow zooming in and out with mouse wheel
                                }}
                            />
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export const ChatPanel = () => {
    const [isOpen, setIsOpen] = useState(false);
    const [messages, setMessages] = useState([
        { role: 'agent', content: 'Hello! I am your AI assistant. Ask me to screen stocks, predict prices, or analyze the market.' }
    ]);
    const [input, setInput] = useState('');
    const [isStreaming, setIsStreaming] = useState(false);
    const [sessionId, setSessionId] = useState('');
    const messagesEndRef = useRef(null);

    // Initialize session ID
    useEffect(() => {
        let storedId = localStorage.getItem('agent_session_id');
        if (!storedId) {
            storedId = 'session_' + Math.random().toString(36).substr(2, 9);
            localStorage.setItem('agent_session_id', storedId);
        }
        setSessionId(storedId);
    }, []);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        if (isOpen) scrollToBottom();
    }, [messages, isOpen]);

    const handleSend = async () => {
        if (!input.trim() || isStreaming) return;
        
        const userMsg = input.trim();
        setInput('');
        setMessages(prev => [...prev, { role: 'user', content: userMsg }]);
        setIsStreaming(true);

        // Add empty agent message that will be populated via stream
        setMessages(prev => [...prev, { role: 'agent', content: '', isTyping: true }]);

        let currentResponse = '';

        const handleChunk = (data) => {
            if (data.content) {
                currentResponse += data.content;
                setMessages(prev => {
                    const newMsgs = [...prev];
                    newMsgs[newMsgs.length - 1] = { role: 'agent', content: currentResponse };
                    return newMsgs;
                });
            } else if (data.tool) {
                setMessages(prev => {
                    const newMsgs = [...prev];
                    const agentMsg = newMsgs[newMsgs.length - 1];
                    agentMsg.toolStatus = data.tool;
                    return newMsgs;
                });
            } else if (data.error) {
                setMessages(prev => {
                    const newMsgs = [...prev];
                    newMsgs[newMsgs.length - 1] = { role: 'agent', content: `Error: ${data.error}` };
                    return newMsgs;
                });
            }
        };

        const handleDone = () => {
            setIsStreaming(false);
            setMessages(prev => {
                const newMsgs = [...prev];
                delete newMsgs[newMsgs.length - 1].isTyping;
                delete newMsgs[newMsgs.length - 1].toolStatus;
                return newMsgs;
            });
        };

        const handleError = (err) => {
            console.error("Chat error:", err);
            setMessages(prev => {
                const newMsgs = [...prev];
                newMsgs[newMsgs.length - 1] = { role: 'agent', content: `Sorry, I encountered an error: ${err.message}` };
                return newMsgs;
            });
            setIsStreaming(false);
        };

        await streamAgentChat(userMsg, sessionId, handleChunk, handleDone, handleError);
    };

    const handleClear = async () => {
        if (!sessionId) return;
        try {
            await clearAgentSession(sessionId);
            // Generate a new session ID to force a fresh thread
            const newId = 'session_' + Math.random().toString(36).substr(2, 9);
            localStorage.setItem('agent_session_id', newId);
            setSessionId(newId);
            
            setMessages([{ role: 'system', content: 'Session cleared.' }]);
            setTimeout(() => {
                setMessages([{ role: 'agent', content: 'Hello! How can I help you today?' }]);
            }, 1000);
        } catch (err) {
            console.error("Error clearing session:", err);
        }
    };

    const handleKeyPress = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    return (
        <>
            {/* Floating Action Button */}
            {!isOpen && (
                <div className="chat-fab" onClick={() => setIsOpen(true)}>
                    <MessageSquare size={28} />
                </div>
            )}

            {/* Chat Panel */}
            {isOpen && (
                <Rnd
                    default={{
                        x: window.innerWidth > 450 ? window.innerWidth - 420 : 10,
                        y: Math.max(20, window.innerHeight - 620),
                        width: 400,
                        height: 600
                    }}
                    minWidth={300}
                    minHeight={400}
                    bounds="window"
                    dragHandleClassName=".chat-header"
                    style={{ zIndex: 9998, position: 'fixed' }}
                >
                    <div className="chat-panel">
                        <div className="chat-header" style={{ cursor: 'move' }}>
                            <div className="chat-title">
                                <MessageSquare size={20} />
                                AI Assistant
                            </div>
                        <div style={{ display: 'flex', gap: '10px' }}>
                            <button 
                                onClick={handleClear} 
                                title="Clear Chat"
                                style={{ background: 'transparent', border: 'none', color: 'var(--text-secondary)', cursor: 'pointer' }}
                            >
                                <Trash2 size={18} />
                            </button>
                            <button 
                                onClick={() => setIsOpen(false)} 
                                style={{ background: 'transparent', border: 'none', color: 'var(--text-secondary)', cursor: 'pointer' }}
                            >
                                <X size={20} />
                            </button>
                        </div>
                    </div>

                    <div className="chat-messages">
                        {messages.map((msg, i) => (
                            <div key={i} className={`chat-bubble ${msg.role}`}>
                                <ReactMarkdown
                                    remarkPlugins={[remarkGfm]}
                                    components={{
                                        img: ({ src, alt }) => {
                                            const match = src?.match(/^chart:pattern:(.+)$/);
                                            if (match) return <ChartRenderer ticker={match[1]} />;
                                            return <img src={src} alt={alt} style={{ maxWidth: '100%' }} />;
                                        },
                                        table: ({ children }) => (
                                            <div style={{ overflowX: 'auto', margin: '8px 0' }}>
                                                <table style={{ borderCollapse: 'collapse', width: '100%', fontSize: '0.85em' }}>{children}</table>
                                            </div>
                                        ),
                                        th: ({ children }) => (
                                            <th style={{ border: '1px solid #444', padding: '6px 10px', background: 'rgba(255,255,255,0.05)', textAlign: 'left' }}>{children}</th>
                                        ),
                                        td: ({ children }) => (
                                            <td style={{ border: '1px solid #333', padding: '6px 10px' }}>{children}</td>
                                        ),
                                        code: ({ inline, children, ...props }) => (
                                            inline
                                                ? <code style={{ background: 'rgba(255,255,255,0.1)', padding: '2px 6px', borderRadius: '4px', fontSize: '0.9em' }} {...props}>{children}</code>
                                                : <pre style={{ background: 'rgba(0,0,0,0.3)', padding: '12px', borderRadius: '8px', overflowX: 'auto', margin: '8px 0' }}><code {...props}>{children}</code></pre>
                                        ),
                                        p: ({ children }) => <p style={{ margin: '4px 0' }}>{children}</p>,
                                        ul: ({ children }) => <ul style={{ margin: '4px 0', paddingLeft: '20px' }}>{children}</ul>,
                                        ol: ({ children }) => <ol style={{ margin: '4px 0', paddingLeft: '20px' }}>{children}</ol>,
                                    }}
                                >
                                    {msg.content}
                                </ReactMarkdown>
                                {msg.isTyping && !msg.toolStatus && (
                                    <div className="chat-loading-dots mt-2" style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.85em', opacity: 0.8 }}>
                                        <div>Agent is thinking</div>
                                        <div style={{ display: 'flex', gap: '4px' }}><span /><span /><span /></div>
                                    </div>
                                )}
                                {msg.toolStatus && (
                                    <div className="chat-loading-dots mt-2" style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.85em', color: '#ffb347' }}>
                                        <div>{msg.toolStatus}</div>
                                        <div style={{ display: 'flex', gap: '4px' }}><span /><span /><span /></div>
                                    </div>
                                )}
                            </div>
                        ))}
                        <div ref={messagesEndRef} />
                    </div>

                    <div className="chat-input-area">
                        <input
                            type="text"
                            placeholder="Ask me anything..."
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            onKeyDown={handleKeyPress}
                            disabled={isStreaming}
                            autoFocus
                        />
                        <button 
                            onClick={handleSend} 
                            disabled={!input.trim() || isStreaming}
                        >
                            <Send size={18} />
                        </button>
                    </div>
                </div>
                </Rnd>
            )}
        </>
    );
};
