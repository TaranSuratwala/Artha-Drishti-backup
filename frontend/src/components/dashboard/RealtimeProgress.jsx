import React, { useEffect, useState } from 'react';
import { io } from 'socket.io-client';

const SOCKET_SERVER_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';

const RealtimeProgress = () => {
  const [tasks, setTasks] = useState([]);

  useEffect(() => {
    // Connect to the backend SocketIO server
    const socket = io(SOCKET_SERVER_URL, {
      transports: ['websocket'],
    });

    socket.on('connect', () => {
      console.log('Connected to backend realtime server');
    });

    socket.on('task_progress', (data) => {
      // data format: { task: 'screen', strategy: 'momentum', status: 'started' }
      // or { task: 'sentiment', ticker: 'RELIANCE', status: 'completed' }
      
      const taskId = `${data.task}-${data.strategy || data.ticker}`;
      
      setTasks(prevTasks => {
        if (data.status === 'completed' || data.status === 'failed') {
          // Remove task after 3 seconds
          setTimeout(() => {
            setTasks(current => current.filter(t => t.id !== taskId));
          }, 3000);
          
          return prevTasks.map(t => 
            t.id === taskId ? { ...t, status: data.status, timestamp: Date.now() } : t
          );
        } else {
          // Add or update task
          const existing = prevTasks.find(t => t.id === taskId);
          if (existing) {
            return prevTasks.map(t => 
              t.id === taskId ? { ...t, ...data, timestamp: Date.now() } : t
            );
          } else {
            return [...prevTasks, { id: taskId, ...data, timestamp: Date.now() }];
          }
        }
      });
    });

    return () => {
      socket.disconnect();
    };
  }, []);

  if (tasks.length === 0) return null;

  return (
    <div style={{
      position: 'fixed',
      bottom: '20px',
      right: '20px',
      zIndex: 9999,
      display: 'flex',
      flexDirection: 'column',
      gap: '10px'
    }}>
      {tasks.map(task => (
        <div 
          key={task.id}
          style={{
            background: 'rgba(30, 41, 59, 0.95)',
            border: `1px solid ${task.status === 'completed' ? '#10b981' : task.status === 'failed' ? '#ef4444' : '#3b82f6'}`,
            borderRadius: '8px',
            padding: '12px 16px',
            color: '#f8fafc',
            boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
            minWidth: '250px',
            transition: 'all 0.3s ease',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            backdropFilter: 'blur(4px)'
          }}
        >
          <div>
            <div style={{ fontSize: '12px', color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '4px' }}>
              {task.task}
            </div>
            <div style={{ fontWeight: 500, fontSize: '14px' }}>
              {task.strategy || task.ticker}
            </div>
          </div>
          
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ fontSize: '13px', color: task.status === 'completed' ? '#10b981' : task.status === 'failed' ? '#ef4444' : '#60a5fa' }}>
              {task.status === 'started' ? 'Running...' : task.status === 'completed' ? 'Done' : 'Error'}
            </span>
            {task.status === 'started' && (
              <div style={{
                width: '12px',
                height: '12px',
                borderRadius: '50%',
                border: '2px solid #3b82f6',
                borderTopColor: 'transparent',
                animation: 'spin 1s linear infinite'
              }} />
            )}
          </div>
        </div>
      ))}
      <style>{`
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
};

export default RealtimeProgress;
