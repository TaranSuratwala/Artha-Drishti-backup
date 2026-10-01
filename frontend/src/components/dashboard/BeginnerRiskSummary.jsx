import React, { useState, useEffect } from 'react';
import { Sparkles, AlertTriangle } from 'lucide-react';
import { Card } from '../ui';
import { fetchBeginnerRiskSummary } from '../../services/api';

export const BeginnerRiskSummary = ({ ticker }) => {
    const [summary, setSummary] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        const fetchSummary = async () => {
            try {
                setLoading(true);
                const data = await fetchBeginnerRiskSummary(ticker);
                
                if (data && data.status === 'success') {
                    setSummary(data.summary);
                } else {
                    setError(data?.error || 'Failed to load beginner summary');
                }
            } catch (err) {
                setError('Network error while loading summary');
            } finally {
                setLoading(false);
            }
        };

        if (ticker) {
            fetchSummary();
        }
    }, [ticker]);

    if (loading) {
        return (
            <Card className="p-4 bg-blue-50 border-l-4 border-blue-400">
                <div className="flex items-center gap-2 text-blue-700 animate-pulse">
                    <Sparkles size={18} />
                    <span className="text-sm font-semibold">AI is analyzing the risk profile for beginners...</span>
                </div>
            </Card>
        );
    }

    if (error) {
        return null; // Fail silently to not disrupt the main dashboard
    }

    if (!summary) {
        return null;
    }

    return (
        <Card className="p-5 bg-gradient-to-r from-blue-50 to-indigo-50 border-l-4 border-indigo-500 shadow-sm">
            <div className="flex items-center gap-2 mb-2 text-indigo-700">
                <Sparkles size={20} className="text-indigo-500" />
                <h3 className="font-bold text-lg">Beginner's Risk Summary</h3>
            </div>
            <p className="text-indigo-900 text-sm leading-relaxed">
                {summary}
            </p>
        </Card>
    );
};

export default BeginnerRiskSummary;
