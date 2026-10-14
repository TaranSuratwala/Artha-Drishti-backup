import React, { useEffect, useState } from 'react';
import { Card, SkeletonLoader } from '../ui';
import { TrendingUp, TrendingDown, Clock, ShieldCheck } from 'lucide-react';
import { fetchPaperOrders } from '../../services/api';

export default function PaperTradingOrders() {
    const [orders, setOrders] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        let isMounted = true;
        const getOrders = async () => {
            try {
                setLoading(true);
                const data = await fetchPaperOrders();
                if (isMounted) {
                    setOrders(data || []);
                }
            } catch (err) {
                if (isMounted) setError('Failed to load paper trading orders');
            } finally {
                if (isMounted) setLoading(false);
            }
        };
        getOrders();
        // Optional polling
        const interval = setInterval(getOrders, 30000);
        return () => {
            isMounted = false;
            clearInterval(interval);
        };
    }, []);

    if (error) {
        return (
            <Card className="overflow-hidden industry-section-card">
                <div className="p-5 border-b border-white/5 bg-slate-900/50">
                    <h3 className="text-sm font-bold text-white flex items-center gap-2">
                        <ShieldCheck className="w-4 h-4 text-emerald-400" /> Model Suggestions (Paper Trading)
                    </h3>
                </div>
                <div className="p-8 text-center text-red-400">
                    {error}
                </div>
            </Card>
        );
    }

    return (
        <Card className="overflow-hidden industry-section-card">
            <div className="p-5 border-b border-white/5 flex justify-between items-center bg-slate-900/50">
                <h3 className="text-sm font-bold text-white flex items-center gap-2">
                    <ShieldCheck className="w-4 h-4 text-emerald-400" /> Model Suggestions (Paper Trading)
                </h3>
            </div>
            <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse" role="grid">
                    <thead>
                        <tr className="border-b border-white/5 bg-slate-800/30 text-xs text-slate-400 uppercase tracking-wider">
                            <th className="p-4 font-semibold">Ticker</th>
                            <th className="p-4 font-semibold">Status</th>
                            <th className="p-4 font-semibold text-right">Side</th>
                            <th className="p-4 font-semibold text-right">Qty</th>
                            <th className="p-4 font-semibold text-right">Price</th>
                            <th className="p-4 font-semibold text-right">Time</th>
                        </tr>
                    </thead>
                    <tbody>
                        {loading ? (
                            [...Array(3)].map((_, i) => (
                                <tr key={`loading-order-${i}`} className="border-b border-white/5">
                                    <td className="p-4"><SkeletonLoader className="h-4 w-16" /></td>
                                    <td className="p-4"><SkeletonLoader className="h-4 w-20" /></td>
                                    <td className="p-4 text-right"><SkeletonLoader className="h-4 w-12 ml-auto" /></td>
                                    <td className="p-4 text-right"><SkeletonLoader className="h-4 w-10 ml-auto" /></td>
                                    <td className="p-4 text-right"><SkeletonLoader className="h-4 w-16 ml-auto" /></td>
                                    <td className="p-4 text-right"><SkeletonLoader className="h-4 w-24 ml-auto" /></td>
                                </tr>
                            ))
                        ) : orders.length > 0 ? (
                            orders.map((order, idx) => (
                                <tr key={`order-${idx}`} className="border-b border-white/5 hover:bg-white/5 transition">
                                    <td className="p-4">
                                        <div className="font-bold text-white">{order.ticker || order.symbol}</div>
                                    </td>
                                    <td className="p-4 text-sm">
                                        <span className={`px-2 py-1 rounded text-xs font-semibold ${
                                            order.status === 'FILLED' ? 'bg-green-500/20 text-green-400' :
                                            order.status === 'PENDING' ? 'bg-yellow-500/20 text-yellow-400' :
                                            order.status === 'REJECTED' ? 'bg-red-500/20 text-red-400' :
                                            'bg-slate-500/20 text-slate-400'
                                        }`}>
                                            {order.status}
                                        </span>
                                    </td>
                                    <td className="p-4 text-right font-medium">
                                        <div className={`flex items-center justify-end gap-1 ${order.side === 'BUY' ? 'text-green-400' : 'text-red-400'}`}>
                                            {order.side === 'BUY' ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
                                            {order.side}
                                        </div>
                                    </td>
                                    <td className="p-4 text-right text-white font-medium">
                                        {order.quantity}
                                    </td>
                                    <td className="p-4 text-right text-gray-300">
                                        ₹{order.price?.toLocaleString() || order.average_price?.toLocaleString() || '-'}
                                    </td>
                                    <td className="p-4 text-right text-gray-400 text-sm flex items-center justify-end gap-1">
                                        <Clock className="w-3 h-3"/>
                                        {order.placed_at ? new Date(order.placed_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '-'}
                                    </td>
                                </tr>
                            ))
                        ) : (
                            <tr>
                                <td colSpan="6" className="p-8 text-center text-gray-500">
                                    No model suggestions or orders today.
                                </td>
                            </tr>
                        )}
                    </tbody>
                </table>
            </div>
        </Card>
    );
}
