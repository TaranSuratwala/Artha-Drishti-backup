import React from 'react';
import { ArrowUpRight, ArrowDownRight, Star, StarOff } from 'lucide-react';
import { useStore } from '../../store/useStore';

export function StockRow({ stock, idx, formatTickerForDisplay, handleOpenTicker, watchlist, handleWatchlistAdd, handleWatchlistRemove }) {
    const q = useStore(state => state.liveQuotes[stock.ticker]);
    const livePrice = q?.price;
    const changePct = q?.change_pct;
    const isUp = changePct > 0;
    const isDown = changePct < 0;

    return (
        <tr>
            <td
                className="font-bold text-blue-400 cursor-pointer hover:text-blue-300 focus:text-blue-300"
                onClick={() => handleOpenTicker(stock.ticker)}
                tabIndex={0}
                onKeyDown={(e) => e.key === 'Enter' && handleOpenTicker(stock.ticker)}
                role="button"
            >
                {formatTickerForDisplay(stock.ticker)}
            </td>
            <td className="text-right text-sm text-gray-400">₹{(stock.close || 0).toFixed(2)}</td>
            <td className="text-right font-bold">
                {livePrice ? (
                    <span className={isUp ? 'text-green-400' : isDown ? 'text-red-400' : 'text-white'}>
                        ₹{livePrice.toFixed(2)}
                    </span>
                ) : (
                    <span className="text-gray-500 text-xs">—</span>
                )}
            </td>
            <td className="text-right text-sm">
                {changePct != null ? (
                    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-bold ${isUp ? 'bg-green-500/20 text-green-400' : isDown ? 'bg-red-500/20 text-red-400' : 'bg-gray-500/20 text-gray-400'}`}>
                        {isUp ? <ArrowUpRight className="w-3 h-3" /> : isDown ? <ArrowDownRight className="w-3 h-3" /> : null}
                        {changePct > 0 ? '+' : ''}{changePct.toFixed(2)}%
                    </span>
                ) : (
                    <span className="text-gray-500 text-xs">—</span>
                )}
            </td>
            <td className="text-right text-xs text-gray-300">
                {q?.day_low != null && q?.day_high != null ? (
                    <span>₹{q.day_low.toFixed(0)} – ₹{q.day_high.toFixed(0)}</span>
                ) : (
                    <span className="text-gray-500">—</span>
                )}
            </td>
            <td className="text-right text-sm text-gray-300">
                {(q?.volume || stock.volume || 0).toLocaleString()}
            </td>
            <td className="text-center industry-action-cell">
                <button
                    onClick={() => watchlist.includes(stock.ticker) ? handleWatchlistRemove(stock.ticker) : handleWatchlistAdd(stock.ticker)}
                    className={`industry-action-btn industry-watchlist-toggle p-1.5 rounded-lg transition ${watchlist.includes(stock.ticker) ? 'bg-yellow-500/20 text-yellow-400' : 'bg-white/10 text-gray-400 hover:text-yellow-400'}`}
                    aria-label={watchlist.includes(stock.ticker) ? `Remove ${formatTickerForDisplay(stock.ticker)} from watchlist` : `Add ${formatTickerForDisplay(stock.ticker)} to watchlist`}
                >
                    {watchlist.includes(stock.ticker) ? <Star className="w-4 h-4 fill-current" /> : <StarOff className="w-4 h-4" />}
                </button>
            </td>
        </tr>
    );
}
