import React from 'react';
import { Search, TrendingUp, BarChart2, ArrowDownRight, Star, X } from 'lucide-react';
import { Card, StatCard, SkeletonLoader } from '../components/ui';
import { MarketOverview, TopMovers, StockRecommendations } from '../components/dashboard';

export default function DashboardPage({
    user,
    loading,
    stocks,
    watchlist,
    filteredStocks,
    searchTerm,
    setSearchTerm,
    handleOpenTicker,
    currentPage,
    setCurrentPage,
    ITEMS_PER_PAGE,
    formatDate,
    formatNumber,
    formatCurrency,
    formatPercent
}) {
    // Pagination logic
    const indexOfLastItem = currentPage * ITEMS_PER_PAGE;
    const indexOfFirstItem = indexOfLastItem - ITEMS_PER_PAGE;
    const currentItems = filteredStocks.slice(indexOfFirstItem, indexOfLastItem);
    const totalPages = Math.ceil(filteredStocks.length / ITEMS_PER_PAGE);

    return (
        <div className="space-y-6 animate-fade-in dashboard-page industry-page-shell" role="tabpanel" id="panel-dashboard" aria-labelledby="tab-dashboard">
            {/* User greeting */}
            {user && (
                <div className="flex items-center gap-3 mb-2">
                    <div className="w-10 h-10 rounded-full bg-gradient-to-br from-blue-500 to-purple-500 flex items-center justify-center text-white font-bold text-lg">
                        {user.avatar_url ? (
                            <img src={user.avatar_url} alt="" className="w-full h-full rounded-full object-cover" />
                        ) : (
                            (user.username || 'U')[0].toUpperCase()
                        )}
                    </div>
                    <div>
                        <h2 className="text-lg font-bold text-white">Welcome back, {user.username}!</h2>
                        <p className="text-xs text-blue-200">Here's your market overview for today</p>
                    </div>
                </div>
            )}

            {/* Stats Grid */}
            {loading ? (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    {[...Array(4)].map((_, i) => <SkeletonLoader key={i} type="card" />)}
                </div>
            ) : (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <StatCard icon={BarChart2} label="Total Stocks" value={stocks.length} color="blue" />
                    <StatCard icon={TrendingUp} label="Gainers" value={stocks.filter(s => s.close > s.open).length} color="green" />
                    <StatCard icon={ArrowDownRight} label="Losers" value={stocks.filter(s => s.close < s.open).length} color="red" />
                    <StatCard icon={Star} label="Watchlist" value={watchlist.length} color="yellow" />
                </div>
            )}

            {/* Top Gainers/Losers - Live Data */}
            <TopMovers onTickerClick={handleOpenTicker} />

            {/* AI Stock Recommendations */}
            <StockRecommendations onTickerClick={handleOpenTicker} />

            {/* Market Overview Widget */}
            <MarketOverview />

            {/* Search */}
            <Card className="p-4 industry-section-card industry-search-card">
                <div className="flex items-center gap-3">
                    <div className="p-2.5 bg-blue-500/20 rounded-xl">
                        <Search className="w-5 h-5 text-blue-300" />
                    </div>
                    <input
                        type="text"
                        data-search-input
                        placeholder="Search by ticker (e.g., RELIANCE, TCS)... Ctrl+K"
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                        className="flex-1 bg-transparent outline-none text-white placeholder-blue-200/50 font-medium"
                        aria-label="Search stocks by ticker"
                    />
                    {searchTerm && (
                        <button
                            onClick={() => setSearchTerm('')}
                            className="p-1 rounded-lg bg-white/10 hover:bg-white/20 text-gray-400 transition"
                            aria-label="Clear search"
                        >
                            <X className="w-4 h-4" />
                        </button>
                    )}
                    <span className="text-xs text-gray-500">
                        {filteredStocks.length} results
                    </span>
                </div>
            </Card>

            {/* Stocks Table */}
            <Card className="overflow-hidden industry-section-card">
                <div className="p-5 border-b border-white/5 flex justify-between items-center bg-slate-900/50">
                    <h3 className="text-sm font-bold text-white flex items-center gap-2">
                        <BarChart2 className="w-4 h-4 text-blue-400" /> Market Data
                    </h3>
                </div>
                <div className="overflow-x-auto">
                    <table className="w-full text-left border-collapse" role="grid">
                        <thead>
                            <tr className="border-b border-white/5 bg-slate-800/30 text-xs text-slate-400 uppercase tracking-wider">
                                <th className="p-4 font-semibold">Ticker</th>
                                <th className="p-4 font-semibold text-right">Date</th>
                                <th className="p-4 font-semibold text-right">Close</th>
                                <th className="p-4 font-semibold text-right">Change</th>
                                <th className="p-4 font-semibold text-right">Volume</th>
                            </tr>
                        </thead>
                        <tbody>
                            {loading ? (
                                [...Array(5)].map((_, i) => (
                                    <tr key={`loading-${i}`} className="border-b border-white/5">
                                        <td className="p-4"><SkeletonLoader className="h-4 w-20" /></td>
                                        <td className="p-4 text-right"><SkeletonLoader className="h-4 w-24 ml-auto" /></td>
                                        <td className="p-4 text-right"><SkeletonLoader className="h-4 w-16 ml-auto" /></td>
                                        <td className="p-4 text-right"><SkeletonLoader className="h-4 w-16 ml-auto" /></td>
                                        <td className="p-4 text-right"><SkeletonLoader className="h-4 w-20 ml-auto" /></td>
                                    </tr>
                                ))
                            ) : currentItems.length > 0 ? (
                                currentItems.map((stock) => {
                                    const diff = stock.close - stock.open;
                                    const pct = (diff / stock.open) * 100;
                                    const isUp = diff >= 0;
                                    return (
                                        <tr
                                            key={`${stock.ticker}-${stock.date}`}
                                            className="border-b border-white/5 hover:bg-white/5 cursor-pointer transition group"
                                            onClick={() => handleOpenTicker(stock.ticker)}
                                            tabIndex={0}
                                            role="row"
                                            onKeyDown={(e) => { if (e.key === 'Enter') handleOpenTicker(stock.ticker) }}
                                        >
                                            <td className="p-4">
                                                <div className="font-bold text-white group-hover:text-blue-400 transition">{stock.ticker}</div>
                                            </td>
                                            <td className="p-4 text-right text-gray-400 text-sm">
                                                {formatDate(stock.date)}
                                            </td>
                                            <td className="p-4 text-right font-medium text-white">
                                                {formatCurrency(stock.close)}
                                            </td>
                                            <td className={`p-4 text-right font-medium ${isUp ? 'text-green-400' : 'text-red-400'}`}>
                                                <div className="flex items-center justify-end gap-1">
                                                    {isUp ? <TrendingUp className="w-3 h-3" /> : <ArrowDownRight className="w-3 h-3" />}
                                                    {formatCurrency(Math.abs(diff))} ({formatPercent(Math.abs(pct))})
                                                </div>
                                            </td>
                                            <td className="p-4 text-right text-gray-400 text-sm">
                                                {formatNumber(stock.volume)}
                                            </td>
                                        </tr>
                                    );
                                })
                            ) : (
                                <tr>
                                    <td colSpan="5" className="p-8 text-center text-gray-500">
                                        No stocks found matching your search.
                                    </td>
                                </tr>
                            )}
                        </tbody>
                    </table>
                </div>

                {/* Pagination Controls */}
                {!loading && totalPages > 1 && (
                    <div className="p-4 border-t border-white/5 flex items-center justify-between bg-slate-900/30">
                        <span className="text-xs text-gray-400 font-medium">
                            Showing {indexOfFirstItem + 1} to Math.min(indexOfLastItem, filteredStocks.length) of {filteredStocks.length}
                        </span>
                        <div className="flex items-center gap-2">
                            <button
                                onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                                disabled={currentPage === 1}
                                className="px-3 py-1.5 rounded-lg bg-slate-800 text-white text-xs font-semibold hover:bg-slate-700 disabled:opacity-50 disabled:cursor-not-allowed transition"
                                aria-label="Previous page"
                            >
                                Prev
                            </button>
                            <span className="text-xs text-white font-bold px-2">
                                {currentPage} / {totalPages}
                            </span>
                            <button
                                onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                                disabled={currentPage === totalPages}
                                className="px-3 py-1.5 rounded-lg bg-slate-800 text-white text-xs font-semibold hover:bg-slate-700 disabled:opacity-50 disabled:cursor-not-allowed transition"
                                aria-label="Next page"
                            >
                                Next
                            </button>
                        </div>
                    </div>
                )}
            </Card>
        </div>
    );
}
