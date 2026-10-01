import React, { useMemo, useState, useEffect } from 'react';
import { BarChart2, TrendingUp, ArrowDownRight, Star, ChevronLeft, ChevronsLeft, ChevronRight, ChevronsRight } from 'lucide-react';
import { useStore } from '../../store/useStore';
import { Card, StatCard, SkeletonLoader } from '../ui';
import { MarketOverview } from './MarketOverview';
import { TopMovers } from './TopMovers';
import { StockRecommendations } from './StockRecommendations';
import { StockRow } from './StockRow';

// We need the GlobalSearchInput logic. Since it's inside AuthenticatedApp.jsx and heavily tied,
// we might want to either extract it or pass it. We'll pass it as a prop for now.
export function DashboardContainer({
    user,
    loading,
    stocks,
    watchlist,
    handleOpenTicker,
    handleWatchlistAdd,
    handleWatchlistRemove,
    GlobalSearchInput,
    formatTickerForDisplay,
    activeTab,
    fetchLiveQuotes,
    subscribeToTicker,
    unsubscribeFromTicker
}) {
    const searchTerm = useStore(state => state.searchTerm);
    const setSearchTerm = useStore(state => state.setSearchTerm);
    const liveQuoteTime = useStore(state => state.liveQuoteTime);

    const [currentPage, setCurrentPage] = useState(1);
    const ITEMS_PER_PAGE = 25;

    // We use searchTerm as our debounced search (since GlobalSearchInput handles debouncing internally)
    const debouncedSearch = searchTerm;

    // Reset page on search change
    useEffect(() => { setCurrentPage(1); }, [debouncedSearch]);

    const filteredStocks = useMemo(() =>
        stocks.filter(s => (s.ticker || '').toLowerCase().includes(debouncedSearch.toLowerCase())),
        [stocks, debouncedSearch]
    );

    const totalPages = Math.max(1, Math.ceil(filteredStocks.length / ITEMS_PER_PAGE));
    const paginatedStocks = useMemo(() => {
        const start = (currentPage - 1) * ITEMS_PER_PAGE;
        return filteredStocks.slice(start, start + ITEMS_PER_PAGE);
    }, [filteredStocks, currentPage, ITEMS_PER_PAGE]);

    const filteredStocksCount = filteredStocks.length;

    // Subscribe to live quotes for currently visible paginated stocks
    useEffect(() => {
        if (activeTab !== 'dashboard' || loading) return;
        const visibleTickers = paginatedStocks.map(s => s.ticker).filter(Boolean);
        if (visibleTickers.length === 0) return;

        // Fetch immediately
        fetchLiveQuotes(visibleTickers);

        // Subscribe to sockets
        visibleTickers.forEach(t => subscribeToTicker(t));

        return () => {
            visibleTickers.forEach(t => unsubscribeFromTicker(t));
        };
    }, [activeTab, loading, paginatedStocks, fetchLiveQuotes, subscribeToTicker, unsubscribeFromTicker]);

    return (
        <div className="space-y-6 animate-fade-in dashboard-page industry-page-shell" role="tabpanel" id="panel-dashboard" aria-labelledby="tab-dashboard">
            {/* User greeting – Nielsen #6: Recognition */}
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
                        <p className="text-xs text-blue-200">Here&apos;s your market overview for today</p>
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

            {/* Search – Nielsen #7: Accelerators */}
            <Card className="p-4 industry-section-card industry-search-card">
                <div className="flex items-center gap-3">
                    <GlobalSearchInput 
                        initialValue={searchTerm} 
                        onSearchChange={setSearchTerm} 
                    />
                    <span className="text-xs text-gray-500">
                        {filteredStocksCount} results
                    </span>
                </div>
            </Card>

            {/* Stocks Table with Pagination */}
            {loading ? (
                <Card className="p-6">
                    <SkeletonLoader type="table" rows={8} />
                </Card>
            ) : (
                <Card className="overflow-hidden industry-table-shell industry-table-dashboard">
                    {/* Live data indicator */}
                    {liveQuoteTime && (
                        <div className="px-4 pt-3 flex items-center gap-2 text-xs text-gray-400">
                            <span className="live-dot" />
                            <span>Realtime stream active &middot; Last update: {liveQuoteTime}</span>
                        </div>
                    )}
                    <div className="overflow-x-auto industry-table-scroll">
                        <table className="industry-dense-table industry-dashboard-table">
                            <thead>
                                <tr>
                                    <th>Ticker</th>
                                    <th className="text-right">DB Close</th>
                                    <th className="text-right">Live Price</th>
                                    <th className="text-right">Change</th>
                                    <th className="text-right">Day Range</th>
                                    <th className="text-right">Volume</th>
                                    <th className="text-center industry-action-col">Action</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-white/5">
                                {paginatedStocks.map((stock, idx) => (
                                    <StockRow 
                                        key={stock.ticker || idx}
                                        stock={stock}
                                        idx={idx}
                                        formatTickerForDisplay={formatTickerForDisplay}
                                        handleOpenTicker={handleOpenTicker}
                                        watchlist={watchlist}
                                        handleWatchlistAdd={handleWatchlistAdd}
                                        handleWatchlistRemove={handleWatchlistRemove}
                                    />
                                ))}
                            </tbody>
                        </table>
                    </div>

                    {/* Pagination – Nielsen #3: User control & #7: Flexibility */}
                    {totalPages > 1 && (
                        <div className="pagination" role="navigation" aria-label="Stock table pagination">
                            <button
                                onClick={() => setCurrentPage(1)}
                                disabled={currentPage === 1}
                                aria-label="First page"
                            >
                                <ChevronsLeft className="w-4 h-4" />
                            </button>
                            <button
                                onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                                disabled={currentPage === 1}
                                aria-label="Previous page"
                            >
                                <ChevronLeft className="w-4 h-4" />
                            </button>

                            {/* Page numbers */}
                            {(() => {
                                const pages = [];
                                const start = Math.max(1, currentPage - 2);
                                const end = Math.min(totalPages, currentPage + 2);
                                for (let i = start; i <= end; i++) {
                                    pages.push(
                                        <button
                                            key={i}
                                            onClick={() => setCurrentPage(i)}
                                            className={i === currentPage ? 'active' : ''}
                                            aria-label={`Page ${i}`}
                                            aria-current={i === currentPage ? 'page' : undefined}
                                        >
                                            {i}
                                        </button>
                                    );
                                }
                                return pages;
                            })()}

                            <button
                                onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                                disabled={currentPage === totalPages}
                                aria-label="Next page"
                            >
                                <ChevronRight className="w-4 h-4" />
                            </button>
                            <button
                                onClick={() => setCurrentPage(totalPages)}
                                disabled={currentPage === totalPages}
                                aria-label="Last page"
                            >
                                <ChevronsRight className="w-4 h-4" />
                            </button>

                            <span className="text-xs text-gray-400 ml-3">
                                Page {currentPage} of {totalPages} ({filteredStocksCount} stocks)
                            </span>
                        </div>
                    )}
                </Card>
            )}
        </div>
    );
}
