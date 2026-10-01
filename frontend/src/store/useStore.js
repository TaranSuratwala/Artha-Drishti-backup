import { create } from 'zustand';

export const useStore = create((set) => ({
  liveQuotes: {},
  setLiveQuotes: (quotes) => set({ liveQuotes: quotes }),
  updateLiveQuotesBatch: (newQuotes) => set((state) => ({
    liveQuotes: { ...state.liveQuotes, ...newQuotes }
  })),
  updateLiveQuote: (ticker, data) => set((state) => ({
    liveQuotes: {
      ...state.liveQuotes,
      [ticker]: { ...state.liveQuotes[ticker], ...data }
    }
  })),

  liveQuoteTime: null,
  setLiveQuoteTime: (time) => set({ liveQuoteTime: time }),

  tickerQuote: null,
  setTickerQuote: (quote) => set({ tickerQuote: quote }),

  searchTerm: '',
  setSearchTerm: (term) => set({ searchTerm: term }),

  stocks: [],
  setStocks: (stocks) => set({ stocks }),

  strategies: [],
  setStrategies: (strategies) => set({ strategies }),
}));
