"""
Market data Pydantic schemas
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class StockQuote(BaseModel):
    """
    Schema for stock quote data
    """
    symbol: str
    name: str
    price: float
    change: float
    changePercent: float
    volume: int
    marketCap: Optional[float] = None
    previousClose: float
    open: float
    dayHigh: float
    dayLow: float
    fiftyTwoWeekHigh: float
    fiftyTwoWeekLow: float
    peRatio: Optional[float] = None
    eps: Optional[float] = None
    beta: Optional[float] = None
    dividendYield: Optional[float] = None
    exchange: str
    currency: str
    timestamp: datetime


class PriceData(BaseModel):
    """
    Schema for historical price data point
    """
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


class HistoricalData(BaseModel):
    """
    Schema for historical price data
    """
    symbol: str
    prices: List[PriceData]
    period: str
    interval: str


class SearchResult(BaseModel):
    """
    Schema for symbol search result
    """
    symbol: str
    name: str
    type: str  # Stock, ETF, Mutual Fund, etc.
    exchange: str
    currency: str


class CompanyInfo(BaseModel):
    """
    Schema for company information
    """
    symbol: str
    name: str
    sector: Optional[str] = None
    industry: Optional[str] = None
    description: Optional[str] = None
    website: Optional[str] = None
    employees: Optional[int] = None
    headquarters: Optional[str] = None
    founded: Optional[str] = None
    ceo: Optional[str] = None


class MarketOverview(BaseModel):
    """
    Schema for market overview
    """
    marketStatus: str  # open, closed, pre-market, after-hours
    indices: List[Dict[str, Any]]
    timestamp: datetime


class TechnicalIndicator(BaseModel):
    """
    Schema for technical indicators
    """
    name: str
    value: float
    signal: str  # buy, sell, neutral
    timestamp: datetime


class NewsItem(BaseModel):
    """
    Schema for news items
    """
    title: str
    summary: Optional[str] = None
    url: str
    source: str
    publishedAt: datetime
    sentiment: Optional[str] = None  # positive, negative, neutral


class PortfolioPosition(BaseModel):
    """
    Schema for portfolio position
    """
    symbol: str
    name: str
    quantity: float
    averageCost: float
    currentPrice: float
    marketValue: float
    totalCost: float
    unrealizedPL: float
    unrealizedPLPercent: float
    realizedPL: Optional[float] = None


class Order(BaseModel):
    """
    Schema for trading order
    """
    orderId: str
    symbol: str
    orderType: str  # market, limit, stop, stop-limit
    side: str  # buy, sell
    quantity: float
    price: Optional[float] = None
    stopPrice: Optional[float] = None
    status: str  # pending, filled, cancelled, rejected
    filledQuantity: float = 0
    averageFillPrice: Optional[float] = None
    createdAt: datetime
    updatedAt: datetime


class Watchlist(BaseModel):
    """
    Schema for watchlist
    """
    id: str
    name: str
    symbols: List[str]
    createdAt: datetime
    updatedAt: datetime