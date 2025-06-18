"""
Market data API endpoints
"""
from typing import List, Optional, Dict, Any
from datetime import datetime, date
from fastapi import APIRouter, HTTPException, Depends, Query, Path

from app.core.security import get_current_active_user
from app.services.market_service import MarketService
from app.schemas.market import (
    StockQuote, 
    HistoricalData, 
    MarketOverview,
    SearchResult,
    CompanyInfo
)


router = APIRouter()
market_service = MarketService()


@router.get("/quote/{symbol}", response_model=StockQuote)
async def get_stock_quote(
    symbol: str = Path(..., description="Stock symbol (e.g., AAPL, GOOGL)"),
    current_user: Dict = Depends(get_current_active_user)
) -> StockQuote:
    """
    Get real-time stock quote for a symbol
    """
    try:
        quote = await market_service.get_quote(symbol.upper())
        if not quote:
            raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found")
        return quote
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/quotes", response_model=List[StockQuote])
async def get_multiple_quotes(
    symbols: str = Query(..., description="Comma-separated list of symbols"),
    current_user: Dict = Depends(get_current_active_user)
) -> List[StockQuote]:
    """
    Get real-time quotes for multiple stocks
    """
    symbol_list = [s.strip().upper() for s in symbols.split(",")]
    if len(symbol_list) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 symbols allowed per request")
    
    try:
        quotes = await market_service.get_multiple_quotes(symbol_list)
        return quotes
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/historical/{symbol}", response_model=HistoricalData)
async def get_historical_data(
    symbol: str = Path(..., description="Stock symbol"),
    period: str = Query("1mo", description="Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, ytd, max)"),
    interval: str = Query("1d", description="Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)"),
    current_user: Dict = Depends(get_current_active_user)
) -> HistoricalData:
    """
    Get historical price data for a stock
    """
    valid_periods = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "ytd", "max"]
    valid_intervals = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]
    
    if period not in valid_periods:
        raise HTTPException(status_code=400, detail=f"Invalid period. Must be one of: {valid_periods}")
    
    if interval not in valid_intervals:
        raise HTTPException(status_code=400, detail=f"Invalid interval. Must be one of: {valid_intervals}")
    
    try:
        data = await market_service.get_historical_data(symbol.upper(), period, interval)
        if not data:
            raise HTTPException(status_code=404, detail=f"No historical data found for {symbol}")
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search", response_model=List[SearchResult])
async def search_symbols(
    query: str = Query(..., description="Search query"),
    limit: int = Query(10, ge=1, le=50, description="Maximum results to return"),
    current_user: Dict = Depends(get_current_active_user)
) -> List[SearchResult]:
    """
    Search for stocks by symbol or company name
    """
    if len(query) < 1:
        raise HTTPException(status_code=400, detail="Query must be at least 1 character")
    
    try:
        results = await market_service.search_symbols(query, limit)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/company/{symbol}", response_model=CompanyInfo)
async def get_company_info(
    symbol: str = Path(..., description="Stock symbol"),
    current_user: Dict = Depends(get_current_active_user)
) -> CompanyInfo:
    """
    Get detailed company information
    """
    try:
        info = await market_service.get_company_info(symbol.upper())
        if not info:
            raise HTTPException(status_code=404, detail=f"Company information not found for {symbol}")
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/market-overview", response_model=MarketOverview)
async def get_market_overview(
    current_user: Dict = Depends(get_current_active_user)
) -> MarketOverview:
    """
    Get market overview including major indices and market status
    """
    try:
        overview = await market_service.get_market_overview()
        return overview
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/watchlist", response_model=List[StockQuote])
async def get_watchlist(
    current_user: Dict = Depends(get_current_active_user)
) -> List[StockQuote]:
    """
    Get user's watchlist with current quotes
    """
    # This would typically fetch the user's watchlist from the database
    # For now, we'll use a default watchlist
    default_watchlist = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
    
    try:
        quotes = await market_service.get_multiple_quotes(default_watchlist)
        return quotes
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/watchlist/{symbol}")
async def add_to_watchlist(
    symbol: str = Path(..., description="Stock symbol to add"),
    current_user: Dict = Depends(get_current_active_user)
) -> Dict[str, str]:
    """
    Add a symbol to user's watchlist
    """
    # This would typically save to the database
    return {"message": f"Symbol {symbol.upper()} added to watchlist"}


@router.delete("/watchlist/{symbol}")
async def remove_from_watchlist(
    symbol: str = Path(..., description="Stock symbol to remove"),
    current_user: Dict = Depends(get_current_active_user)
) -> Dict[str, str]:
    """
    Remove a symbol from user's watchlist
    """
    # This would typically delete from the database
    return {"message": f"Symbol {symbol.upper()} removed from watchlist"}