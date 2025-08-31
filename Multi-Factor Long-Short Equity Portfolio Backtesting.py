import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from datetime import datetime, timedelta
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections import defaultdict
import logging
import inspect
from functools import wraps

@dataclass
class SecurityData:
    """Container for all security-related data"""
    prices: pd.DataFrame  # OHLCV data
    fundamentals: pd.DataFrame  # P/E, P/B, ROE, etc.
    market_data: pd.DataFrame  # Market cap, shares outstanding
    corporate_actions: pd.DataFrame  # Splits, dividends
    
class DataManager:
    """Manages all data sources and provides clean, aligned datasets"""
    
    def __init__(self, start_date: str, end_date: str):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.data_cache = {}
        self.logger = logging.getLogger(__name__)
        
    def load_price_data(self, symbols: List[str]) -> pd.DataFrame:
        """Load and clean price data with proper handling of corporate actions"""
        # In production, this would connect to your data provider
        # Here's the structure you'd want:
        
        price_data = {}
        for symbol in symbols:
            # Load raw OHLCV data
            raw_prices = self._fetch_raw_prices(symbol)
            
            # Adjust for splits and dividends
            adjusted_prices = self._adjust_for_corporate_actions(raw_prices, symbol)
            
            # Handle missing data and outliers
            clean_prices = self._clean_price_data(adjusted_prices)
            
            price_data[symbol] = clean_prices
            
        return pd.DataFrame(price_data)
    
    def _adjust_for_corporate_actions(self, prices: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Apply split and dividend adjustments"""
        corporate_actions = self._get_corporate_actions(symbol)
        
        adjusted_prices = prices.copy()
        
        for _, action in corporate_actions.iterrows():
            if action['type'] == 'split':
                # Adjust all prices before split date
                mask = adjusted_prices.index < action['date']
                split_ratio = action['ratio']
                adjusted_prices.loc[mask, ['open', 'high', 'low', 'close']] /= split_ratio
                adjusted_prices.loc[mask, 'volume'] *= split_ratio
                
            elif action['type'] == 'dividend':
                # Adjust prices for dividend
                mask = adjusted_prices.index < action['ex_date']
                dividend_amount = action['amount']
                adjusted_prices.loc[mask, ['open', 'high', 'low', 'close']] -= dividend_amount
                
        return adjusted_prices
    
    def _clean_price_data(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Clean price data for outliers and missing values"""
        clean_prices = prices.copy()
        
        # Remove obvious errors (negative prices, zero volume on trading days)
        clean_prices = clean_prices[clean_prices['close'] > 0]
        
        # Handle missing data with forward fill (limited)
        clean_prices = clean_prices.fillna(method='ffill', limit=5)
        
        # Detect and handle outliers (price jumps > 50% without corporate actions)
        price_changes = clean_prices['close'].pct_change()
        outlier_threshold = 0.5
        outliers = abs(price_changes) > outlier_threshold
        
        if outliers.any():
            self.logger.warning(f"Detected {outliers.sum()} potential outliers")
            # In production, you'd have more sophisticated outlier handling
        
        return clean_prices
    
    def get_universe(self, date: pd.Timestamp) -> List[str]:
        """Get tradeable universe at a given date"""
        # Implement liquidity, market cap, and other filters
        # This is critical for avoiding look-ahead bias
        pass


class CustomFactorRegistry:
    """Registry for managing custom factor definitions"""
    
    def __init__(self):
        self.factors = {}
        self.factor_metadata = {}
        
    def register_factor(self, name: str, func: Callable, 
                       description: str = "", 
                       required_data: List[str] = None,
                       parameters: Dict = None):
        """Register a custom factor function"""
        self.factors[name] = func
        self.factor_metadata[name] = {
            'description': description,
            'required_data': required_data or [],
            'parameters': parameters or {},
            'signature': str(inspect.signature(func))
        }
        
    def get_factor(self, name: str) -> Callable:
        """Get a registered factor function"""
        if name not in self.factors:
            raise ValueError(f"Factor '{name}' not found. Available factors: {list(self.factors.keys())}")
        return self.factors[name]
    
    def list_factors(self) -> Dict:
        """List all registered factors with metadata"""
        return self.factor_metadata.copy()

def factor_function(name: str, description: str = "", 
                   required_data: List[str] = None,
                   parameters: Dict = None):
    """Decorator for registering factor functions"""
    def decorator(func):
        # Store metadata in function attributes
        func._factor_name = name
        func._factor_description = description
        func._required_data = required_data or []
        func._parameters = parameters or {}
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator

class FactorCalculator:
    """Enhanced factor calculator with custom factor support"""
    
    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager
        self.factor_cache = {}
        self.custom_registry = CustomFactorRegistry()
        self.logger = logging.getLogger(__name__)
        
        # Register built-in factors
        self._register_builtin_factors()
        
    def _register_builtin_factors(self):
        """Register all built-in factor functions"""
        
        # Register momentum factors
        self.custom_registry.register_factor(
            'momentum_1m', self._momentum_1m,
            description="1-month price momentum",
            required_data=['prices'],
            parameters={'lookback_days': 21}
        )
        
        self.custom_registry.register_factor(
            'momentum_3m', self._momentum_3m,
            description="3-month price momentum",
            required_data=['prices'],
            parameters={'lookback_days': 63}
        )
        
        self.custom_registry.register_factor(
            'momentum_12m', self._momentum_12m,
            description="12-month price momentum",
            required_data=['prices'],
            parameters={'lookback_days': 252}
        )
        
        # Register value factors
        self.custom_registry.register_factor(
            'pe_ratio', self._pe_ratio,
            description="Price-to-Earnings ratio",
            required_data=['prices', 'fundamentals']
        )
        
        self.custom_registry.register_factor(
            'pb_ratio', self._pb_ratio,
            description="Price-to-Book ratio",
            required_data=['prices', 'fundamentals']
        )
        
        # Register quality factors
        self.custom_registry.register_factor(
            'roe', self._roe,
            description="Return on Equity",
            required_data=['fundamentals']
        )
        
        self.custom_registry.register_factor(
            'debt_to_equity', self._debt_to_equity,
            description="Debt-to-Equity ratio",
            required_data=['fundamentals']
        )
    
    def register_custom_factor(self, func: Callable, name: str = None,
                             description: str = "", 
                             required_data: List[str] = None,
                             parameters: Dict = None):
        """Register a custom factor function"""
        
        # Use function name if no name provided
        if name is None:
            name = getattr(func, '_factor_name', func.__name__)
            
        # Get metadata from decorator if available
        description = description or getattr(func, '_factor_description', "")
        required_data = required_data or getattr(func, '_required_data', [])
        parameters = parameters or getattr(func, '_parameters', {})
        
        self.custom_registry.register_factor(name, func, description, required_data, parameters)
        self.logger.info(f"Registered custom factor: {name}")
    
    def calculate_factor(self, factor_name: str, data_dict: Dict[str, pd.DataFrame], 
                        date: pd.Timestamp = None, **kwargs) -> pd.Series:
        """Calculate a specific factor given data"""
        
        # Check cache first
        cache_key = f"{factor_name}_{date}_{hash(str(sorted(kwargs.items())))}"
        if cache_key in self.factor_cache:
            return self.factor_cache[cache_key]
        
        # Get factor function
        factor_func = self.custom_registry.get_factor(factor_name)
        
        # Get factor metadata
        metadata = self.custom_registry.factor_metadata[factor_name]
        
        # Validate required data is available
        for required in metadata['required_data']:
            if required not in data_dict:
                raise ValueError(f"Factor '{factor_name}' requires '{required}' data")
        
        # Merge default parameters with user parameters
        factor_params = metadata['parameters'].copy()
        factor_params.update(kwargs)
        
        try:
            # Call factor function
            if date is not None:
                # If specific date requested, slice data
                sliced_data = {}
                for key, df in data_dict.items():
                    if isinstance(df.index, pd.DatetimeIndex):
                        sliced_data[key] = df.loc[:date]  # Up to and including date
                    else:
                        sliced_data[key] = df
                result = factor_func(sliced_data, date, **factor_params)
            else:
                result = factor_func(data_dict, **factor_params)
            
            # Cache result
            self.factor_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating factor '{factor_name}': {str(e)}")
            raise
    
    def calculate_multiple_factors(self, factor_names: List[str], 
                                 data_dict: Dict[str, pd.DataFrame],
                                 date: pd.Timestamp = None,
                                 factor_params: Dict[str, Dict] = None) -> pd.DataFrame:
        """Calculate multiple factors at once"""
        
        factor_params = factor_params or {}
        results = {}
        
        for factor_name in factor_names:
            params = factor_params.get(factor_name, {})
            try:
                factor_result = self.calculate_factor(factor_name, data_dict, date, **params)
                results[factor_name] = factor_result
            except Exception as e:
                self.logger.warning(f"Failed to calculate factor '{factor_name}': {str(e)}")
                continue
        
        if not results:
            return pd.DataFrame()
        
        # Combine results into DataFrame
        factor_df = pd.DataFrame(results)
        return factor_df
    
    def create_composite_factor(self, factor_names: List[str], 
                              weights: List[float],
                              data_dict: Dict[str, pd.DataFrame],
                              date: pd.Timestamp = None,
                              factor_params: Dict[str, Dict] = None,
                              standardize: bool = True) -> pd.Series:
        """Create a composite factor from multiple factors"""
        
        if len(factor_names) != len(weights):
            raise ValueError("Number of factors must match number of weights")
        
        # Calculate individual factors
        factor_df = self.calculate_multiple_factors(factor_names, data_dict, date, factor_params)
        
        if factor_df.empty:
            return pd.Series(dtype=float)
        
        # Standardize factors if requested
        if standardize:
            factor_df = factor_df.apply(self._standardize_factor, axis=0)
        
        # Create weighted composite
        weights = np.array(weights)
        composite = (factor_df * weights).sum(axis=1)
        
        return composite
    
    def _standardize_factor(self, factor_series: pd.Series) -> pd.Series:
        """Standardize factor to z-scores"""
        factor_clean = factor_series.dropna()
        if len(factor_clean) < 2:
            return factor_series
        
        mean = factor_clean.mean()
        std = factor_clean.std()
        
        if std == 0:
            return pd.Series(0, index=factor_series.index)
        
        standardized = (factor_series - mean) / std
        return standardized
    
    # Built-in factor implementations
    def _momentum_1m(self, data_dict: Dict, date: pd.Timestamp = None, 
                    lookback_days: int = 21, **kwargs) -> pd.Series:
        """1-month momentum factor"""
        prices = data_dict['prices']['close'] if 'close' in data_dict['prices'].columns else data_dict['prices']
        if date is not None:
            prices = prices.loc[:date]
        return prices.pct_change(lookback_days).iloc[-1] if len(prices) > lookback_days else pd.Series(dtype=float)
    
    def _momentum_3m(self, data_dict: Dict, date: pd.Timestamp = None, 
                    lookback_days: int = 63, **kwargs) -> pd.Series:
        """3-month momentum factor"""
        prices = data_dict['prices']['close'] if 'close' in data_dict['prices'].columns else data_dict['prices']
        if date is not None:
            prices = prices.loc[:date]
        return prices.pct_change(lookback_days).iloc[-1] if len(prices) > lookback_days else pd.Series(dtype=float)
    
    def _momentum_12m(self, data_dict: Dict, date: pd.Timestamp = None, 
                     lookback_days: int = 252, **kwargs) -> pd.Series:
        """12-month momentum factor"""
        prices = data_dict['prices']['close'] if 'close' in data_dict['prices'].columns else data_dict['prices']
        if date is not None:
            prices = prices.loc[:date]
        # Skip last month to avoid microstructure effects
        if len(prices) > lookback_days + 21:
            return prices.shift(21).pct_change(lookback_days).iloc[-1]
        return pd.Series(dtype=float)
    
    def _pe_ratio(self, data_dict: Dict, date: pd.Timestamp = None, **kwargs) -> pd.Series:
        """Price-to-Earnings ratio"""
        prices = data_dict['prices']['close'] if 'close' in data_dict['prices'].columns else data_dict['prices']
        fundamentals = data_dict['fundamentals']
        
        if date is not None:
            current_price = prices.loc[date] if date in prices.index else prices.iloc[-1]
            # Get most recent fundamental data before date
            fund_data = fundamentals.loc[:date].iloc[-1] if len(fundamentals.loc[:date]) > 0 else fundamentals.iloc[-1]
        else:
            current_price = prices.iloc[-1]
            fund_data = fundamentals.iloc[-1]
        
        eps = fund_data.get('earnings_per_share', pd.Series(dtype=float))
        pe_ratio = current_price / eps
        return pe_ratio.replace([np.inf, -np.inf], np.nan)
    
    def _pb_ratio(self, data_dict: Dict, date: pd.Timestamp = None, **kwargs) -> pd.Series:
        """Price-to-Book ratio"""
        prices = data_dict['prices']['close'] if 'close' in data_dict['prices'].columns else data_dict['prices']
        fundamentals = data_dict['fundamentals']
        
        if date is not None:
            current_price = prices.loc[date] if date in prices.index else prices.iloc[-1]
            fund_data = fundamentals.loc[:date].iloc[-1] if len(fundamentals.loc[:date]) > 0 else fundamentals.iloc[-1]
        else:
            current_price = prices.iloc[-1]
            fund_data = fundamentals.iloc[-1]
        
        book_value = fund_data.get('book_value_per_share', pd.Series(dtype=float))
        pb_ratio = current_price / book_value
        return pb_ratio.replace([np.inf, -np.inf], np.nan)
    
    def _roe(self, data_dict: Dict, date: pd.Timestamp = None, **kwargs) -> pd.Series:
        """Return on Equity"""
        fundamentals = data_dict['fundamentals']
        
        if date is not None:
            fund_data = fundamentals.loc[:date].iloc[-1] if len(fundamentals.loc[:date]) > 0 else fundamentals.iloc[-1]
        else:
            fund_data = fundamentals.iloc[-1]
        
        net_income = fund_data.get('net_income', pd.Series(dtype=float))
        shareholders_equity = fund_data.get('shareholders_equity', pd.Series(dtype=float))
        roe = net_income / shareholders_equity
        return roe.replace([np.inf, -np.inf], np.nan)
    
    def _debt_to_equity(self, data_dict: Dict, date: pd.Timestamp = None, **kwargs) -> pd.Series:
        """Debt-to-Equity ratio"""
        fundamentals = data_dict['fundamentals']
        
        if date is not None:
            fund_data = fundamentals.loc[:date].iloc[-1] if len(fundamentals.loc[:date]) > 0 else fundamentals.iloc[-1]
        else:
            fund_data = fundamentals.iloc[-1]
        
        total_debt = fund_data.get('total_debt', pd.Series(dtype=float))
        shareholders_equity = fund_data.get('shareholders_equity', pd.Series(dtype=float))
        debt_to_equity = total_debt / shareholders_equity
        return debt_to_equity.replace([np.inf, -np.inf], np.nan)

    def _align_fundamental_data(self, fundamentals: pd.DataFrame, 
                              price_dates: pd.DatetimeIndex) -> pd.DataFrame:
        """Ensure fundamental data is point-in-time to avoid look-ahead bias"""
        # This is crucial - you can only use fundamental data that was available at each point in time
        aligned_data = pd.DataFrame(index=price_dates, columns=fundamentals.columns)
        
        for date in price_dates:
            # Find the most recent fundamental data available before this date
            available_data = fundamentals[fundamentals.index <= date]
            if not available_data.empty:
                most_recent = available_data.iloc[-1]
                aligned_data.loc[date] = most_recent
                
        return aligned_data.fillna(method='ffill')

    # Legacy example custom factor methods
    def calculate_momentum_factors(self, prices: pd.DataFrame, 
                                 lookback_periods: List[int] = [21, 63, 252]) -> pd.DataFrame:
        """Legacy momentum calculation - kept for backward compatibility"""
        data_dict = {'prices': prices}
        
        momentum_factors = pd.DataFrame(index=prices.index, columns=prices.columns)
        
        for period in lookback_periods:
            factor_name = f'momentum_{period}d'
            # Register temporary factor if not exists
            if factor_name not in self.custom_registry.factors:
                def temp_momentum(data_dict, date=None, lookback_days=period):
                    return self._momentum_1m(data_dict, date, lookback_days)
                self.custom_registry.register_factor(factor_name, temp_momentum)
            
            for date in prices.index:
                result = self.calculate_factor(factor_name, data_dict, date)
                if not result.empty:
                    momentum_factors.loc[date] = result
        
        # Create composite
        momentum_factors['momentum_composite'] = self.create_composite_factor(
            [f'momentum_{p}d' for p in lookback_periods],
            [0.2, 0.3, 0.5],
            data_dict
        )
        
        return momentum_factors

    def calculate_value_factors(self, prices: pd.DataFrame, 
                              fundamentals: pd.DataFrame) -> pd.DataFrame:
        """Calculate value factors using fundamental data"""
        value_factors = pd.DataFrame(index=prices.index, columns=prices.columns)
        
        # Ensure fundamental data is point-in-time (critical!)
        aligned_fundamentals = self._align_fundamental_data(fundamentals, prices.index)
        
        # Calculate market cap
        market_cap = prices * aligned_fundamentals['shares_outstanding']
        
        # Price-to-Book ratio
        book_value = aligned_fundamentals['book_value_per_share']
        value_factors['pb_ratio'] = prices / book_value
        
        # Price-to-Earnings ratio
        earnings_per_share = aligned_fundamentals['earnings_per_share']
        value_factors['pe_ratio'] = prices / earnings_per_share
        
        # Enterprise Value to EBITDA
        enterprise_value = market_cap + aligned_fundamentals['total_debt'] - aligned_fundamentals['cash']
        ebitda = aligned_fundamentals['ebitda']
        value_factors['ev_ebitda'] = enterprise_value / ebitda
        
        # Convert to scores (lower ratios = higher value scores)
        for col in ['pb_ratio', 'pe_ratio', 'ev_ebitda']:
            value_factors[f'{col}_score'] = self._convert_to_scores(-value_factors[col])
            
        # Composite value score
        value_factors['value_composite'] = (
            value_factors['pb_ratio_score'] * 0.4 +
            value_factors['pe_ratio_score'] * 0.4 +
            value_factors['ev_ebitda_score'] * 0.2
        )
        
        return value_factors
    
    def calculate_quality_factors(self, fundamentals: pd.DataFrame) -> pd.DataFrame:
        """Calculate quality factors"""
        quality_factors = pd.DataFrame(index=fundamentals.index, columns=fundamentals.columns)

        # Ensure fundamental data is point-in-time       
        aligned_fundamentals = self._align_fundamental_data(fundamentals, prices.index)

        # Return on Equity
        roe = aligned_fundamentals['net_income'] / aligned_fundamentals['shareholders_equity']
        quality_factors['roe'] = roe
        
        # Debt-to-Equity ratio
        debt_to_equity = aligned_fundamentals['total_debt'] / aligned_fundamentals['shareholders_equity']
        quality_factors['debt_to_equity'] = debt_to_equity
        
        # Earnings stability (standard deviation of ROE over 5 years)
        roe_rolling_std = roe.rolling(window=20, min_periods=12).std()  # 5 years quarterly
        quality_factors['earnings_stability'] = -roe_rolling_std  # Negative because lower std is better
        
        # Convert to scores
        quality_factors['roe_score'] = self._convert_to_scores(quality_factors['roe'])
        quality_factors['debt_score'] = self._convert_to_scores(-quality_factors['debt_to_equity'])
        quality_factors['stability_score'] = self._convert_to_scores(quality_factors['earnings_stability'])
        
        # Composite quality score
        quality_factors['quality_composite'] = (
            quality_factors['roe_score'] * 0.4 +
            quality_factors['debt_score'] * 0.3 +
            quality_factors['stability_score'] * 0.3
        )
        
        return quality_factors


class PortfolioConstructor:
    """Constructs portfolios based on factor signals with realistic constraints"""
    
    def __init__(self, transaction_cost_model, risk_model):
        self.transaction_cost_model = transaction_cost_model
        self.risk_model = risk_model
        self.logger = logging.getLogger(__name__)
        
    def construct_portfolio(self, factor_scores: pd.DataFrame, 
                          current_portfolio: pd.Series,
                          universe: List[str],
                          date: pd.Timestamp,
                          **constraints) -> pd.Series:
        """Construct optimal portfolio given factor signals and constraints"""
        
        # Filter universe for tradeable securities
        tradeable_universe = self._filter_universe(universe, date)
        
        # Get factor exposures for tradeable securities
        exposures = factor_scores.loc[date, tradeable_universe].dropna()
        
        if exposures.empty:
            self.logger.warning(f"No valid exposures for date {date}")
            return current_portfolio
        
        # Construct portfolio using optimization
        if constraints.get('method', 'quantile') == 'quantile':
            new_portfolio = self._quantile_portfolio(exposures, **constraints)
        elif constraints.get('method') == 'optimization':
            new_portfolio = self._optimized_portfolio(exposures, current_portfolio, date, **constraints)
        else:
            raise ValueError(f"Unknown portfolio construction method: {constraints.get('method')}")
            
        return new_portfolio
    
    def _quantile_portfolio(self, exposures: pd.Series, 
                          long_pct: float = 0.2, 
                          short_pct: float = 0.2,
                          **kwargs) -> pd.Series:
        """Construct portfolio using quantile-based approach"""
        
        # Sort securities by factor exposure
        sorted_exposures = exposures.sort_values(ascending=False)
        n_securities = len(sorted_exposures)
        
        # Determine long and short positions
        n_long = int(n_securities * long_pct)
        n_short = int(n_securities * short_pct)
        
        portfolio = pd.Series(0.0, index=exposures.index)
        
        if n_long > 0:
            long_securities = sorted_exposures.head(n_long).index
            portfolio[long_securities] = 1.0 / n_long
            
        if n_short > 0:
            short_securities = sorted_exposures.tail(n_short).index
            portfolio[short_securities] = -1.0 / n_short
            
        return portfolio
    
    def _optimized_portfolio(self, exposures: pd.Series,
                           current_portfolio: pd.Series,
                           date: pd.Timestamp,
                           target_volatility: float = 0.15,
                           max_turnover: float = 0.5,
                           **kwargs) -> pd.Series:
        """Construct portfolio using mean-variance optimization with transaction costs"""
        
        from scipy.optimize import minimize
        
        # Get expected returns (factor exposures)
        expected_returns = exposures
        
        # Get covariance matrix from risk model
        covariance_matrix = self.risk_model.get_covariance_matrix(exposures.index, date)
        
        # Current positions (aligned with universe)
        current_positions = current_portfolio.reindex(exposures.index, fill_value=0.0)
        
        # Objective function: maximize expected return - transaction costs - risk penalty
        def objective(weights):
            weights = pd.Series(weights, index=exposures.index)
            
            # Expected return
            expected_return = (weights * expected_returns).sum()
            
            # Transaction costs
            turnover = abs(weights - current_positions).sum()
            transaction_costs = self.transaction_cost_model.estimate_costs(
                abs(weights - current_positions), date
            )
            
            # Risk penalty
            portfolio_variance = np.dot(weights.values, np.dot(covariance_matrix, weights.values))
            risk_penalty = 0.5 * portfolio_variance / (target_volatility ** 2)
            
            # Maximize return - costs - risk
            return -(expected_return - transaction_costs - risk_penalty)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: w.sum()},  # Dollar neutral (if long-short)
            {'type': 'ineq', 'fun': lambda w: max_turnover - abs(w - current_positions.values).sum()}
        ]
        
        # Bounds (position limits)
        max_position = kwargs.get('max_position_size', 0.05)
        bounds = [(-max_position, max_position) for _ in range(len(exposures))]
        
        # Initial guess
        x0 = current_positions.values
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if not result.success:
            self.logger.warning(f"Optimization failed for date {date}: {result.message}")
            return current_positions
            
        return pd.Series(result.x, index=exposures.index)
    
    def _filter_universe(self, universe: List[str], date: pd.Timestamp) -> List[str]:
        """Filter universe for tradeable securities"""
        # Apply liquidity, market cap, and other filters
        # This is crucial for realistic backtesting
        
        tradeable = []
        for symbol in universe:
            # Check if security is liquid enough
            if self._is_liquid(symbol, date):
                # Check if security meets market cap requirements
                if self._meets_market_cap_requirement(symbol, date):
                    # Check if security is not in corporate action period
                    if not self._in_corporate_action_period(symbol, date):
                        tradeable.append(symbol)
                        
        return tradeable
    
    def _is_liquid(self, symbol: str, date: pd.Timestamp) -> bool:
        """Check if security meets liquidity requirements"""
        # Implement liquidity checks (average daily volume, bid-ask spread, etc.)
        # TO BE DEVELOPED, Return True if liquid enough to trade
        return True  # Placeholder
    
    def _meets_market_cap_requirement(self, symbol: str, date: pd.Timestamp) -> bool:
        """Check market cap requirements"""
        # Implement market cap filters
        # TO BE DEVELOPED
        return True  # Placeholder
    
    def _in_corporate_action_period(self, symbol: str, date: pd.Timestamp) -> bool:
        """Check if security is in a corporate action period"""
        # Avoid trading around earnings, splits, etc.
        # TO BE DEVELOPED
        return False  # Placeholder


class TransactionCostModel:
    """Models realistic transaction costs including market impact"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def estimate_costs(self, trade_sizes: pd.Series, date: pd.Timestamp, 
                      prices: pd.Series = None) -> float:
        """Estimate total transaction costs for a set of trades"""
        
        total_costs = 0.0
        
        for symbol, trade_size in trade_sizes.items():
            if abs(trade_size) < 1e-6:  # Skip tiny trades
                continue
                
            # Get security-specific parameters
            price = prices[symbol] if prices is not None else 100.0  # Default price
            adv = self._get_average_daily_volume(symbol, date)
            
            # Commission costs
            commission = self._calculate_commission(trade_size, price)
            
            # Bid-ask spread costs
            spread_cost = self._calculate_spread_cost(trade_size, price, symbol, date)
            
            # Market impact costs
            impact_cost = self._calculate_market_impact(trade_size, price, adv, symbol)
            
            # Timing costs (opportunity cost of gradual execution)
            timing_cost = self._calculate_timing_cost(trade_size, price, symbol)
            
            security_cost = commission + spread_cost + impact_cost + timing_cost
            total_costs += security_cost
            
        return total_costs
    
    def _calculate_commission(self, trade_size: float, price: float) -> float:
        """Calculate commission costs"""
        notional = abs(trade_size) * price
        
        # Tiered commission structure
        if notional < 10000:
            commission_rate = 0.001  # 10 bps
        elif notional < 100000:
            commission_rate = 0.0005  # 5 bps
        else:
            commission_rate = 0.0003  # 3 bps
            
        return notional * commission_rate
    
    def _calculate_spread_cost(self, trade_size: float, price: float, 
                             symbol: str, date: pd.Timestamp) -> float:
        """Calculate bid-ask spread costs"""
        
        # Estimate spread based on security characteristics
        market_cap = self._get_market_cap(symbol, date)
        volume = self._get_average_daily_volume(symbol, date)
        
        # Spread model based on market microstructure research
        if market_cap > 10e9:  # Large cap
            spread_bps = 2
        elif market_cap > 2e9:  # Mid cap
            spread_bps = 5
        else:  # Small cap
            spread_bps = 10
            
        # Adjust for volume
        if volume < 100000:  # Low volume
            spread_bps *= 2
            
        spread_cost = abs(trade_size) * price * (spread_bps / 10000) * 0.5  # Pay half spread
        return spread_cost
    
    def _calculate_market_impact(self, trade_size: float, price: float, 
                               adv: float, symbol: str) -> float:
        """Calculate market impact using square-root law"""
        
        if adv <= 0:
            return 0.0
            
        # Participation rate (what fraction of daily volume we're trading)
        participation_rate = abs(trade_size) / adv
        
        # Market impact model: impact = sigma * (participation_rate)^0.5
        # Where sigma is the daily volatility
        daily_volatility = self._get_daily_volatility(symbol)
        
        # Square-root market impact law
        impact_bps = daily_volatility * 100 * np.sqrt(participation_rate) * 10
        
        # Apply sign (temporary impact)
        impact_cost = abs(trade_size) * price * (impact_bps / 10000)
        
        return impact_cost
    
    def _calculate_timing_cost(self, trade_size: float, price: float, symbol: str) -> float:
        """Calculate timing costs for gradual execution"""
        
        # Assume we execute over multiple periods to reduce impact
        # This creates timing risk (opportunity cost)
        
        daily_volatility = self._get_daily_volatility(symbol)
        execution_days = min(5, max(1, abs(trade_size) / self._get_average_daily_volume(symbol, None) * 10))
        
        # Timing cost proportional to volatility and execution time
        timing_cost = abs(trade_size) * price * daily_volatility * np.sqrt(execution_days) * 0.1
        
        return timing_cost
    
    def _get_average_daily_volume(self, symbol: str, date: pd.Timestamp) -> float:
        """Get average daily volume for symbol"""
        # In production, fetch from local data or an api provider like EODHD, FMP, Alpha Vantage, etc.
        return 1000000  # Placeholder
    
    def _get_market_cap(self, symbol: str, date: pd.Timestamp) -> float:
        """Get market cap for symbol"""
        # In production, fetch from local data or an api provider like EODHD, FMP, Alpha Vantage, etc.
        return 5e9  # Placeholder
    
    def _get_daily_volatility(self, symbol: str) -> float:
        """Get daily volatility for symbol"""
        # In production, calculate from historical returns
        return 0.02  # Placeholder (2% daily vol)


class RiskModel:
    """Multi-factor risk model for portfolio risk management"""
    
    def __init__(self):
        self.factor_returns = {}
        self.specific_risks = {}
        
    def get_covariance_matrix(self, securities: List[str], date: pd.Timestamp) -> np.ndarray:
        """Get covariance matrix for securities"""
        
        # Factor-based covariance model: Cov = B * F * B' + Delta
        # Where B = factor loadings, F = factor covariance, Delta = specific variance
        
        factor_loadings = self._get_factor_loadings(securities, date)
        factor_covariance = self._get_factor_covariance(date)
        specific_variance = self._get_specific_variance(securities, date)
        
        # Calculate systematic covariance
        systematic_cov = np.dot(factor_loadings, np.dot(factor_covariance, factor_loadings.T))
        
        # Add specific risk (diagonal matrix)
        total_covariance = systematic_cov + np.diag(specific_variance)
        
        return total_covariance
    
    def _get_factor_loadings(self, securities: List[str], date: pd.Timestamp) -> np.ndarray:
        """Get factor loadings (exposures) for securities"""
        # In production, this would come from a risk model provider or internal calculation
        
        n_securities = len(securities)
        n_factors = 10  # Market, Size, Value, Momentum, Quality, etc.
        
        # Generate realistic factor loadings
        loadings = np.random.randn(n_securities, n_factors) * 0.5
        
        # First factor is market (beta around 1)
        loadings[:, 0] = 1.0 + np.random.randn(n_securities) * 0.3
        
        return loadings
    
    def _get_factor_covariance(self, date: pd.Timestamp) -> np.ndarray:
        """Get factor covariance matrix"""
        # In production, estimate from historical factor returns
        
        n_factors = 10
        
        # Generate reasonable factor covariance
        factor_vol = np.array([0.15, 0.08, 0.06, 0.05, 0.04, 0.04, 0.03, 0.03, 0.03, 0.02])
        factor_corr = np.eye(n_factors)
        
        # Add some correlations between factors
        factor_corr[0, 1] = -0.3  # Market and size
        factor_corr[1, 0] = -0.3
        factor_corr[2, 4] = 0.2   # Value and quality
        factor_corr[4, 2] = 0.2
        
        # Convert to covariance
        factor_cov = np.outer(factor_vol, factor_vol) * factor_corr
        
        return factor_cov
    
    def _get_specific_variance(self, securities: List[str], date: pd.Timestamp) -> np.ndarray:
        """Get specific (idiosyncratic) variance for securities"""
        # In production, estimate from historical residual returns
        
        n_securities = len(securities)
        
        # Specific volatility typically 20-40% annualized
        specific_vol = 0.25 + np.random.randn(n_securities) * 0.1
        specific_vol = np.clip(specific_vol, 0.1, 0.5)  # Reasonable bounds
        
        # Convert to daily variance
        specific_variance = (specific_vol / np.sqrt(252)) ** 2
        
        return specific_variance

class RiskManager:
    """Manages portfolio risk and applies constraints"""
    
    def __init__(self, risk_model: RiskModel):
        self.risk_model = risk_model
        self.risk_limits = {
            'max_portfolio_volatility': 0.20,
            'max_sector_exposure': 0.10,
            'max_single_name': 0.05,
            'max_turnover': 0.50
        }
        
    def check_risk_constraints(self, portfolio: pd.Series, date: pd.Timestamp) -> Tuple[bool, List[str]]:
        """Check if portfolio satisfies risk constraints"""
        
        violations = []
        
        # Portfolio volatility check
        portfolio_vol = self._calculate_portfolio_volatility(portfolio, date)
        if portfolio_vol > self.risk_limits['max_portfolio_volatility']:
            violations.append(f"Portfolio volatility {portfolio_vol:.3f} exceeds limit {self.risk_limits['max_portfolio_volatility']}")
        
        # Single name concentration
        max_position = abs(portfolio).max()
        if max_position > self.risk_limits['max_single_name']:
            violations.append(f"Max position {max_position:.3f} exceeds limit {self.risk_limits['max_single_name']}")
        
        # Sector concentration (would need sector mapping)
        sector_exposures = self._calculate_sector_exposures(portfolio)
        max_sector_exposure = max(sector_exposures.values()) if sector_exposures else 0
        if max_sector_exposure > self.risk_limits['max_sector_exposure']:
            violations.append(f"Max sector exposure {max_sector_exposure:.3f} exceeds limit {self.risk_limits['max_sector_exposure']}")
        
        return len(violations) == 0, violations
    
    def _calculate_portfolio_volatility(self, portfolio: pd.Series, date: pd.Timestamp) -> float:
        """Calculate portfolio volatility"""
        securities = portfolio.index.tolist()
        covariance_matrix = self.risk_model.get_covariance_matrix(securities, date)
        
        portfolio_variance = np.dot(portfolio.values, np.dot(covariance_matrix, portfolio.values))
        portfolio_volatility = np.sqrt(portfolio_variance * 252)  # Annualized
        
        return portfolio_volatility
    
    def _calculate_sector_exposures(self, portfolio: pd.Series) -> Dict[str, float]:
        """Calculate exposure to each sector"""
        # In production, you'd have a security-to-sector mapping
        # For now, return empty dict
        return {}


class BacktestingEngine:
    """Main backtesting engine that orchestrates the entire process"""
    
    def __init__(self, data_manager: DataManager, 
                 factor_calculator: FactorCalculator,
                 portfolio_constructor: PortfolioConstructor,
                 risk_manager: RiskManager,
                 transaction_cost_model: TransactionCostModel):
        
        self.data_manager = data_manager
        self.factor_calculator = factor_calculator
        self.portfolio_constructor = portfolio_constructor
        self.risk_manager = risk_manager
        self.transaction_cost_model = transaction_cost_model
        
        # Backtesting state
        self.portfolio_history = {}
        self.return_history = {}
        self.transaction_costs = {}
        self.factor_exposures = {}
        
        self.logger = logging.getLogger(__name__)

    def register_custom_factor(self, func: Callable, name: str = None, **kwargs):
        """Register a custom factor with the engine"""
        self.factor_calculator.register_custom_factor(func, name, **kwargs)
    
    def _calculate_factor_scores(self, universe: List[str], date: pd.Timestamp, 
                               factor_config: Dict) -> pd.DataFrame:
        """Enhanced factor score calculation supporting custom factors"""
        
        # Prepare data dictionary
        data_dict = self._prepare_data_dict(universe, date)
        
        if not data_dict:
            self.logger.warning(f"No data available for {date}")
            return pd.DataFrame()
        
        # Get factor configuration
        factors_to_calculate = factor_config.get('factors', [])
        factor_weights = factor_config.get('factor_weights', {})
        factor_params = factor_config.get('factor_parameters', {})
        composite_method = factor_config.get('composite_method', 'weighted_average')
        
        if not factors_to_calculate:
            # Fallback to legacy behavior
            return self._legacy_factor_calculation(universe, date, factor_config, data_dict)
        
        # Calculate individual factors
        factor_results = {}
        
        for factor_name in factors_to_calculate:
            try:
                params = factor_params.get(factor_name, {})
                factor_result = self.factor_calculator.calculate_factor(
                    factor_name, data_dict, date, **params
                )
                
                if not factor_result.empty:
                    factor_results[factor_name] = factor_result
                else:
                    self.logger.warning(f"Empty result for factor '{factor_name}' on {date}")
                    
            except Exception as e:
                self.logger.error(f"Error calculating factor '{factor_name}' on {date}: {str(e)}")
                continue
        
        if not factor_results:
            self.logger.warning(f"No valid factor results for {date}")
            return pd.DataFrame()
        
        # Create composite score
        factor_scores = self._create_composite_score(
            factor_results, factor_weights, composite_method
        )
        
        # Return as DataFrame for consistency
        result_df = pd.DataFrame(index=[date], columns=universe)
        result_df.loc[date] = factor_scores.reindex(universe, fill_value=np.nan)
        
        return result_df
    
    def _prepare_data_dict(self, universe: List[str], date: pd.Timestamp) -> Dict[str, pd.DataFrame]:
        """Prepare data dictionary for factor calculations"""
        
        data_dict = {}
        
        try:
            # Get price data
            price_data = self.data_manager.get_price_data(universe, date)
            if not price_data.empty:
                data_dict['prices'] = price_data
            
            # Get fundamental data
            fundamental_data = self.data_manager.get_fundamental_data(universe, date)
            if not fundamental_data.empty:
                data_dict['fundamentals'] = fundamental_data
            
            # Get market data
            market_data = self.data_manager.get_market_data(universe, date)
            if not market_data.empty:
                data_dict['market_data'] = market_data
            
            # Add any other data sources as needed
            
        except Exception as e:
            self.logger.error(f"Error preparing data for {date}: {str(e)}")
            return {}
        
        return data_dict
    
    def _create_composite_score(self, factor_results: Dict[str, pd.Series], 
                              factor_weights: Dict[str, float],
                              method: str = 'weighted_average') -> pd.Series:
        """Create composite factor score from individual factors"""
        
        if not factor_results:
            return pd.Series(dtype=float)
        
        # Convert to DataFrame for easier manipulation
        factor_df = pd.DataFrame(factor_results)
        
        # Standardize factors
        standardized_factors = factor_df.apply(self.factor_calculator._standardize_factor, axis=0)
        
        if method == 'weighted_average':
            # Use provided weights or equal weights
            if factor_weights:
                weights = pd.Series(factor_weights).reindex(standardized_factors.columns, fill_value=0)
                weights = weights / weights.sum()  # Normalize weights
            else:
                weights = pd.Series(1.0 / len(standardized_factors.columns), 
                                  index=standardized_factors.columns)
            
            composite_score = (standardized_factors * weights).sum(axis=1)
            
        elif method == 'equal_weight':
            composite_score = standardized_factors.mean(axis=1)
            
        elif method == 'rank_average':
            # Average of factor ranks
            factor_ranks = standardized_factors.rank(axis=0, pct=True)
            composite_score = factor_ranks.mean(axis=1)
            
        elif method == 'ic_weighted':
            # Weight by historical information coefficient (simplified)
            # In production, you'd calculate actual ICs
            ic_weights = pd.Series(1.0, index=standardized_factors.columns)
            ic_weights = ic_weights / ic_weights.sum()
            composite_score = (standardized_factors * ic_weights).sum(axis=1)
            
        else:
            raise ValueError(f"Unknown composite method: {method}")
        
        return composite_score
        
    def run_backtest(self, start_date: str, end_date: str, 
                    rebalance_frequency: str = 'M',
                    initial_capital: float = 1000000,
                    factor_config: Dict = None) -> Dict:
        """Run the complete backtesting process"""
        
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Generate rebalancing dates
        rebalance_dates = pd.date_range(start=start_dt, end=end_dt, freq=rebalance_frequency)
        
        # Initialize portfolio
        current_portfolio = pd.Series(dtype=float)
        portfolio_value = initial_capital
        
        self.logger.info(f"Starting backtest from {start_date} to {end_date}")
        
        for i, rebal_date in enumerate(rebalance_dates):
            self.logger.info(f"Rebalancing on {rebal_date.strftime('%Y-%m-%d')} ({i+1}/{len(rebalance_dates)})")
            
            try:
                # Get universe for this date
                universe = self.data_manager.get_universe(rebal_date)
                
                if not universe:
                    self.logger.warning(f"No universe available for {rebal_date}")
                    continue
                
                # Calculate factor scores
                factor_scores = self._calculate_factor_scores(universe, rebal_date, factor_config)
                
                if factor_scores.empty:
                    self.logger.warning(f"No factor scores available for {rebal_date}")
                    continue
                
                # Construct new portfolio
                new_portfolio = self.portfolio_constructor.construct_portfolio(
                    factor_scores, current_portfolio, universe, rebal_date,
                    **factor_config.get('portfolio_config', {})
                )
                
                # Risk management checks
                risk_ok, violations = self.risk_manager.check_risk_constraints(new_portfolio, rebal_date)
                
                if not risk_ok:
                    self.logger.warning(f"Risk violations on {rebal_date}: {violations}")
                    # In production, you might adjust the portfolio or skip rebalancing
                
                # Calculate transaction costs
                trades = new_portfolio - current_portfolio.reindex(new_portfolio.index, fill_value=0.0)
                prices = self.data_manager.get_prices(new_portfolio.index, rebal_date)
                transaction_cost = self.transaction_cost_model.estimate_costs(trades, rebal_date, prices)
                
                # Execute trades (simulate)
                current_portfolio = new_portfolio.copy()
                portfolio_value -= transaction_cost
                
                # Store results
                self.portfolio_history[rebal_date] = current_portfolio.copy()
                self.transaction_costs[rebal_date] = transaction_cost
                self.factor_exposures[rebal_date] = factor_scores.loc[rebal_date]
                
                # Calculate returns to next rebalancing date
                if i < len(rebalance_dates) - 1:
                    next_date = rebalance_dates[i + 1]
                    period_return = self._calculate_period_return(current_portfolio, rebal_date, next_date)
                    self.return_history[rebal_date] = period_return
                    portfolio_value *= (1 + period_return)
                
            except Exception as e:
                self.logger.error(f"Error processing {rebal_date}: {str(e)}")
                continue
        
        # Calculate performance metrics
        results = self._calculate_performance_metrics(initial_capital)
        
        self.logger.info("Backtest completed successfully")
        return results
    
    def _calculate_period_return(self, portfolio: pd.Series, start_date: pd.Timestamp, 
                               end_date: pd.Timestamp) -> float:
        """Calculate portfolio return over a period"""
        
        # Get price returns for all securities in portfolio
        securities = portfolio.index.tolist()
        start_prices = self.data_manager.get_prices(securities, start_date)
        end_prices = self.data_manager.get_prices(securities, end_date)
        
        # Calculate security returns
        security_returns = (end_prices / start_prices - 1).fillna(0.0)
        
        # Calculate portfolio return
        portfolio_return = (portfolio * security_returns).sum()
        
        return portfolio_return
    
    def _calculate_performance_metrics(self, initial_capital: float) -> Dict:
        """Calculate comprehensive performance metrics"""
        
        # Convert returns to time series
        returns_series = pd.Series(self.return_history)
        returns_series = returns_series.sort_index()
        
        # Calculate cumulative returns
        cumulative_returns = (1 + returns_series).cumprod()
        
        # Performance metrics
        total_return = cumulative_returns.iloc[-1] - 1
        annualized_return = (1 + total_return) ** (252 / len(returns_series)) - 1
        volatility = returns_series.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Risk metrics
        max_drawdown = self._calculate_max_drawdown(cumulative_returns)
        
        # Transaction cost analysis
        total_transaction_costs = sum(self.transaction_costs.values())
        transaction_cost_pct = total_transaction_costs / initial_capital
        
        results = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_transaction_costs': total_transaction_costs,
            'transaction_cost_pct': transaction_cost_pct,
            'returns_series': returns_series,
            'cumulative_returns': cumulative_returns,
            'portfolio_history': self.portfolio_history,
            'factor_exposures': self.factor_exposures
        }
        
        return results
    
    def _calculate_max_drawdown(self, cumulative_returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        return max_drawdown

    def _legacy_factor_calculation(self, universe: List[str], date: pd.Timestamp, 
                                 factor_config: Dict, data_dict: Dict) -> pd.DataFrame:
        """Legacy factor calculation for backward compatibility"""
        
        factor_scores = pd.DataFrame(index=[date], columns=universe)
        
        # Calculate individual factors using legacy methods
        if factor_config.get('use_momentum', False):
            momentum_scores = self.factor_calculator.calculate_factor(
                'momentum_3m', data_dict, date
            )
            if not momentum_scores.empty:
                factor_scores.loc[date] = momentum_scores
        
        if factor_config.get('use_value', False):
            value_weight = factor_config.get('value_weight', 0.5)
            pe_scores = self.factor_calculator.calculate_factor('pe_ratio', data_dict, date)
            if not pe_scores.empty:
                # Convert PE to value score (lower PE = higher value)
                value_scores = self.factor_calculator._standardize_factor(-pe_scores)
                
                if factor_scores.loc[date].notna().any():
                    current_scores = factor_scores.loc[date].fillna(0)
                    factor_scores.loc[date] = current_scores + value_scores * value_weight
                else:
                    factor_scores.loc[date] = value_scores
        
        if factor_config.get('use_quality', False):
            quality_weight = factor_config.get('quality_weight', 0.3)
            roe_scores = self.factor_calculator.calculate_factor('roe', data_dict, date)
            if not roe_scores.empty:
                quality_scores = self.factor_calculator._standardize_factor(roe_scores)
                
                if factor_scores.loc[date].notna().any():
                    current_scores = factor_scores.loc[date].fillna(0)
                    factor_scores.loc[date] = current_scores + quality_scores * quality_weight
                else:
                    factor_scores.loc[date] = quality_scores
        
        return factor_scores.dropna(axis=1)

### Usage Example 1 ###
def main_one():
    """Example usage of the backtesting engine"""
    
    # Initialize components
    data_manager = DataManager('2020-01-01', '2023-12-31')
    factor_calculator = FactorCalculator(data_manager)
    transaction_cost_model = TransactionCostModel()
    risk_model = RiskModel()
    risk_manager = RiskManager(risk_model)
    portfolio_constructor = PortfolioConstructor(transaction_cost_model, risk_model)
    
    # Create backtesting engine
    engine = BacktestingEngine(
        data_manager=data_manager,
        factor_calculator=factor_calculator,
        portfolio_constructor=portfolio_constructor,
        risk_manager=risk_manager,
        transaction_cost_model=transaction_cost_model
    )
    
    # Configure factors and portfolio construction
    factor_config = {
        'use_momentum': True,
        'use_value': True,
        'use_quality': True,
        'momentum_weight': 0.4,
        'value_weight': 0.4,
        'quality_weight': 0.2,
        'portfolio_config': {
            'method': 'quantile',
            'long_pct': 0.2,
            'short_pct': 0.2,
            'max_position_size': 0.05
        }
    }
    
    # Run backtest
    results = engine.run_backtest(
        start_date='2020-01-01',
        end_date='2023-12-31',
        rebalance_frequency='M',  # Monthly rebalancing
        initial_capital=10000000,  # $10M
        factor_config=factor_config
    )
    
    # Print results
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Annualized Return: {results['annualized_return']:.2%}")
    print(f"Volatility: {results['volatility']:.2%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2%}")
    print(f"Transaction Costs: {results['transaction_cost_pct']:.2%} of capital")

### Usage Example 2 ###
# Define custom 1-week volatility factor
@factor_function(
    name="volatility_1w",
    description="1-week realized volatility",
    required_data=["prices"],
    parameters={"lookback_days": 5, "annualize": True}
)
def calculate_1w_volatility(data_dict: Dict, date: pd.Timestamp = None, 
                           lookback_days: int = 5, annualize: bool = True) -> pd.Series:
    """
    Calculate 1-week (5-day) realized volatility
    """
    # Get price data
    if 'close' in data_dict['prices'].columns:
        prices = data_dict['prices']['close']
    else:
        prices = data_dict['prices']
    
    # Slice data up to the calculation date
    if date is not None:
        prices = prices.loc[:date]
    
    # Need at least lookback_days + 1 observations
    if len(prices) < lookback_days + 1:
        return pd.Series(dtype=float, index=prices.columns)
    
    # Calculate daily returns
    returns = prices.pct_change().dropna()
    
    # Calculate volatility over the lookback period
    volatility = returns.tail(lookback_days).std()
    
    # Annualize if requested
    if annualize:
        volatility = volatility * np.sqrt(252)  # 252 trading days per year
    
    return volatility

# Define custom factor: Price Gap
@factor_function(
    name="price_gap",
    description="Average price gap (open vs previous close)",
    required_data=["prices"],
    parameters={"lookback_days": 20}
)
def calculate_price_gap(data_dict: Dict, date: pd.Timestamp = None,
                       lookback_days: int = 20) -> pd.Series:
    """
    Calculate average price gap factor
    Measures the average gap between open and previous close
    """
    prices = data_dict['prices']
    
    if date is not None:
        prices = prices.loc[:date]
    
    if len(prices) < lookback_days + 1:
        return pd.Series(dtype=float)
    
    # Calculate gaps
    prev_close = prices['close'].shift(1)
    current_open = prices['open']
    gaps = (current_open - prev_close) / prev_close
    
    # Average gap over lookback period
    avg_gap = gaps.tail(lookback_days).mean()
    
    return avg_gap

# Define custom factor: Momentum-volatility interaction
@factor_function(
    name="momentum_vol_interaction",
    description="Interaction between momentum and volatility",
    required_data=["prices"],
    parameters={"momentum_days": 21, "vol_days": 5}
)
def calculate_momentum_vol_interaction(data_dict: Dict, date: pd.Timestamp = None,
                                     momentum_days: int = 21, vol_days: int = 5) -> pd.Series:
    """
    Calculate momentum-volatility interaction factor
    High momentum + low volatility = positive signal
    """
    if 'close' in data_dict['prices'].columns:
        prices = data_dict['prices']['close']
    else:
        prices = data_dict['prices']
    
    if date is not None:
        prices = prices.loc[:date]
    
    if len(prices) < max(momentum_days, vol_days) + 1:
        return pd.Series(dtype=float)
    
    # Calculate momentum
    momentum = prices.pct_change(momentum_days).iloc[-1]
    
    # Calculate volatility
    returns = prices.pct_change().dropna()
    volatility = returns.tail(vol_days).std()
    
    # Interaction: momentum / volatility (higher momentum, lower vol = better)
    interaction = momentum / volatility
    interaction = interaction.replace([np.inf, -np.inf], np.nan)
    
    return interaction

def main_two():
    """Enhanced example usage with custom factors"""
    """ This example showcases the code structure to execute multiple backtests inside the same main function"""

    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Initialize components
    data_manager = DataManager('2020-01-01', '2023-12-31')
    factor_calculator = FactorCalculator(data_manager)
    transaction_cost_model = TransactionCostModel()
    risk_model = RiskModel()
    risk_manager = RiskManager(risk_model)
    portfolio_constructor = PortfolioConstructor(transaction_cost_model, risk_model)
    
    # Create backtesting engine
    engine = BacktestingEngine(
        data_manager=data_manager,
        factor_calculator=factor_calculator,
        portfolio_constructor=portfolio_constructor,
        risk_manager=risk_manager,
        transaction_cost_model=transaction_cost_model
    )
    
    # Register custom factors
    print("Registering custom factors...")
    engine.register_custom_factor(calculate_1w_volatility)
    engine.register_custom_factor(calculate_price_gap)
    engine.register_custom_factor(calculate_momentum_vol_interaction)
    
    # List all available factors
    print("\nAvailable factors:")
    available_factors = factor_calculator.custom_registry.list_factors()
    for name, metadata in available_factors.items():
        print(f"  {name}: {metadata['description']}")
    
    # Strategy 1: Pure volatility-based strategy (low vol)
    volatility_strategy_config = {
        'factors': ['volatility_1w'],
        'factor_weights': {'volatility_1w': -1.0},  # Negative weight = prefer low volatility
        'factor_parameters': {
            'volatility_1w': {'lookback_days': 5, 'annualize': True}
        },
        'composite_method': 'weighted_average',
        'portfolio_config': {
            'method': 'quantile',
            'long_pct': 0.2,    # Long top 20% (lowest volatility)
            'short_pct': 0.2,   # Short bottom 20% (highest volatility)
            'max_position_size': 0.05
        }
    }
    
    # Strategy 2: Multi-factor strategy with custom factors
    multifactor_strategy_config = {
        'factors': ['momentum_3m', 'volatility_1w', 'pe_ratio', 'momentum_vol_interaction'],
        'factor_weights': {
            'momentum_3m': 0.3,              # Positive momentum
            'volatility_1w': -0.2,           # Low volatility
            'pe_ratio': -0.2,                # Low P/E (value)
            'momentum_vol_interaction': 0.3   # High momentum/vol ratio
        },
        'factor_parameters': {
            'volatility_1w': {'lookback_days': 5, 'annualize': True},
            'momentum_vol_interaction': {'momentum_days': 21, 'vol_days': 5}
        },
        'composite_method': 'weighted_average',
        'portfolio_config': {
            'method': 'quantile',
            'long_pct': 0.15,
            'short_pct': 0.15,
            'max_position_size': 0.04
        }
    }
    
    # Strategy 3: Custom factor basket with equal weights
    equal_weight_custom_config = {
        'factors': ['volatility_1w', 'price_gap', 'momentum_vol_interaction'],
        'factor_weights': {},  # Empty dict means equal weights
        'factor_parameters': {
            'volatility_1w': {'lookback_days': 7, 'annualize': False},  # Weekly vol, not annualized
            'price_gap': {'lookback_days': 10},
            'momentum_vol_interaction': {'momentum_days': 15, 'vol_days': 7}
        },
        'composite_method': 'equal_weight',
        'portfolio_config': {
            'method': 'quantile',
            'long_pct': 0.25,
            'short_pct': 0.25,
            'max_position_size': 0.03
        }
    }
    
    # Run backtests for different strategies
    strategies = {
        'Low Volatility': volatility_strategy_config,
        'Multi-Factor': multifactor_strategy_config,
        'Equal Weight Custom': equal_weight_custom_config
    }
    
    results = {}
    
    for strategy_name, config in strategies.items():
        print(f"\n{'='*50}")
        print(f"Running backtest for: {strategy_name}")
        print(f"{'='*50}")
        
        try:
            strategy_results = engine.run_backtest(
                start_date='2021-01-01',
                end_date='2023-12-31',
                rebalance_frequency='M',  # Monthly rebalancing
                initial_capital=10000000,  # $10M
                factor_config=config
            )
            
            results[strategy_name] = strategy_results
            
            # Print key metrics
            print(f"\nResults for {strategy_name}:")
            print(f"  Total Return: {strategy_results['total_return']:.2%}")
            print(f"  Annualized Return: {strategy_results['annualized_return']:.2%}")
            print(f"  Volatility: {strategy_results['volatility']:.2%}")
            print(f"  Sharpe Ratio: {strategy_results['sharpe_ratio']:.2f}")
            print(f"  Max Drawdown: {strategy_results['max_drawdown']:.2%}")
            print(f"  Transaction Costs: {strategy_results['transaction_cost_pct']:.2%} of capital")
            
        except Exception as e:
            print(f"Error running {strategy_name}: {str(e)}")
            continue
    
    # Compare strategies
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("STRATEGY COMPARISON")
        print(f"{'='*60}")
        
        comparison_df = pd.DataFrame({
            name: {
                'Total Return': res['total_return'],
                'Annualized Return': res['annualized_return'],
                'Volatility': res['volatility'],
                'Sharpe Ratio': res['sharpe_ratio'],
                'Max Drawdown': res['max_drawdown'],
                'Transaction Costs %': res['transaction_cost_pct']
            }
            for name, res in results.items()
        }).T
        
        print(comparison_df.round(4))
        
        # Find best strategy by Sharpe ratio
        best_strategy = comparison_df['Sharpe Ratio'].idxmax()
        print(f"\nBest strategy by Sharpe ratio: {best_strategy}")
    
    # Demonstrate dynamic factor testing
    print(f"\n{'='*60}")
    print("DYNAMIC FACTOR TESTING")
    print(f"{'='*60}")
    
    # Test different volatility lookback periods
    vol_lookbacks = [3, 5, 7, 10, 15]
    vol_test_results = {}
    
    for lookback in vol_lookbacks:
        config = {
            'factors': ['volatility_1w'],
            'factor_weights': {'volatility_1w': -1.0},
            'factor_parameters': {
                'volatility_1w': {'lookback_days': lookback, 'annualize': True}
            },
            'composite_method': 'weighted_average',
            'portfolio_config': {
                'method': 'quantile',
                'long_pct': 0.2,
                'short_pct': 0.2,
                'max_position_size': 0.05
            }
        }
        
        try:
            test_results = engine.run_backtest(
                start_date='2022-01-01',
                end_date='2023-12-31',
                rebalance_frequency='M',
                initial_capital=10000000,
                factor_config=config
            )
            
            vol_test_results[f'{lookback}d'] = test_results['sharpe_ratio']
            print(f"Volatility {lookback}d lookback - Sharpe: {test_results['sharpe_ratio']:.3f}")
            
        except Exception as e:
            print(f"Error testing {lookback}d volatility: {str(e)}")
    
    if vol_test_results:
        best_vol_lookback = max(vol_test_results.keys(), key=lambda k: vol_test_results[k])
        print(f"\nBest volatility lookback: {best_vol_lookback} (Sharpe: {vol_test_results[best_vol_lookback]:.3f})")


if __name__ == "__main__":

    main_one()
    main_two()



