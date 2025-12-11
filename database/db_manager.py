"""
Database Manager for IntelliTradeAI
Handles all database operations for trades, positions, portfolio, and alerts
"""

import os
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime
from typing import Dict, List, Optional, Any
import uuid


class DatabaseManager:
    """Manages all database operations for the trading system"""
    
    def __init__(self):
        self.db_url = os.getenv('DATABASE_URL')
        self.demo_mode = False
        if not self.db_url:
            self.demo_mode = True
            print("WARNING: DATABASE_URL not set. Running in demo mode without database persistence.")
    
    def get_connection(self):
        """Get database connection"""
        if self.demo_mode:
            return None
        return psycopg2.connect(self.db_url)
    
    def _check_demo_mode(self):
        """Check if in demo mode and return empty data"""
        if self.demo_mode:
            return True
        return False
    
    # ==================== TRADE OPERATIONS ====================
    
    def log_trade(self, trade_data: Dict[str, Any]) -> str:
        """
        Log a new trade to the database
        
        Args:
            trade_data: Dictionary containing trade information
            
        Returns:
            trade_id: Unique identifier for the trade
        """
        trade_id = trade_data.get('trade_id', str(uuid.uuid4()))
        
        if self._check_demo_mode():
            return trade_id
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO trades (
                        trade_id, asset_type, symbol, action, quantity, 
                        entry_price, trading_mode, status, ai_confidence, 
                        ai_signal, notes, fees
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING trade_id
                """, (
                    trade_id,
                    trade_data['asset_type'],
                    trade_data['symbol'],
                    trade_data['action'],
                    trade_data['quantity'],
                    trade_data['entry_price'],
                    trade_data['trading_mode'],
                    trade_data.get('status', 'open'),
                    trade_data.get('ai_confidence'),
                    trade_data.get('ai_signal'),
                    trade_data.get('notes'),
                    trade_data.get('fees', 0)
                ))
                conn.commit()
                return trade_id
    
    def close_trade(self, trade_id: str, exit_price: float, exit_timestamp: Optional[datetime] = None) -> Dict:
        """Close an open trade and calculate P&L"""
        if exit_timestamp is None:
            exit_timestamp = datetime.now()
        
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM trades WHERE trade_id = %s AND status = 'open'
                """, (trade_id,))
                trade = cur.fetchone()
                
                if not trade:
                    raise ValueError(f"No open trade found with ID {trade_id}")
                
                quantity = float(trade['quantity'])
                entry_price = float(trade['entry_price'])
                fees = float(trade['fees'])
                
                if trade['action'] == 'BUY':
                    realized_pnl = (exit_price - entry_price) * quantity - fees
                else:
                    realized_pnl = (entry_price - exit_price) * quantity - fees
                
                cur.execute("""
                    UPDATE trades 
                    SET exit_price = %s, 
                        exit_timestamp = %s, 
                        status = 'closed',
                        realized_pnl = %s,
                        updated_at = NOW()
                    WHERE trade_id = %s
                    RETURNING *
                """, (exit_price, exit_timestamp, realized_pnl, trade_id))
                
                updated_trade = cur.fetchone()
                conn.commit()
                
                if updated_trade is None:
                    raise ValueError(f"Failed to close trade {trade_id}")
                return dict(updated_trade)
    
    def get_all_trades(self, status: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """Get all trades, optionally filtered by status"""
        if self._check_demo_mode():
            return []
        
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if status:
                    cur.execute("""
                        SELECT * FROM trades 
                        WHERE status = %s 
                        ORDER BY timestamp DESC 
                        LIMIT %s
                    """, (status, limit))
                else:
                    cur.execute("""
                        SELECT * FROM trades 
                        ORDER BY timestamp DESC 
                        LIMIT %s
                    """, (limit,))
                
                return [dict(row) for row in cur.fetchall()]
    
    # ==================== POSITION OPERATIONS ====================
    
    def update_position(self, position_data: Dict[str, Any]) -> str:
        """Update or create a position"""
        position_id = position_data.get('position_id', str(uuid.uuid4()))
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO positions (
                        position_id, asset_type, symbol, quantity, 
                        avg_entry_price, current_price, unrealized_pnl,
                        unrealized_pnl_percent, total_invested, current_value, status
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (symbol, asset_type, status) 
                    DO UPDATE SET
                        quantity = EXCLUDED.quantity,
                        avg_entry_price = EXCLUDED.avg_entry_price,
                        current_price = EXCLUDED.current_price,
                        unrealized_pnl = EXCLUDED.unrealized_pnl,
                        unrealized_pnl_percent = EXCLUDED.unrealized_pnl_percent,
                        total_invested = EXCLUDED.total_invested,
                        current_value = EXCLUDED.current_value,
                        updated_at = NOW()
                    RETURNING position_id
                """, (
                    position_id,
                    position_data['asset_type'],
                    position_data['symbol'],
                    position_data['quantity'],
                    position_data['avg_entry_price'],
                    position_data.get('current_price'),
                    position_data.get('unrealized_pnl'),
                    position_data.get('unrealized_pnl_percent'),
                    position_data.get('total_invested'),
                    position_data.get('current_value'),
                    position_data.get('status', 'active')
                ))
                conn.commit()
                return position_id
    
    def get_active_positions(self) -> List[Dict]:
        """Get all active positions"""
        if self._check_demo_mode():
            return []
        
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM positions 
                    WHERE status = 'active' 
                    ORDER BY symbol
                """)
                return [dict(row) for row in cur.fetchall()]
    
    def close_position(self, symbol: str, asset_type: str) -> Dict:
        """Close a position"""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    UPDATE positions 
                    SET status = 'closed', updated_at = NOW()
                    WHERE symbol = %s AND asset_type = %s AND status = 'active'
                    RETURNING *
                """, (symbol, asset_type))
                result = cur.fetchone()
                conn.commit()
                return dict(result) if result else {}
    
    # ==================== PORTFOLIO OPERATIONS ====================
    
    def get_portfolio(self, user_id: str = 'default_user') -> Dict:
        """Get portfolio summary"""
        if self._check_demo_mode():
            return {
                'user_id': user_id,
                'total_value': 10000.0,
                'cash_balance': 10000.0,
                'crypto_balance': 0.0,
                'stock_balance': 0.0,
                'options_balance': 0.0,
                'total_realized_pnl': 0.0,
                'total_unrealized_pnl': 0.0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0
            }
        
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM portfolio WHERE user_id = %s
                """, (user_id,))
                result = cur.fetchone()
                return dict(result) if result else {}
    
    def update_portfolio(self, portfolio_data: Dict[str, Any], user_id: str = 'default_user'):
        """Update portfolio statistics"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE portfolio 
                    SET total_value = %s,
                        cash_balance = %s,
                        crypto_balance = %s,
                        stock_balance = %s,
                        options_balance = %s,
                        total_realized_pnl = %s,
                        total_unrealized_pnl = %s,
                        total_trades = %s,
                        winning_trades = %s,
                        losing_trades = %s,
                        win_rate = %s,
                        last_updated = NOW()
                    WHERE user_id = %s
                """, (
                    portfolio_data.get('total_value'),
                    portfolio_data.get('cash_balance'),
                    portfolio_data.get('crypto_balance', 0),
                    portfolio_data.get('stock_balance', 0),
                    portfolio_data.get('options_balance', 0),
                    portfolio_data.get('total_realized_pnl', 0),
                    portfolio_data.get('total_unrealized_pnl', 0),
                    portfolio_data.get('total_trades', 0),
                    portfolio_data.get('winning_trades', 0),
                    portfolio_data.get('losing_trades', 0),
                    portfolio_data.get('win_rate', 0),
                    user_id
                ))
                conn.commit()
    
    # ==================== ALERT OPERATIONS ====================
    
    def create_alert(self, alert_data: Dict[str, Any]) -> str:
        """Create a price alert"""
        alert_id = alert_data.get('alert_id', str(uuid.uuid4()))
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO trade_alerts (
                        alert_id, symbol, asset_type, alert_type, 
                        target_price, current_price, action, quantity, status
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING alert_id
                """, (
                    alert_id,
                    alert_data['symbol'],
                    alert_data['asset_type'],
                    alert_data['alert_type'],
                    alert_data['target_price'],
                    alert_data.get('current_price'),
                    alert_data['action'],
                    alert_data['quantity'],
                    alert_data.get('status', 'active')
                ))
                conn.commit()
                return alert_id
    
    def get_active_alerts(self) -> List[Dict]:
        """Get all active alerts"""
        if self._check_demo_mode():
            return []
        
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM trade_alerts 
                    WHERE status = 'active' 
                    ORDER BY created_at DESC
                """)
                return [dict(row) for row in cur.fetchall()]
    
    def trigger_alert(self, alert_id: str) -> Dict:
        """Mark an alert as triggered"""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    UPDATE trade_alerts 
                    SET status = 'triggered', 
                        triggered_at = NOW(),
                        updated_at = NOW()
                    WHERE alert_id = %s
                    RETURNING *
                """, (alert_id,))
                result = cur.fetchone()
                conn.commit()
                return dict(result) if result else {}
    
    # ==================== OPTIONS OPERATIONS ====================
    
    def update_options_chain(self, options_data: List[Dict[str, Any]]):
        """Bulk update options chain data"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                for option in options_data:
                    cur.execute("""
                        INSERT INTO options_chains (
                            symbol, expiration_date, strike_price, option_type,
                            bid, ask, last_price, volume, open_interest,
                            implied_volatility, delta, gamma, theta, vega, rho,
                            in_the_money
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (symbol, expiration_date, strike_price, option_type)
                        DO UPDATE SET
                            bid = EXCLUDED.bid,
                            ask = EXCLUDED.ask,
                            last_price = EXCLUDED.last_price,
                            volume = EXCLUDED.volume,
                            open_interest = EXCLUDED.open_interest,
                            implied_volatility = EXCLUDED.implied_volatility,
                            delta = EXCLUDED.delta,
                            gamma = EXCLUDED.gamma,
                            theta = EXCLUDED.theta,
                            vega = EXCLUDED.vega,
                            rho = EXCLUDED.rho,
                            in_the_money = EXCLUDED.in_the_money,
                            updated_at = NOW()
                    """, (
                        option['symbol'],
                        option['expiration_date'],
                        option['strike_price'],
                        option['option_type'],
                        option.get('bid'),
                        option.get('ask'),
                        option.get('last_price'),
                        option.get('volume'),
                        option.get('open_interest'),
                        option.get('implied_volatility'),
                        option.get('delta'),
                        option.get('gamma'),
                        option.get('theta'),
                        option.get('vega'),
                        option.get('rho'),
                        option.get('in_the_money')
                    ))
                conn.commit()
    
    def get_options_chain(self, symbol: str, expiration_date: Optional[str] = None) -> List[Dict]:
        """Get options chain for a symbol"""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if expiration_date:
                    cur.execute("""
                        SELECT * FROM options_chains 
                        WHERE symbol = %s AND expiration_date = %s
                        ORDER BY strike_price, option_type
                    """, (symbol, expiration_date))
                else:
                    cur.execute("""
                        SELECT * FROM options_chains 
                        WHERE symbol = %s
                        ORDER BY expiration_date, strike_price, option_type
                    """, (symbol,))
                
                return [dict(row) for row in cur.fetchall()]
    
    # ==================== WALLET OPERATIONS ====================
    
    def add_wallet(self, wallet_address: str, blockchain: str) -> str:
        """Add a crypto wallet"""
        wallet_id = str(uuid.uuid4())
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO crypto_wallets (wallet_id, wallet_address, blockchain)
                    VALUES (%s, %s, %s)
                    RETURNING wallet_id
                """, (wallet_id, wallet_address, blockchain))
                conn.commit()
                return wallet_id
    
    def get_active_wallets(self) -> List[Dict]:
        """Get all active wallets"""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM crypto_wallets 
                    WHERE is_active = true
                    ORDER BY created_at DESC
                """)
                return [dict(row) for row in cur.fetchall()]
    
    def update_wallet_balance(self, wallet_address: str, balance: float):
        """Update wallet balance"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE crypto_wallets 
                    SET balance = %s, updated_at = NOW()
                    WHERE wallet_address = %s
                """, (balance, wallet_address))
                conn.commit()
    
    # ==================== USER PROFILE OPERATIONS ====================
    
    def save_user_profile(self, user_id: str, profile_data: Dict[str, Any]) -> str:
        """Save or update user profile with risk assessment and trading plan"""
        import json
        
        if self._check_demo_mode():
            return user_id
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO user_profiles (
                        user_id, investment_amount, investment_tier, risk_level,
                        risk_scores, trading_plan, preferences, auto_trade_settings,
                        onboarding_completed
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (user_id)
                    DO UPDATE SET
                        investment_amount = EXCLUDED.investment_amount,
                        investment_tier = EXCLUDED.investment_tier,
                        risk_level = EXCLUDED.risk_level,
                        risk_scores = EXCLUDED.risk_scores,
                        trading_plan = EXCLUDED.trading_plan,
                        preferences = EXCLUDED.preferences,
                        auto_trade_settings = EXCLUDED.auto_trade_settings,
                        onboarding_completed = EXCLUDED.onboarding_completed,
                        updated_at = NOW()
                    RETURNING user_id
                """, (
                    user_id,
                    profile_data.get('investment_amount', 0),
                    profile_data.get('investment_tier', 'Starter'),
                    profile_data.get('risk_level', 'moderate'),
                    json.dumps(profile_data.get('assessment_scores', [])),
                    json.dumps(profile_data),
                    json.dumps(profile_data.get('preferences', {})),
                    json.dumps(profile_data.get('auto_trade_settings', {})),
                    profile_data.get('onboarding_completed', True)
                ))
                conn.commit()
                return user_id
    
    def get_user_profile(self, user_id: str) -> Optional[Dict]:
        """Get user profile with trading plan"""
        import json
        
        if self._check_demo_mode():
            return None
        
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM user_profiles WHERE user_id = %s
                """, (user_id,))
                result = cur.fetchone()
                if result:
                    profile = dict(result)
                    if profile.get('trading_plan'):
                        profile['trading_plan'] = json.loads(profile['trading_plan'])
                    if profile.get('preferences'):
                        profile['preferences'] = json.loads(profile['preferences'])
                    if profile.get('auto_trade_settings'):
                        profile['auto_trade_settings'] = json.loads(profile['auto_trade_settings'])
                    if profile.get('risk_scores'):
                        profile['risk_scores'] = json.loads(profile['risk_scores'])
                    return profile
                return None
    
    def save_esignature_record(self, signature_data: Dict[str, Any]) -> str:
        """Save e-signature record for compliance audit trail"""
        import json
        
        signature_id = str(uuid.uuid4())
        
        if self._check_demo_mode():
            return signature_id
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO esignature_records (
                        signature_id, user_id, user_email, agreement_type,
                        agreement_hash, trading_config, ip_address, signed_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING signature_id
                """, (
                    signature_id,
                    signature_data.get('user_name', 'unknown'),
                    signature_data.get('user_email', 'unknown'),
                    'automatic_trading_authorization',
                    signature_data.get('agreement_hash', ''),
                    json.dumps(signature_data.get('trading_config', {})),
                    signature_data.get('ip_address', 'unknown'),
                    signature_data.get('signed_at', datetime.now().isoformat())
                ))
                conn.commit()
                return signature_id
    
    def get_esignature_records(self, user_id: str) -> List[Dict]:
        """Get all e-signature records for a user"""
        if self._check_demo_mode():
            return []
        
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM esignature_records 
                    WHERE user_id = %s 
                    ORDER BY signed_at DESC
                """, (user_id,))
                return [dict(row) for row in cur.fetchall()]
    
    def has_active_auto_trading_consent(self, user_id: str) -> bool:
        """Check if user has active automatic trading consent"""
        if self._check_demo_mode():
            return False
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT COUNT(*) FROM esignature_records 
                    WHERE user_id = %s 
                    AND agreement_type = 'automatic_trading_authorization'
                    AND revoked_at IS NULL
                """, (user_id,))
                count = cur.fetchone()[0]
                return count > 0
    
    def revoke_auto_trading_consent(self, user_id: str) -> bool:
        """Revoke automatic trading consent"""
        if self._check_demo_mode():
            return True
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE esignature_records 
                    SET revoked_at = NOW()
                    WHERE user_id = %s 
                    AND agreement_type = 'automatic_trading_authorization'
                    AND revoked_at IS NULL
                """, (user_id,))
                conn.commit()
                return True
