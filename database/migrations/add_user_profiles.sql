-- Migration to add user profiles and e-signature tables
-- Run this migration to add support for user onboarding and legal compliance

-- User Profiles Table
CREATE TABLE IF NOT EXISTS user_profiles (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) UNIQUE NOT NULL,
    investment_amount DECIMAL(15, 2) DEFAULT 0,
    investment_tier VARCHAR(50) DEFAULT 'Starter',
    risk_level VARCHAR(50) DEFAULT 'moderate',
    risk_scores JSONB,
    trading_plan JSONB,
    preferences JSONB,
    auto_trade_settings JSONB,
    onboarding_completed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- E-Signature Records Table for Compliance Audit Trail
CREATE TABLE IF NOT EXISTS esignature_records (
    id SERIAL PRIMARY KEY,
    signature_id VARCHAR(255) UNIQUE NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    user_email VARCHAR(255) NOT NULL,
    agreement_type VARCHAR(100) NOT NULL,
    agreement_hash VARCHAR(64) NOT NULL,
    trading_config JSONB,
    ip_address VARCHAR(45),
    signed_at TIMESTAMP NOT NULL,
    revoked_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create indices for faster lookups
CREATE INDEX IF NOT EXISTS idx_user_profiles_user_id ON user_profiles(user_id);
CREATE INDEX IF NOT EXISTS idx_esignature_user_id ON esignature_records(user_id);
CREATE INDEX IF NOT EXISTS idx_esignature_agreement_type ON esignature_records(agreement_type);
