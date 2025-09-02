-- PostgreSQL Initialization Script for Mimir-Lens Integration
-- Sets up database schema, users, and optimizations

-- ==========================================================================
-- DATABASE SETUP
-- ==========================================================================

-- Create database if it doesn't exist (handled by POSTGRES_DB env var)
-- But we can set database-level settings here

-- ==========================================================================
-- EXTENSIONS
-- ==========================================================================

-- Enable essential extensions
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS btree_gin;
CREATE EXTENSION IF NOT EXISTS btree_gist;
CREATE EXTENSION IF NOT EXISTS uuid-ossp;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ==========================================================================
-- USERS AND PERMISSIONS
-- ==========================================================================

-- Create application user (if not exists)
DO
$do$
BEGIN
   IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'mimir_app') THEN
      CREATE USER mimir_app WITH ENCRYPTED PASSWORD 'app_password_change_me';
   END IF;
END
$do$;

-- Create read-only user for monitoring
DO
$do$
BEGIN
   IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'mimir_readonly') THEN
      CREATE USER mimir_readonly WITH ENCRYPTED PASSWORD 'readonly_password_change_me';
   END IF;
END
$do$;

-- ==========================================================================
-- SCHEMA SETUP
-- ==========================================================================

-- Create schemas for organization
CREATE SCHEMA IF NOT EXISTS mimir_core;
CREATE SCHEMA IF NOT EXISTS mimir_cache;
CREATE SCHEMA IF NOT EXISTS mimir_analytics;
CREATE SCHEMA IF NOT EXISTS lens_integration;

-- Grant schema permissions
GRANT USAGE ON SCHEMA mimir_core TO mimir_app;
GRANT USAGE ON SCHEMA mimir_cache TO mimir_app;
GRANT USAGE ON SCHEMA mimir_analytics TO mimir_app;
GRANT USAGE ON SCHEMA lens_integration TO mimir_app;

GRANT USAGE ON SCHEMA mimir_core TO mimir_readonly;
GRANT USAGE ON SCHEMA mimir_cache TO mimir_readonly;
GRANT USAGE ON SCHEMA mimir_analytics TO mimir_readonly;
GRANT USAGE ON SCHEMA lens_integration TO mimir_readonly;

-- ==========================================================================
-- CORE TABLES
-- ==========================================================================

-- Repositories table
CREATE TABLE IF NOT EXISTS mimir_core.repositories (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    url TEXT NOT NULL,
    branch VARCHAR(100) DEFAULT 'main',
    last_indexed TIMESTAMP WITH TIME ZONE,
    status VARCHAR(50) DEFAULT 'pending',
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexing jobs table
CREATE TABLE IF NOT EXISTS mimir_core.indexing_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    repository_id UUID REFERENCES mimir_core.repositories(id) ON DELETE CASCADE,
    job_type VARCHAR(50) NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- File metadata table
CREATE TABLE IF NOT EXISTS mimir_core.file_metadata (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    repository_id UUID REFERENCES mimir_core.repositories(id) ON DELETE CASCADE,
    file_path TEXT NOT NULL,
    file_type VARCHAR(100),
    size_bytes INTEGER,
    last_modified TIMESTAMP WITH TIME ZONE,
    content_hash VARCHAR(64),
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ==========================================================================
-- CACHE TABLES
-- ==========================================================================

-- Query cache table
CREATE TABLE IF NOT EXISTS mimir_cache.query_cache (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query_hash VARCHAR(64) NOT NULL UNIQUE,
    query_text TEXT NOT NULL,
    response_data JSONB NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    hit_count INTEGER DEFAULT 1,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_accessed TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Session cache table  
CREATE TABLE IF NOT EXISTS mimir_cache.session_cache (
    session_id VARCHAR(128) PRIMARY KEY,
    data JSONB NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ==========================================================================
-- LENS INTEGRATION TABLES
-- ==========================================================================

-- Lens sync status
CREATE TABLE IF NOT EXISTS lens_integration.sync_status (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    repository_id UUID REFERENCES mimir_core.repositories(id) ON DELETE CASCADE,
    lens_index_id VARCHAR(255),
    sync_type VARCHAR(50) NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    last_sync TIMESTAMP WITH TIME ZONE,
    next_sync TIMESTAMP WITH TIME ZONE,
    retry_count INTEGER DEFAULT 0,
    error_message TEXT,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Query performance metrics
CREATE TABLE IF NOT EXISTS lens_integration.query_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query_id VARCHAR(128) NOT NULL,
    query_type VARCHAR(50) NOT NULL,
    response_time_ms INTEGER NOT NULL,
    lens_response_time_ms INTEGER,
    fallback_used BOOLEAN DEFAULT FALSE,
    results_count INTEGER,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ==========================================================================
-- ANALYTICS TABLES
-- ==========================================================================

-- Usage analytics
CREATE TABLE IF NOT EXISTS mimir_analytics.usage_stats (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    endpoint VARCHAR(255) NOT NULL,
    method VARCHAR(10) NOT NULL,
    status_code INTEGER NOT NULL,
    response_time_ms INTEGER NOT NULL,
    user_agent TEXT,
    ip_address INET,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Performance metrics
CREATE TABLE IF NOT EXISTS mimir_analytics.performance_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name VARCHAR(100) NOT NULL,
    metric_value NUMERIC NOT NULL,
    tags JSONB,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ==========================================================================
-- INDEXES FOR PERFORMANCE
-- ==========================================================================

-- Repository indexes
CREATE INDEX IF NOT EXISTS idx_repositories_name ON mimir_core.repositories(name);
CREATE INDEX IF NOT EXISTS idx_repositories_status ON mimir_core.repositories(status);
CREATE INDEX IF NOT EXISTS idx_repositories_last_indexed ON mimir_core.repositories(last_indexed);

-- Indexing jobs indexes
CREATE INDEX IF NOT EXISTS idx_indexing_jobs_repository_id ON mimir_core.indexing_jobs(repository_id);
CREATE INDEX IF NOT EXISTS idx_indexing_jobs_status ON mimir_core.indexing_jobs(status);
CREATE INDEX IF NOT EXISTS idx_indexing_jobs_created_at ON mimir_core.indexing_jobs(created_at);

-- File metadata indexes
CREATE INDEX IF NOT EXISTS idx_file_metadata_repository_id ON mimir_core.file_metadata(repository_id);
CREATE INDEX IF NOT EXISTS idx_file_metadata_file_path ON mimir_core.file_metadata(file_path);
CREATE INDEX IF NOT EXISTS idx_file_metadata_file_type ON mimir_core.file_metadata(file_type);
CREATE INDEX IF NOT EXISTS idx_file_metadata_content_hash ON mimir_core.file_metadata(content_hash);

-- Cache indexes
CREATE INDEX IF NOT EXISTS idx_query_cache_expires_at ON mimir_cache.query_cache(expires_at);
CREATE INDEX IF NOT EXISTS idx_session_cache_expires_at ON mimir_cache.session_cache(expires_at);

-- Lens integration indexes
CREATE INDEX IF NOT EXISTS idx_sync_status_repository_id ON lens_integration.sync_status(repository_id);
CREATE INDEX IF NOT EXISTS idx_sync_status_status ON lens_integration.sync_status(status);
CREATE INDEX IF NOT EXISTS idx_sync_status_next_sync ON lens_integration.sync_status(next_sync);

-- Analytics indexes
CREATE INDEX IF NOT EXISTS idx_usage_stats_endpoint ON mimir_analytics.usage_stats(endpoint);
CREATE INDEX IF NOT EXISTS idx_usage_stats_timestamp ON mimir_analytics.usage_stats(timestamp);
CREATE INDEX IF NOT EXISTS idx_query_metrics_timestamp ON lens_integration.query_metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_timestamp ON mimir_analytics.performance_metrics(timestamp);

-- ==========================================================================
-- FUNCTIONS AND TRIGGERS
-- ==========================================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for updated_at
CREATE TRIGGER update_repositories_updated_at 
    BEFORE UPDATE ON mimir_core.repositories 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_file_metadata_updated_at 
    BEFORE UPDATE ON mimir_core.file_metadata 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_session_cache_updated_at 
    BEFORE UPDATE ON mimir_cache.session_cache 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_sync_status_updated_at 
    BEFORE UPDATE ON lens_integration.sync_status 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ==========================================================================
-- CLEANUP FUNCTIONS
-- ==========================================================================

-- Function to clean up expired cache entries
CREATE OR REPLACE FUNCTION cleanup_expired_cache()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM mimir_cache.query_cache WHERE expires_at < NOW();
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    DELETE FROM mimir_cache.session_cache WHERE expires_at < NOW();
    GET DIAGNOSTICS deleted_count = deleted_count + ROW_COUNT;
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Function to clean up old analytics data
CREATE OR REPLACE FUNCTION cleanup_old_analytics(retention_days INTEGER DEFAULT 90)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM mimir_analytics.usage_stats 
    WHERE timestamp < NOW() - INTERVAL '1 day' * retention_days;
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    DELETE FROM mimir_analytics.performance_metrics 
    WHERE timestamp < NOW() - INTERVAL '1 day' * retention_days;
    GET DIAGNOSTICS deleted_count = deleted_count + ROW_COUNT;
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- ==========================================================================
-- PERMISSIONS
-- ==========================================================================

-- Grant permissions to application user
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA mimir_core TO mimir_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA mimir_cache TO mimir_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA mimir_analytics TO mimir_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA lens_integration TO mimir_app;

GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA mimir_core TO mimir_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA mimir_cache TO mimir_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA mimir_analytics TO mimir_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA lens_integration TO mimir_app;

-- Grant read-only permissions to monitoring user
GRANT SELECT ON ALL TABLES IN SCHEMA mimir_core TO mimir_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA mimir_cache TO mimir_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA mimir_analytics TO mimir_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA lens_integration TO mimir_readonly;

-- ==========================================================================
-- INITIAL DATA
-- ==========================================================================

-- Insert system configuration
INSERT INTO mimir_analytics.performance_metrics (metric_name, metric_value, tags) 
VALUES ('system_init', 1, '{"component": "database", "action": "initialize"}')
ON CONFLICT DO NOTHING;