#!/usr/bin/env python3
"""
Snowflake Deployment Script for Enterprise Data Warehouse

This script handles the deployment of Snowflake database objects including:
- Databases and schemas
- Warehouses
- Roles and permissions
- Audit tables
- Monitoring views

Usage:
    python deploy_snowflake_objects.py --env [dev|test|prod] [--dry-run]
"""

import os
import sys
import argparse
import logging
import yaml
from pathlib import Path
from typing import Dict, List, Optional
import snowflake.connector
from snowflake.connector import DictCursor
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SnowflakeDeployer:
    """Handles deployment of Snowflake objects for the data warehouse"""
    
    def __init__(self, environment: str = 'dev'):
        self.environment = environment
        self.connection = None
        self.config = self._load_deployment_config()
        self.env_config = self.config['environments'][environment]
        
    def _load_deployment_config(self) -> Dict:
        """Load deployment configuration"""
        config_path = Path('governance') / 'deployment_config.yml'
        
        # Default configuration if file doesn't exist
        if not config_path.exists():
            return self._get_default_config()
            
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _get_default_config(self) -> Dict:
        """Get default deployment configuration"""
        return {
            'environments': {
                'dev': {
                    'database': 'DEV_DW',
                    'warehouse': 'DEV_WH',
                    'schemas': ['STAGING', 'MARTS', 'REFERENCE_DATA', 'ANALYTICS', 'AUDIT'],
                    'warehouse_size': 'SMALL'
                },
                'test': {
                    'database': 'TEST_DW',
                    'warehouse': 'TEST_WH', 
                    'schemas': ['STAGING', 'MARTS', 'REFERENCE_DATA', 'ANALYTICS', 'AUDIT'],
                    'warehouse_size': 'MEDIUM'
                },
                'prod': {
                    'database': 'PROD_DW',
                    'warehouse': 'PROD_WH',
                    'schemas': ['STAGING', 'MARTS', 'REFERENCE_DATA', 'ANALYTICS', 'AUDIT'],
                    'warehouse_size': 'LARGE'
                }
            },
            'roles': {
                'DATA_ENGINEER': {
                    'permissions': ['ALL'],
                    'databases': ['ALL']
                },
                'DATA_ANALYST': {
                    'permissions': ['SELECT', 'REFERENCES'],
                    'databases': ['ALL'],
                    'schemas': ['MARTS', 'REFERENCE_DATA', 'ANALYTICS']
                },
                'DATA_SCIENTIST': {
                    'permissions': ['SELECT', 'REFERENCES'],
                    'databases': ['ALL'],
                    'schemas': ['MARTS', 'REFERENCE_DATA', 'ANALYTICS']
                }
            }
        }
    
    def connect(self) -> None:
        """Establish connection to Snowflake"""
        try:
            self.connection = snowflake.connector.connect(
                account=os.getenv('SNOWFLAKE_ACCOUNT'),
                user=os.getenv('SNOWFLAKE_USER'),
                password=os.getenv('SNOWFLAKE_PASSWORD'),
                role=os.getenv('SNOWFLAKE_ROLE', 'SYSADMIN'),
                warehouse=os.getenv('SNOWFLAKE_WAREHOUSE', 'COMPUTE_WH')
            )
            logger.info(f"Connected to Snowflake account: {os.getenv('SNOWFLAKE_ACCOUNT')}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Snowflake: {e}")
            sys.exit(1)
    
    def disconnect(self) -> None:
        """Close Snowflake connection"""
        if self.connection:
            self.connection.close()
            logger.info("Disconnected from Snowflake")
    
    def execute_sql(self, sql: str, description: str = "", dry_run: bool = False) -> Optional[List[Dict]]:
        """Execute SQL with error handling"""
        if dry_run:
            logger.info(f"[DRY-RUN] Would execute: {description}")
            logger.debug(f"SQL: {sql}")
            return None
            
        try:
            cursor = self.connection.cursor(DictCursor)
            cursor.execute(sql)
            results = cursor.fetchall()
            logger.info(f"Successfully executed: {description}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to execute {description}: {e}")
            logger.debug(f"SQL: {sql}")
            raise
    
    def create_database(self, dry_run: bool = False) -> None:
        """Create main database"""
        database_name = self.env_config['database']
        
        sql = f"""
        CREATE DATABASE IF NOT EXISTS {database_name}
        COMMENT = 'Enterprise Data Warehouse - {self.environment.upper()} Environment'
        """
        
        self.execute_sql(sql, f"Create database {database_name}", dry_run)
    
    def create_warehouse(self, dry_run: bool = False) -> None:
        """Create compute warehouse"""
        warehouse_name = self.env_config['warehouse']
        warehouse_size = self.env_config['warehouse_size']
        
        sql = f"""
        CREATE WAREHOUSE IF NOT EXISTS {warehouse_name}
        WITH 
            WAREHOUSE_SIZE = '{warehouse_size}'
            AUTO_SUSPEND = 300
            AUTO_RESUME = TRUE
            INITIALLY_SUSPENDED = TRUE
        COMMENT = 'Data Warehouse compute for {self.environment.upper()}'
        """
        
        self.execute_sql(sql, f"Create warehouse {warehouse_name}", dry_run)
    
    def create_schemas(self, dry_run: bool = False) -> None:
        """Create schemas within the database"""
        database_name = self.env_config['database']
        
        for schema_name in self.env_config['schemas']:
            sql = f"""
            CREATE SCHEMA IF NOT EXISTS {database_name}.{schema_name}
            COMMENT = 'Schema for {schema_name.lower().replace("_", " ")} objects'
            """
            
            self.execute_sql(sql, f"Create schema {schema_name}", dry_run)
    
    def create_roles(self, dry_run: bool = False) -> None:
        """Create roles and permissions"""
        for role_name, role_config in self.config['roles'].items():
            # Create role
            sql = f"""
            CREATE ROLE IF NOT EXISTS {role_name}
            COMMENT = 'Role for {role_name.lower().replace("_", " ")}'
            """
            
            self.execute_sql(sql, f"Create role {role_name}", dry_run)
    
    def grant_permissions(self, dry_run: bool = False) -> None:
        """Grant permissions to roles"""
        database_name = self.env_config['database']
        warehouse_name = self.env_config['warehouse']
        
        for role_name, role_config in self.config['roles'].items():
            # Grant warehouse usage
            sql = f"GRANT USAGE ON WAREHOUSE {warehouse_name} TO ROLE {role_name}"
            self.execute_sql(sql, f"Grant warehouse usage to {role_name}", dry_run)
            
            # Grant database permissions
            permissions = role_config['permissions']
            allowed_schemas = role_config.get('schemas', self.env_config['schemas'])
            
            if 'ALL' in permissions:
                sql = f"GRANT ALL ON DATABASE {database_name} TO ROLE {role_name}"
                self.execute_sql(sql, f"Grant all database permissions to {role_name}", dry_run)
                
                for schema_name in allowed_schemas:
                    sql = f"GRANT ALL ON SCHEMA {database_name}.{schema_name} TO ROLE {role_name}"
                    self.execute_sql(sql, f"Grant all schema permissions to {role_name}", dry_run)
                    
                    sql = f"GRANT ALL ON ALL TABLES IN SCHEMA {database_name}.{schema_name} TO ROLE {role_name}"
                    self.execute_sql(sql, f"Grant all table permissions to {role_name}", dry_run)
                    
                    sql = f"GRANT ALL ON FUTURE TABLES IN SCHEMA {database_name}.{schema_name} TO ROLE {role_name}"
                    self.execute_sql(sql, f"Grant future table permissions to {role_name}", dry_run)
            else:
                # Grant specific permissions
                sql = f"GRANT USAGE ON DATABASE {database_name} TO ROLE {role_name}"
                self.execute_sql(sql, f"Grant database usage to {role_name}", dry_run)
                
                for schema_name in allowed_schemas:
                    sql = f"GRANT USAGE ON SCHEMA {database_name}.{schema_name} TO ROLE {role_name}"
                    self.execute_sql(sql, f"Grant schema usage to {role_name}", dry_run)
                    
                    for permission in permissions:
                        if permission in ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'REFERENCES']:
                            sql = f"GRANT {permission} ON ALL TABLES IN SCHEMA {database_name}.{schema_name} TO ROLE {role_name}"
                            self.execute_sql(sql, f"Grant {permission} to {role_name}", dry_run)
                            
                            sql = f"GRANT {permission} ON FUTURE TABLES IN SCHEMA {database_name}.{schema_name} TO ROLE {role_name}"
                            self.execute_sql(sql, f"Grant future {permission} to {role_name}", dry_run)
    
    def create_audit_tables(self, dry_run: bool = False) -> None:
        """Create audit and monitoring tables"""
        database_name = self.env_config['database']
        
        # DBT run log table
        sql = f"""
        CREATE TABLE IF NOT EXISTS {database_name}.AUDIT.dbt_run_log (
            run_id VARCHAR(255),
            run_started_at TIMESTAMP_NTZ,
            run_completed_at TIMESTAMP_NTZ,
            invocation_id VARCHAR(255),
            target_name VARCHAR(100),
            model_count INTEGER,
            test_count INTEGER,
            success_count INTEGER,
            error_count INTEGER,
            skipped_count INTEGER,
            status VARCHAR(20),
            created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
        )
        """
        
        self.execute_sql(sql, "Create dbt_run_log table", dry_run)
        
        # Fact table statistics
        sql = f"""
        CREATE TABLE IF NOT EXISTS {database_name}.AUDIT.fact_table_stats (
            table_name VARCHAR(255),
            record_count BIGINT,
            valid_records BIGINT,
            invalid_records BIGINT,
            freshness_status VARCHAR(20),
            run_timestamp TIMESTAMP_NTZ,
            batch_id VARCHAR(255),
            created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
        )
        """
        
        self.execute_sql(sql, "Create fact_table_stats table", dry_run)
        
        # SCD change log
        sql = f"""
        CREATE TABLE IF NOT EXISTS {database_name}.AUDIT.scd_change_log (
            table_name VARCHAR(255),
            total_records BIGINT,
            current_records BIGINT,
            historical_records BIGINT,
            new_records BIGINT,
            updated_records BIGINT,
            run_timestamp TIMESTAMP_NTZ,
            batch_id VARCHAR(255),
            created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
        )
        """
        
        self.execute_sql(sql, "Create scd_change_log table", dry_run)
        
        # Data quality scores
        sql = f"""
        CREATE TABLE IF NOT EXISTS {database_name}.AUDIT.data_quality_scores (
            table_name VARCHAR(255),
            column_name VARCHAR(255),
            test_name VARCHAR(255),
            test_result VARCHAR(20),
            test_value FLOAT,
            threshold_value FLOAT,
            run_timestamp TIMESTAMP_NTZ,
            batch_id VARCHAR(255),
            created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
        )
        """
        
        self.execute_sql(sql, "Create data_quality_scores table", dry_run)
    
    def create_monitoring_views(self, dry_run: bool = False) -> None:
        """Create monitoring and reporting views"""
        database_name = self.env_config['database']
        
        # Data freshness monitoring view
        sql = f"""
        CREATE OR REPLACE VIEW {database_name}.ANALYTICS.v_data_freshness AS
        SELECT 
            table_name,
            max(run_timestamp) as last_update,
            datediff('hour', max(run_timestamp), current_timestamp()) as hours_since_update,
            case 
                when datediff('hour', max(run_timestamp), current_timestamp()) <= 2 then 'FRESH'
                when datediff('hour', max(run_timestamp), current_timestamp()) <= 24 then 'ACCEPTABLE'
                else 'STALE'
            end as freshness_status
        FROM {database_name}.AUDIT.fact_table_stats
        GROUP BY table_name
        """
        
        self.execute_sql(sql, "Create data freshness view", dry_run)
        
        # Data quality dashboard view
        sql = f"""
        CREATE OR REPLACE VIEW {database_name}.ANALYTICS.v_data_quality_dashboard AS
        SELECT 
            table_name,
            count(*) as total_tests,
            sum(case when test_result = 'PASS' then 1 else 0 end) as passed_tests,
            sum(case when test_result = 'FAIL' then 1 else 0 end) as failed_tests,
            round(100.0 * sum(case when test_result = 'PASS' then 1 else 0 end) / count(*), 2) as quality_score,
            max(run_timestamp) as last_tested
        FROM {database_name}.AUDIT.data_quality_scores
        GROUP BY table_name
        """
        
        self.execute_sql(sql, "Create data quality dashboard view", dry_run)
        
        # DBT run summary view
        sql = f"""
        CREATE OR REPLACE VIEW {database_name}.ANALYTICS.v_dbt_run_summary AS
        SELECT 
            target_name,
            status,
            count(*) as run_count,
            avg(datediff('minute', run_started_at, run_completed_at)) as avg_duration_minutes,
            max(run_completed_at) as last_run,
            sum(model_count) as total_models_processed,
            sum(success_count) as total_successful,
            sum(error_count) as total_errors
        FROM {database_name}.AUDIT.dbt_run_log
        WHERE run_started_at >= dateadd('day', -7, current_date())
        GROUP BY target_name, status
        """
        
        self.execute_sql(sql, "Create DBT run summary view", dry_run)
    
    def create_stored_procedures(self, dry_run: bool = False) -> None:
        """Create utility stored procedures"""
        database_name = self.env_config['database']
        
        # Procedure to refresh all tables
        sql = f"""
        CREATE OR REPLACE PROCEDURE {database_name}.ANALYTICS.refresh_all_tables()
        RETURNS STRING
        LANGUAGE JAVASCRIPT
        EXECUTE AS CALLER
        AS
        $$
        var result = "";
        var tables = [];
        
        // Get list of all fact and dimension tables
        var sql_facts = "SHOW TABLES IN SCHEMA " + "{database_name}.MARTS" + " LIKE 'FACT_%'";
        var stmt_facts = snowflake.createStatement({{sqlText: sql_facts}});
        var rs_facts = stmt_facts.execute();
        
        while (rs_facts.next()) {{
            tables.push(rs_facts.getColumnValue(2)); // table name
        }}
        
        var sql_dims = "SHOW TABLES IN SCHEMA " + "{database_name}.MARTS" + " LIKE 'DIM_%'";
        var stmt_dims = snowflake.createStatement({{sqlText: sql_dims}});
        var rs_dims = stmt_dims.execute();
        
        while (rs_dims.next()) {{
            tables.push(rs_dims.getColumnValue(2)); // table name
        }}
        
        // Refresh each table
        for (var i = 0; i < tables.length; i++) {{
            try {{
                var refresh_sql = "ALTER TABLE " + "{database_name}.MARTS." + tables[i] + " REFRESH";
                var refresh_stmt = snowflake.createStatement({{sqlText: refresh_sql}});
                refresh_stmt.execute();
                result += "Refreshed " + tables[i] + "\\n";
            }} catch (err) {{
                result += "Failed to refresh " + tables[i] + ": " + err.message + "\\n";
            }}
        }}
        
        return result;
        $$
        """
        
        self.execute_sql(sql, "Create refresh_all_tables procedure", dry_run)
    
    def validate_deployment(self, dry_run: bool = False) -> None:
        """Validate the deployment by checking created objects"""
        if dry_run:
            logger.info("[DRY-RUN] Would validate deployment")
            return
            
        database_name = self.env_config['database']
        
        # Check database exists
        sql = f"SHOW DATABASES LIKE '{database_name}'"
        results = self.execute_sql(sql, "Validate database exists")
        
        if not results:
            raise Exception(f"Database {database_name} was not created successfully")
        
        # Check schemas exist
        for schema_name in self.env_config['schemas']:
            sql = f"SHOW SCHEMAS IN DATABASE {database_name} LIKE '{schema_name}'"
            results = self.execute_sql(sql, f"Validate schema {schema_name} exists")
            
            if not results:
                raise Exception(f"Schema {schema_name} was not created successfully")
        
        # Check warehouse exists
        warehouse_name = self.env_config['warehouse']
        sql = f"SHOW WAREHOUSES LIKE '{warehouse_name}'"
        results = self.execute_sql(sql, "Validate warehouse exists")
        
        if not results:
            raise Exception(f"Warehouse {warehouse_name} was not created successfully")
        
        logger.info("Deployment validation completed successfully")
    
    def deploy(self, dry_run: bool = False) -> None:
        """Execute full deployment"""
        logger.info(f"Starting deployment to {self.environment.upper()} environment")
        
        if not dry_run:
            self.connect()
        
        try:
            # Core infrastructure
            self.create_database(dry_run)
            self.create_warehouse(dry_run) 
            self.create_schemas(dry_run)
            
            # Security
            self.create_roles(dry_run)
            self.grant_permissions(dry_run)
            
            # Monitoring and audit
            self.create_audit_tables(dry_run)
            self.create_monitoring_views(dry_run)
            self.create_stored_procedures(dry_run)
            
            # Validation
            self.validate_deployment(dry_run)
            
            logger.info(f"Deployment to {self.environment.upper()} completed successfully!")
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            raise
        finally:
            if not dry_run:
                self.disconnect()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Deploy Snowflake objects for enterprise data warehouse')
    parser.add_argument('--env', choices=['dev', 'test', 'prod'], default='dev',
                       help='Target environment for deployment')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be deployed without executing')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate required environment variables
    required_vars = ['SNOWFLAKE_ACCOUNT', 'SNOWFLAKE_USER', 'SNOWFLAKE_PASSWORD']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars and not args.dry_run:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        sys.exit(1)
    
    try:
        deployer = SnowflakeDeployer(args.env)
        deployer.deploy(args.dry_run)
        
    except Exception as e:
        logger.error(f"Deployment script failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
