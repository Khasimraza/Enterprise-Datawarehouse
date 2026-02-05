#!/usr/bin/env python3
"""
Enterprise Data Warehouse Model Generator

This script automatically generates DBT models for fact and dimension tables
based on configuration defined in governance/data_catalog.yml

Usage:
    python generate_fact_dimension_models.py [--type facts|dimensions] [--config path]
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from jinja2 import Template, Environment, FileSystemLoader
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataWarehouseModelGenerator:
    """Generates DBT models for enterprise data warehouse"""
    
    def __init__(self, config_path: str = "governance/data_catalog.yml"):
        """Initialize the generator with configuration"""
        self.config_path = Path(config_path)
        self.load_config()
        self.setup_jinja_env()
        
    def load_config(self):
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            self.fact_tables = self.config.get('fact_tables', {})
            self.dimension_tables = self.config.get('dimension_tables', {})
            
            logger.info(f"Loaded config: {len(self.fact_tables)} fact tables, {len(self.dimension_tables)} dimension tables")
            
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            sys.exit(1)
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            sys.exit(1)
    
    def setup_jinja_env(self):
        """Setup Jinja2 environment for template rendering"""
        template_dir = Path(__file__).parent / "templates"
        template_dir.mkdir(exist_ok=True)
        
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
    def generate_fact_table_model(self, table_name: str, config: Dict) -> str:
        """Generate DBT model for fact table"""
        
        template_str = '''
{#- Fact table template with comprehensive functionality -#}
{{ config(
    materialized='incremental',
    incremental_strategy='merge',
    unique_key={{ config.unique_key | tojson }},
    cluster_by={{ config.get('cluster_by', ['date_key']) | tojson }},
    tags={{ config.tags | tojson }},
    pre_hook="call system$log_info('Starting {{ table_name }} processing')",
    post_hook=[
        "{{ log_fact_stats() }}",
        "{{ validate_fact_table() }}",
        "call system$log_info('Completed {{ table_name }} processing')"
    ]
) }}

{%- set business_keys = config.business_keys -%}
{%- set measures = config.measures -%}
{%- set dimension_refs = config.dimension_references -%}

-- Generated fact table: {{ table_name }}
-- Description: {{ config.description }}
-- Generated on: {{ generation_timestamp }}

with source_data as (
    select
        -- Business keys
        {%- for key in business_keys %}
        {{ key }},
        {%- endfor %}
        
        -- Dimension foreign keys
        {%- for dim_ref in dimension_refs %}
        {{ dim_ref.column }} as {{ dim_ref.name }}_key,
        {%- endfor %}
        
        -- Measures with data type casting
        {%- for measure in measures %}
        {%- if measure.data_type %}
        cast({{ measure.column }} as {{ measure.data_type }}) as {{ measure.name or measure.column }},
        {%- else %}
        {{ measure.column }}{% if measure.name %} as {{ measure.name }}{% endif %},
        {%- endif %}
        {%- endfor %}
        
        -- Audit and metadata columns
        {%- if config.get('source_updated_column') %}
        {{ config.source_updated_column }} as source_updated_at,
        {%- endif %}
        current_timestamp() as created_at,
        current_timestamp() as updated_at,
        '{{ "{{ invocation_id }}" }}' as batch_id,
        '{{ config.get("grain", "daily") }}' as grain_level
        
    from {{ "{{ ref('" + config.source_table + "') }}" }}
    
    {%- if config.get('where_clause') %}
    where {{ config.where_clause }}
    {%- endif %}
    
    {% raw %}
    {% if is_incremental() %}
        {% if config.get('source_updated_column') %}
        and {{ config.source_updated_column }} > (
            select coalesce(max(source_updated_at), '1900-01-01'::timestamp) 
            from {{ this }}
        )
        {% else %}
        and created_at > (
            select coalesce(max(created_at), '1900-01-01'::timestamp) 
            from {{ this }}
        )
        {% endif %}
    {% endif %}
    {% endraw %}
),

data_quality_layer as (
    select 
        *,
        -- Data quality scoring
        case 
            when {%- for key in business_keys %}{{ key }} is not null{% if not loop.last %} and {% endif %}{% endfor %}
            {%- for measure in measures %}
            {%- if measure.get('required', False) %}
             and {{ measure.name or measure.column }} is not null
            {%- endif %}
            {%- endfor %}
            then 'VALID'
            else 'INVALID'
        end as data_quality_status,
        
        -- Completeness indicators
        {%- for measure in measures %}
        case when {{ measure.name or measure.column }} is not null then 1 else 0 end as {{ measure.name or measure.column }}_complete,
        {%- endfor %}
        
        -- Row hash for change detection
        {{ "{{ dbt_utils.generate_surrogate_key(" }}{{ (business_keys + [m.name or m.column for m in measures]) | tojson }}{{ ") }}" }} as row_hash
        
    from source_data
),

business_logic_layer as (
    select 
        *,
        
        {%- if config.get('derived_measures') %}
        -- Derived measures
        {%- for derived in config.derived_measures %}
        {{ derived.calculation }} as {{ derived.name }},
        {%- endfor %}
        {%- endif %}
        
        {%- if config.get('business_rules') %}
        -- Business rules
        {%- for rule in config.business_rules %}
        {{ rule.logic }} as {{ rule.name }},
        {%- endfor %}
        {%- endif %}
        
        -- Standard derived fields
        {%- set amount_measures = measures | selectattr('name', 'search', 'amount|revenue|cost') | list %}
        {%- set quantity_measures = measures | selectattr('name', 'search', 'quantity|units|count') | list %}
        
        {%- if amount_measures and quantity_measures %}
        case 
            when {{ quantity_measures[0].name or quantity_measures[0].column }} > 0 
            then {{ amount_measures[0].name or amount_measures[0].column }} / {{ quantity_measures[0].name or quantity_measures[0].column }}
            else 0 
        end as average_unit_value,
        {%- endif %}
        
        -- Data lineage
        '{{ "{{ this.schema }}" }}.{{ "{{ this.table }}" }}' as target_table,
        '{{ config.source_table }}' as source_table_name
        
    from data_quality_layer
),

final as (
    select
        -- Primary key
        {{ "{{ dbt_utils.generate_surrogate_key(" }}{{ business_keys | tojson }}{{ ") }}" }} as {{ table_name }}_key,
        
        -- All business data
        *,
        
        -- Performance metrics
        {%- if config.get('source_updated_column') %}
        extract(epoch from current_timestamp() - source_updated_at) as data_age_seconds,
        {%- endif %}
        
        -- Data freshness classification
        case 
            {%- if config.get('source_updated_column') %}
            when extract(epoch from current_timestamp() - source_updated_at) <= 3600 then 'FRESH'
            when extract(epoch from current_timestamp() - source_updated_at) <= 86400 then 'ACCEPTABLE'
            {%- else %}
            when extract(epoch from current_timestamp() - created_at) <= 3600 then 'FRESH'
            when extract(epoch from current_timestamp() - created_at) <= 86400 then 'ACCEPTABLE'
            {%- endif %}
            else 'STALE'
        end as data_freshness_status
        
    from business_logic_layer
    where data_quality_status = 'VALID'
      {%- if config.get('additional_filters') %}
      {%- for filter in config.additional_filters %}
      and {{ filter }}
      {%- endfor %}
      {%- endif %}
)

select * from final
'''
        
        template = Template(template_str)
        return template.render(
            table_name=table_name,
            config=config,
            generation_timestamp="{{run_started_at}}"
        )
    
    def generate_dimension_table_model(self, table_name: str, config: Dict) -> str:
        """Generate DBT model for dimension table"""
        
        if config.get('scd_type') == 'type2':
            return self.generate_scd_type2_model(table_name, config)
        else:
            return self.generate_scd_type1_model(table_name, config)
    
    def generate_scd_type1_model(self, table_name: str, config: Dict) -> str:
        """Generate SCD Type 1 dimension model"""
        
        template_str = '''
{#- SCD Type 1 Dimension Template -#}
{{ config(
    materialized='table',
    tags={{ config.tags | tojson }},
    post_hook=[
        "{{ log_dimension_stats() }}",
        "{{ update_data_lineage() }}"
    ]
) }}

{%- set natural_key = config.natural_key -%}
{%- set attributes = config.attributes -%}

-- Generated dimension table: {{ table_name }}
-- Description: {{ config.description }}
-- SCD Type: Type 1 (Overwrite)

with source_data as (
    select
        -- Natural key
        {{ natural_key }} as natural_key,
        
        -- Attributes with transformations
        {%- for attr in attributes %}
        {%- if attr.get('transformation') %}
        {{ attr.transformation }}({{ attr.column }}) as {{ attr.name }},
        {%- else %}
        {{ attr.column }}{% if attr.name and attr.name != attr.column %} as {{ attr.name }}{% endif %},
        {%- endif %}
        {%- endfor %}
        
        -- Metadata columns
        {%- if config.get('source_updated_column') %}
        {{ config.source_updated_column }} as source_updated_at,
        {%- endif %}
        current_timestamp() as created_at,
        current_timestamp() as updated_at,
        '{{ "{{ invocation_id }}" }}' as batch_id
        
    from {{ "{{ ref('" + config.source_table + "') }}" }}
    
    {%- if config.get('where_clause') %}
    where {{ config.where_clause }}
    {%- endif %}
),

enhanced_data as (
    select
        *,
        
        {%- if config.get('derived_attributes') %}
        -- Derived attributes
        {%- for derived in config.derived_attributes %}
        {{ derived.calculation }} as {{ derived.name }},
        {%- endfor %}
        {%- endif %}
        
        -- Data quality indicators
        case 
            when natural_key is not null
            {%- for attr in attributes %}
            {%- if attr.get('required', False) %}
             and {{ attr.name or attr.column }} is not null
            {%- endif %}
            {%- endfor %}
            then 'VALID'
            else 'INVALID'
        end as data_quality_status,
        
        -- Row hash for change detection
        {{ "{{ dbt_utils.generate_surrogate_key(" }}{{ [attr.name or attr.column for attr in attributes] | tojson }}{{ ") }}" }} as row_hash
        
    from source_data
),

final as (
    select
        -- Surrogate key
        {{ "{{ dbt_utils.generate_surrogate_key(['" + natural_key + "']) }}" }} as {{ table_name }}_key,
        
        -- All dimension data
        *
        
    from enhanced_data
    where data_quality_status = 'VALID'
)

select * from final
'''
        
        template = Template(template_str)
        return template.render(
            table_name=table_name,
            config=config
        )
    
    def generate_scd_type2_model(self, table_name: str, config: Dict) -> str:
        """Generate SCD Type 2 dimension model"""
        
        template_str = '''
{#- SCD Type 2 Dimension Template -#}
{{ config(
    materialized='incremental',
    incremental_strategy='merge',
    unique_key=['{{ config.natural_key }}', 'effective_from'],
    cluster_by=['{{ config.natural_key }}', 'effective_from'],
    tags={{ config.tags | tojson }},
    post_hook=[
        "{{ log_scd_stats() }}",
        "{{ validate_scd_integrity('" + table_name + "') }}"
    ]
) }}

-- Generated SCD Type 2 dimension: {{ table_name }}
-- Description: {{ config.description }}

{{ "{{ scd_type2('" + config.source_table + "', '" + config.natural_key + "', '" + config.get('updated_at_column', 'updated_at') + "') }}" }}
'''
        
        template = Template(template_str)
        return template.render(
            table_name=table_name,
            config=config
        )
    
    def generate_schema_yml(self, table_type: str, tables: Dict) -> str:
        """Generate schema.yml file for models"""
        
        schema_config = {
            'version': 2,
            'models': []
        }
        
        for table_name, config in tables.items():
            model_config = {
                'name': table_name,
                'description': config.get('description', f'Generated {table_type} table'),
                'meta': {
                    'owner': 'data_engineering',
                    'tags': config.get('tags', []),
                    'generated': True
                },
                'columns': []
            }
            
            # Add primary key column
            model_config['columns'].append({
                'name': f"{table_name}_key",
                'description': f"Surrogate key for {table_name}",
                'tests': ['unique', 'not_null'],
                'meta': {'primary_key': True}
            })
            
            if table_type == 'fact':
                self._add_fact_columns(model_config, config)
            elif table_type == 'dimension':
                self._add_dimension_columns(model_config, config)
            
            schema_config['models'].append(model_config)
        
        return yaml.dump(schema_config, default_flow_style=False, sort_keys=False)
    
    def _add_fact_columns(self, model_config: Dict, config: Dict):
        """Add fact table specific columns to schema"""
        
        # Business keys
        for key in config.get('business_keys', []):
            model_config['columns'].append({
                'name': key,
                'description': f"Business key: {key}",
                'tests': ['not_null']
            })
        
        # Dimension foreign keys
        for dim_ref in config.get('dimension_references', []):
            model_config['columns'].append({
                'name': f"{dim_ref['name']}_key",
                'description': f"Foreign key to {dim_ref['name']} dimension",
                'tests': ['not_null', 'relationships'],
                'meta': {'foreign_key': True}
            })
        
        # Measures
        for measure in config.get('measures', []):
            tests = ['not_null'] if measure.get('required', False) else []
            if measure.get('data_type') in ['decimal', 'float', 'numeric']:
                tests.append('positive_values')
            
            model_config['columns'].append({
                'name': measure.get('name', measure['column']),
                'description': measure.get('description', f"Measure: {measure['column']}"),
                'tests': tests,
                'meta': {'measure': True, 'aggregation': measure.get('aggregation', 'sum')}
            })
    
    def _add_dimension_columns(self, model_config: Dict, config: Dict):
        """Add dimension table specific columns to schema"""
        
        # Natural key
        model_config['columns'].append({
            'name': 'natural_key',
            'description': f"Natural key for {config.get('natural_key', 'dimension')}",
            'tests': ['not_null']
        })
        
        # Attributes
        for attr in config.get('attributes', []):
            model_config['columns'].append({
                'name': attr.get('name', attr['column']),
                'description': attr.get('description', f"Attribute: {attr['column']}"),
                'tests': attr.get('tests', [])
            })
        
        # SCD Type 2 specific columns
        if config.get('scd_type') == 'type2':
            scd_columns = [
                ('effective_from', 'Start date for this version of the record'),
                ('effective_to', 'End date for this version of the record'),
                ('is_current', 'Flag indicating if this is the current version'),
                ('scd_change_reason', 'Reason for the SCD change')
            ]
            
            for col_name, col_desc in scd_columns:
                model_config['columns'].append({
                    'name': col_name,
                    'description': col_desc,
                    'tests': ['not_null'] if col_name in ['effective_from', 'is_current'] else []
                })
    
    def create_directory_structure(self):
        """Create necessary directory structure"""
        directories = [
            'models/marts/facts',
            'models/marts/dimensions',
            'models/staging',
            'models/intermediate'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def generate_fact_models(self):
        """Generate all fact table models"""
        fact_dir = Path('models/marts/facts')
        
        for table_name, config in self.fact_tables.items():
            model_sql = self.generate_fact_table_model(table_name, config)
            
            with open(fact_dir / f'{table_name}.sql', 'w') as f:
                f.write(model_sql)
            
            logger.info(f"Generated fact table model: {table_name}")
    
    def generate_dimension_models(self):
        """Generate all dimension table models"""
        dim_dir = Path('models/marts/dimensions')
        
        for table_name, config in self.dimension_tables.items():
            model_sql = self.generate_dimension_table_model(table_name, config)
            
            with open(dim_dir / f'{table_name}.sql', 'w') as f:
                f.write(model_sql)
            
            logger.info(f"Generated dimension table model: {table_name}")
    
    def generate_schema_files(self):
        """Generate schema.yml files"""
        # Fact tables schema
        fact_schema = self.generate_schema_yml('fact', self.fact_tables)
        with open('models/marts/facts/schema.yml', 'w') as f:
            f.write(fact_schema)
        
        # Dimension tables schema
        dim_schema = self.generate_schema_yml('dimension', self.dimension_tables)
        with open('models/marts/dimensions/schema.yml', 'w') as f:
            f.write(dim_schema)
        
        logger.info("Generated schema.yml files")
    
    def generate_all_models(self, model_type: Optional[str] = None):
        """Generate all models or specific type"""
        self.create_directory_structure()
        
        if model_type is None or model_type == 'facts':
            self.generate_fact_models()
        
        if model_type is None or model_type == 'dimensions':
            self.generate_dimension_models()
        
        if model_type is None:
            self.generate_schema_files()
        
        total_models = len(self.fact_tables) + len(self.dimension_tables)
        logger.info(f"Model generation complete! Generated {total_models} models total.")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Generate DBT models for enterprise data warehouse')
    parser.add_argument('--type', choices=['facts', 'dimensions'], help='Generate specific model type')
    parser.add_argument('--config', default='governance/data_catalog.yml', help='Path to configuration file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        generator = DataWarehouseModelGenerator(args.config)
        generator.generate_all_models(args.type)
    except Exception as e:
        logger.error(f"Model generation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
