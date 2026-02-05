{% macro generate_fact_table(table_name, business_keys, measures, dimensions, grain='daily', source_table=none) %}

{%- set source_table = source_table or 'staging_' + table_name.replace('fact_', '') -%}

-- Macro to generate standardized fact tables with comprehensive auditing and data quality
-- Usage: {{ generate_fact_table('fact_sales', ['order_id', 'line_item_id'], ['quantity', 'amount'], ['customer', 'product', 'date']) }}

{{ config(
    materialized='incremental',
    incremental_strategy='merge',
    unique_key=business_keys + ['effective_date'],
    cluster_by=(['effective_date'] + business_keys),
    tags=['fact', grain, 'generated']
) }}

with source_data as (
    select
        -- Business keys
        {%- for key in business_keys %}
        {{ key }},
        {%- endfor %}
        
        -- Dimension foreign keys
        {%- for dim in dimensions %}
        {%- if dim == 'date' %}
        date({{ business_keys[0] if 'date' in business_keys[0] else 'transaction_date' }}) as date_key,
        {%- else %}
        {{ dim }}_id as {{ dim }}_key,
        {%- endif %}
        {%- endfor %}
        
        -- Measures
        {%- for measure in measures %}
        {%- if measure is mapping %}
        {{ measure.column }} as {{ measure.name }},
        {%- else %}
        {{ measure }},
        {%- endif %}
        {%- endfor %}
        
        -- Standard audit columns
        coalesce(updated_at, created_at, current_timestamp()) as source_updated_at,
        current_timestamp() as dbt_created_at,
        current_timestamp() as dbt_updated_at,
        '{{ invocation_id }}' as batch_id,
        '{{ grain }}' as grain_level
        
    from {{ ref(source_table) }}
    
    {% if is_incremental() %}
        where coalesce(updated_at, created_at, current_timestamp()) > 
              (select coalesce(max(source_updated_at), '1900-01-01'::timestamp) from {{ this }})
    {% endif %}
),

data_quality_checks as (
    select 
        *,
        -- Data quality indicators
        case 
            when {%- for key in business_keys %}{{ key }} is not null{% if not loop.last %} and {% endif %}{% endfor %}
            then 'VALID'
            else 'INVALID'
        end as data_quality_status,
        
        -- Completeness checks
        {%- for measure in measures %}
        {%- set measure_name = measure.name if measure is mapping else measure %}
        case when {{ measure_name }} is not null then 1 else 0 end as {{ measure_name }}_completeness_flag,
        {%- endfor %}
        
        -- Row hash for change detection
        {{ dbt_utils.generate_surrogate_key(business_keys + measures) }} as row_hash
        
    from source_data
),

enriched_data as (
    select 
        *,
        -- Derived measures based on common patterns
        {%- if 'amount' in measures|map('string')|list and 'quantity' in measures|map('string')|list %}
        case 
            when quantity > 0 then amount / quantity 
            else 0 
        end as unit_price,
        {%- endif %}
        
        {%- if 'gross_amount' in measures|map('string')|list and 'discount_amount' in measures|map('string')|list %}
        gross_amount - coalesce(discount_amount, 0) as net_amount,
        {%- endif %}
        
        -- Effective date for SCD
        date(source_updated_at) as effective_date,
        
        -- Data lineage information
        '{{ this.schema }}.{{ this.table }}' as target_table,
        '{{ source_table }}' as source_table_name,
        
        -- Performance metrics
        extract(epoch from current_timestamp() - source_updated_at) as data_age_seconds
        
    from data_quality_checks
),

final as (
    select
        -- Generate surrogate key
        {{ dbt_utils.generate_surrogate_key(business_keys + ['effective_date']) }} as {{ table_name }}_key,
        
        -- All columns from enriched data
        *,
        
        -- Data freshness indicators
        case 
            when data_age_seconds <= 3600 then 'FRESH'  -- within 1 hour
            when data_age_seconds <= 86400 then 'ACCEPTABLE'  -- within 1 day
            else 'STALE'
        end as data_freshness_status
        
    from enriched_data
    where data_quality_status = 'VALID'
)

select * from final

{% endmacro %}


{% macro log_fact_stats() %}
    {% if execute %}
        {% set stats_query %}
            insert into {{ target.schema }}_audit.fact_table_stats (
                table_name,
                record_count,
                valid_records,
                invalid_records,
                freshness_status,
                run_timestamp,
                batch_id
            )
            select 
                '{{ this.table }}' as table_name,
                count(*) as record_count,
                sum(case when data_quality_status = 'VALID' then 1 else 0 end) as valid_records,
                sum(case when data_quality_status = 'INVALID' then 1 else 0 end) as invalid_records,
                mode(data_freshness_status) as freshness_status,
                current_timestamp() as run_timestamp,
                '{{ invocation_id }}' as batch_id
            from {{ this }}
        {% endset %}
        
        {% do run_query(stats_query) %}
    {% endif %}
{% endmacro %}


{% macro validate_fact_table() %}
    {% if execute %}
        {% set validation_query %}
            -- Check for duplicate keys
            select 
                '{{ this.table }}' as table_name,
                'DUPLICATE_KEYS' as test_type,
                count(*) as issue_count
            from (
                select {{ this.table.replace('fact_', '') }}_key, count(*)
                from {{ this }}
                group by {{ this.table.replace('fact_', '') }}_key
                having count(*) > 1
            ) duplicates
            
            union all
            
            -- Check for null foreign keys
            select 
                '{{ this.table }}' as table_name,
                'NULL_FOREIGN_KEYS' as test_type,
                count(*) as issue_count
            from {{ this }}
            where date_key is null
        {% endset %}
        
        {% set results = run_query(validation_query) %}
        {% if results %}
            {% for row in results %}
                {% if row[2] > 0 %}
                    {{ log("WARNING: Found " ~ row[2] ~ " issues of type " ~ row[1] ~ " in " ~ row[0], info=true) }}
                {% endif %}
            {% endfor %}
        {% endif %}
    {% endif %}
{% endmacro %}
