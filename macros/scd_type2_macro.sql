{% macro scd_type2(source_table, unique_key, updated_at_column='updated_at', effective_from='effective_from', effective_to='effective_to') %}

-- Slowly Changing Dimension Type 2 implementation with comprehensive change tracking
-- Supports both full refresh and incremental processing

{{ config(
    materialized='incremental',
    incremental_strategy='merge',
    unique_key=[unique_key, effective_from],
    cluster_by=[unique_key, effective_from]
) }}

{%- set cols = adapter.get_columns_in_relation(ref(source_table)) -%}
{%- set col_names = cols | map(attribute='name') | reject('equalto', updated_at_column) | list -%}
{%- set change_hash_cols = col_names | reject('equalto', unique_key) | list -%}

with source_data as (
    select
        {%- for col in col_names %}
        {{ col }},
        {%- endfor %}
        {{ updated_at_column }} as source_updated_at,
        
        -- Generate change detection hash
        {{ dbt_utils.generate_surrogate_key(change_hash_cols) }} as change_hash,
        
        -- Generate unique identifier
        {{ dbt_utils.generate_surrogate_key([unique_key]) }} as scd_id,
        
        -- Rank records by update time for each unique key
        row_number() over (
            partition by {{ unique_key }} 
            order by {{ updated_at_column }} desc
        ) as rn
        
    from {{ ref(source_table) }}
    
    {% if is_incremental() %}
        -- Only process records that have been updated since last run
        where {{ updated_at_column }} > (
            select coalesce(max(source_updated_at), '1900-01-01'::timestamp) 
            from {{ this }}
        )
    {% endif %}
),

latest_source_data as (
    select * from source_data where rn = 1
),

{% if is_incremental() %}
existing_data as (
    select 
        *,
        {{ dbt_utils.generate_surrogate_key([unique_key]) }} as scd_id
    from {{ this }}
    where {{ effective_to }} is null  -- Only current records
),

changed_records as (
    select 
        s.scd_id,
        current_timestamp() as change_detected_at
    from latest_source_data s
    inner join existing_data e on s.scd_id = e.scd_id
    where s.change_hash != e.change_hash
),

-- Expire changed records
expired_records as (
    select 
        e.*,
        current_timestamp() as {{ effective_to }},
        false as is_current,
        'CHANGED' as scd_change_reason
    from existing_data e
    inner join changed_records c on e.scd_id = c.scd_id
),

-- Keep unchanged records
unchanged_records as (
    select 
        e.*
    from existing_data e
    left join changed_records c on e.scd_id = c.scd_id
    where c.scd_id is null
),

-- New records (including new versions of changed records)
new_and_changed_records as (
    select 
        {%- for col in col_names %}
        s.{{ col }},
        {%- endfor %}
        s.source_updated_at,
        s.change_hash,
        current_timestamp() as {{ effective_from }},
        null::timestamp as {{ effective_to }},
        true as is_current,
        case 
            when c.scd_id is not null then 'UPDATED'
            else 'NEW'
        end as scd_change_reason,
        current_timestamp() as created_at,
        current_timestamp() as updated_at,
        '{{ invocation_id }}' as batch_id
    from latest_source_data s
    left join existing_data e on s.scd_id = e.scd_id
    left join changed_records c on s.scd_id = c.scd_id
    where e.scd_id is null  -- New records
       or c.scd_id is not null  -- Changed records
),

{% else %}
-- Full refresh: treat all records as new
new_and_changed_records as (
    select 
        {%- for col in col_names %}
        {{ col }},
        {%- endfor %}
        source_updated_at,
        change_hash,
        current_timestamp() as {{ effective_from }},
        null::timestamp as {{ effective_to }},
        true as is_current,
        'INITIAL_LOAD' as scd_change_reason,
        current_timestamp() as created_at,
        current_timestamp() as updated_at,
        '{{ invocation_id }}' as batch_id
    from latest_source_data
),
{% endif %}

final as (
    {% if is_incremental() %}
    -- Combine expired, unchanged, and new/changed records
    select * from expired_records
    union all
    select * from unchanged_records  
    union all
    {% endif %}
    select * from new_and_changed_records
)

select 
    -- Generate surrogate key for SCD record
    {{ dbt_utils.generate_surrogate_key([unique_key, effective_from]) }} as {{ this.table }}_key,
    *
from final

{% endmacro %}


{% macro log_scd_stats() %}
    {% if execute %}
        {% set scd_stats_query %}
            insert into {{ target.schema }}_audit.scd_change_log (
                table_name,
                total_records,
                current_records,
                historical_records,
                new_records,
                updated_records,
                run_timestamp,
                batch_id
            )
            select 
                '{{ this.table }}' as table_name,
                count(*) as total_records,
                sum(case when is_current then 1 else 0 end) as current_records,
                sum(case when not is_current then 1 else 0 end) as historical_records,
                sum(case when scd_change_reason = 'NEW' then 1 else 0 end) as new_records,
                sum(case when scd_change_reason = 'UPDATED' then 1 else 0 end) as updated_records,
                current_timestamp() as run_timestamp,
                '{{ invocation_id }}' as batch_id
            from {{ this }}
            where batch_id = '{{ invocation_id }}'
        {% endset %}
        
        {% do run_query(scd_stats_query) %}
    {% endif %}
{% endmacro %}


{% macro get_scd_changes(table_name, days_back=7) %}
    {% set changes_query %}
        select 
            {{ unique_key }},
            scd_change_reason,
            effective_from,
            effective_to,
            batch_id
        from {{ ref(table_name) }}
        where effective_from >= current_date - {{ days_back }}
        order by effective_from desc
    {% endset %}
    
    {{ return(changes_query) }}
{% endmacro %}


{% macro validate_scd_integrity(table_name) %}
    {% set validation_query %}
        -- Check for overlapping effective dates
        with overlaps as (
            select 
                {{ unique_key }},
                count(*) as overlap_count
            from {{ ref(table_name) }}
            where is_current = true
            group by {{ unique_key }}
            having count(*) > 1
        ),
        
        -- Check for gaps in effective dates
        gaps as (
            select 
                {{ unique_key }},
                effective_from,
                lag(effective_to) over (
                    partition by {{ unique_key }} 
                    order by effective_from
                ) as prev_effective_to
            from {{ ref(table_name) }}
            where effective_to is not null
        )
        
        select 
            'OVERLAPPING_CURRENT' as issue_type,
            count(*) as issue_count
        from overlaps
        
        union all
        
        select 
            'DATE_GAPS' as issue_type,
            count(*) as issue_count
        from gaps
        where prev_effective_to is not null 
          and effective_from != prev_effective_to
    {% endset %}
    
    {% if execute %}
        {% set results = run_query(validation_query) %}
        {% for row in results %}
            {% if row[1] > 0 %}
                {{ log("SCD Integrity Issue in " ~ table_name ~ ": " ~ row[1] ~ " " ~ row[0] ~ " found", info=true) }}
            {% endif %}
        {% endfor %}
    {% endif %}
{% endmacro %}
