# Prepares an RDBMS into parquet and yaml format as expected by Tab2Graph.

import argparse
import logging
import pathlib
import os

import dateutil
import sqlalchemy
import pandas as pd
import yaml
import json

logger = logging.getLogger(__name__)
logger.setLevel('INFO')

def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--username', type=str, required=True)
    parser.add_argument('--password', type=str, required=True)
    parser.add_argument('--host', type=str, required=True)
    parser.add_argument('--database', type=str, required=True)
    parser.add_argument('--protocol', type=str, required=False, default='mysql')
    parser.add_argument('--target-table', type=str, required=True)
    parser.add_argument('--target-column', type=str, required=True)
    parser.add_argument('--output-data-path', type=str, required=True)
    parser.add_argument('--output-config-path', type=pathlib.Path, required=True)
    parser.add_argument('--column-type-path', type=pathlib.Path, required=False, default=None)
    parser.add_argument('--drop-text', action='store_true')
    return parser

def detect_type(table_name, df, column, column_type_dict=None):
    column_name = str(column.name)
    series = df[column_name]
    column_type = column.type
    dtype = None

    if isinstance(column_type, sqlalchemy.types.NullType):
        logger.warning(f'  NullType detected for column {column_name} in table {table_name}, skipping.')
    elif isinstance(column_type, sqlalchemy.dialects.mysql.ENUM):
        logger.info(f'  {table_name}.{column_name}: {column.type} -> Categorical')
        dtype = 'Categorical'
    elif isinstance(column_type, sqlalchemy.dialects.mysql.SET):
        logger.info(f'  {table_name}.{column_name}: {column.type} -> CommaSeparatedTokenSequence')
        dtype = 'CommaSeparatedTokenSequence'
    elif isinstance(column_type, (
        sqlalchemy.dialects.mysql.BIGINT,
        sqlalchemy.dialects.mysql.MEDIUMINT,
        sqlalchemy.dialects.mysql.SMALLINT,
        sqlalchemy.dialects.mysql.TINYINT,
        sqlalchemy.dialects.mysql.INTEGER,
    )):
        # We don't have a good way to determine whether an integer column is categorical or numeric, so we
        # just choose a heuristic.
        cardinality = df[column_name].nunique()
        logger.warning(f'  Integer column detected for column {column_name} in table {table_name}.')
        if cardinality < 100:
            logger.warning(f'  Setting the column to categorical due to low cardinality ({cardinality}/{df.shape[0]})')
            dtype = 'Categorical'
        else:
            logger.warning(f'  Setting the column to numeric due to high cardinality ({cardinality}/{df.shape[0]})')
            dtype = 'Numeric'
        logger.warning('  Please remember to check dataframes.yaml to see if the decision is correct.')
    elif isinstance(column_type, (
        sqlalchemy.dialects.mysql.CHAR,
        sqlalchemy.dialects.mysql.VARCHAR,
        sqlalchemy.dialects.mysql.LONGTEXT,
        sqlalchemy.dialects.mysql.MEDIUMTEXT,
        sqlalchemy.dialects.mysql.TEXT,
    )):
        logger.warning(f'  String column detected for column {column_name} in table {table_name}.')
        try:
            logger.info(f'  Trying conversion to datetime...')
            df[column_name] = pd.to_datetime(df[column_name])
            logger.info(f'  Success.')
            dtype = 'DateTime'
        except:
            logger.info(f'  Failed.')
            if (
                column_type_dict is not None and
                column_type_dict.get(table_name, {}).get(column_name, '') in ['Text', 'Categorical', 'Numerical']
            ):
                logger.info(f'  Use GPT4 result')
                dtype = column_type_dict[table_name][column_name].replace('Numerical', 'Numeric')
            elif df[column_name].str.contains(r'\s', regex=True).any():
                logger.info(f'  Whitespace detected.  Setting the column to text.')
                dtype = 'Text'
            else:
                logger.info('  No whitespace detected.  Checking cardinality...')
                cardinality = df[column_name].nunique()
                if cardinality < 10000:
                    logger.warning(f'  Setting the column to categorical due to low cardinality ({cardinality}/{df.shape[0]})')
                    dtype = 'Categorical'
                else:
                    logger.warning(f'  Setting the column to text due to high cardinality ({cardinality}/{df.shape[0]})')
                    dtype = 'Text'
            logger.warning('  Please remember to check dataframes.yaml to see if the decision is correct.')
    elif isinstance(column_type, (
        sqlalchemy.dialects.mysql.DATETIME,
        sqlalchemy.dialects.mysql.TIMESTAMP,
        sqlalchemy.types.DATE,
    )):
        df[column_name] = pd.to_datetime(df[column_name])
        dtype = 'DateTime'
    elif isinstance(column_type, (
        sqlalchemy.dialects.mysql.TIME,     # 00:00:00-23:59:59
    )):
        df[column_name] = df[column_name].apply(lambda x: None if x is None else '1970-01-01 ' + str(x))
        df[column_name] = pd.to_datetime(df[column_name])
        dtype = 'DateTime'
    elif isinstance(column_type, (
        sqlalchemy.dialects.mysql.YEAR,         # XXX: Should I treat year as a datetime information?
    )):
        df[column_name] = df[column_name].astype('str') + '-01-01'
        df[column_name] = pd.to_datetime(df[column_name])
        dtype = 'DateTime'
    elif isinstance(column_type, (
        sqlalchemy.dialects.mysql.DECIMAL,
        sqlalchemy.dialects.mysql.DOUBLE,
        sqlalchemy.dialects.mysql.FLOAT,
    )):
        dtype = 'Numeric'
    elif isinstance(column_type, (
        sqlalchemy.dialects.mysql.LONGBLOB,
        sqlalchemy.dialects.mysql.MEDIUMBLOB,
        sqlalchemy.types.BINARY,
        sqlalchemy.types.BLOB,
        sqlalchemy.types.VARBINARY,
    )):
        logger.warning('  Binary blob column detected for column {column_name} in table {table_name}, skipping.')
        dtype = None
    else:
        raise TypeError(f'Unrecognized data type {column.type}')

    logger.info(f'  {table_name}.{column_name}: {column.type} -> {dtype}')
    return dtype


def map_col_to_ntype(col, ntype, table_name, dataframe_dict, column_to_ntype):
    logger.info(f'Mapped column {col} in {table_name} to node type {ntype}.')
    col_name = str(col.name)
    dataframe_dict[table_name]['nodes'][col_name] = ntype
    column_to_ntype[str(col)] = ntype


def is_primary_key_dtype(dtype):
    return dtype in ['Categorical', 'Numeric', 'Text']


def is_primary_key(column, dtype):
    return column.primary_key and is_primary_key_dtype(dtype)


def main():
    args = get_argparser().parse_args()
    meta, dataframes = load_dataframes(
        args.database,
        args.username,
        args.password,
        args.host,
        args.protocol
    )
    build_spec_and_dataset(
        meta,
        dataframes,
        args.database,
        args.target_table,
        args.target_column,
        args.output_data_path,
        args.output_config_path,
        args.column_type_path,
        args.drop_text
    )

def load_dataframes(
    database,
    username,
    password,
    host,
    protocol
):
    connstr = f'{protocol}://{username}:{password}@{host}/{database}?charset=utf8'
    logger.info(f'Connecting to {connstr}...')
    engine = sqlalchemy.create_engine(connstr)
    meta = sqlalchemy.MetaData()
    logger.info(f'Reading metadata of database...')
    meta.reflect(engine)
    tables = meta.tables

    dataframes = {}
    for table_name, table in tables.items():
        try:
            df = pd.read_sql_table(table_name, engine)
            dataframes[table_name] = df
        except UnicodeDecodeError:
            raise Exception('Non-UTF-8 encoding is not supported.')

    return meta, dataframes

def build_spec_and_dataset(
    meta,
    dataframes,
    database_name,
    target_table,
    target_column,
    output_data_path,
    output_config_path,
    column_type_path,
    drop_text=True
):
    dataframe_dict = {}     # to be filled
    dataframe_specs = {
        'name': database_name,
        'dataframes': dataframe_dict,
    }
    task_specs = {
        'target': {
            'dataframe': target_table,
            'column': target_column,
            'time_columns': {},
            'split': {
                'train': 0.8,
                'validation': 0.1,
                'test': 0.1,    # to be filled
            },
            'evaluation_metric': 'auroc',
        }
    }
    target_column = f'{target_table}.{target_column}'
    tables = meta.tables

    # Initialize table info and export table to parquet
    for table_name, table in tables.items():
        dataframe_dict[table_name] = {
            'source': None,
            'primary_key': None,
            'nodes': {},
            'columns': {},
        }

    column_to_ntype = {}
    column_type_dict = None
    if column_type_path:
        column_type_dict = json.load(open(column_type_path, 'r'))[database_name]
    datatypes = {
        table_name: {
            column: detect_type(table_name, dataframes[table_name], column, column_type_dict)
            for column in table.columns
        }
        for table_name, table in tables.items()
    }

    # Check if the target column is a foreign key.
    # If so, emit a warning and drop the reference table.

    # Find primary keys
    # The node type name is the same as the table name.
    logger.info('Handling primary keys...')
    for table_name, table in tables.items():
        primary_keys = [column for column in table.columns if column.primary_key]
        logger.info(f'Primary keys for {table_name} are {primary_keys}.')
        num_primary_keys = len(primary_keys)

        filtered_primary_keys = []
        for column in primary_keys:
            if len(column.foreign_keys) != 0:
                logger.warning(f'Filtering primary key {column} since it is also a foreign key.')
            elif not is_primary_key_dtype(datatypes[table_name][column]):
                logger.warning(f'Filtering primary key {column} because of its data type.')
            else:
                filtered_primary_keys.append(column)
        num_filtered_primary_keys = len(filtered_primary_keys)
        logger.info(f'Primary keys for {table_name} without foreign keys are {primary_keys}.')

        if num_filtered_primary_keys == 1 and num_primary_keys == 1:
            primary_key = filtered_primary_keys[0]
            map_col_to_ntype(primary_key, table_name, table_name, dataframe_dict, column_to_ntype)
        else:
            for pk in filtered_primary_keys:
                ntype = str(pk).replace('.', '_')
                map_col_to_ntype(pk, ntype, table_name, dataframe_dict, column_to_ntype)

        # We set the primary_key field only if there is one primary key and it is not a foreign key at the same time.
        if num_filtered_primary_keys == 1 and num_primary_keys == 1:
            dataframe_dict[table_name]['primary_key'] = str(primary_key.name)

    # Some primary keys can also be foreign keys referencing other primary keys.
    # So we need to propagate the node type assignment.
    logger.info('Handling primary keys that are also foreign keys...')
    for table_name, table in tables.items():
        for column in table.columns:
            if not is_primary_key(column, datatypes[table_name][column]):
                continue
            reference_path = [column]
            current_column = column
            while len(current_column.foreign_keys) > 0:
                assert len(current_column.foreign_keys) == 1, f"Found more than 1 references in the same column {column!s}: {current_column.foreign_keys}."
                child_column = next(iter(current_column.foreign_keys)).column
                if child_column in reference_path:
                    logger.warning(f'  Cycle detected in reference path: {reference_path}')
                    break
                reference_path.append(child_column)
                current_column = child_column
            ntype = column_to_ntype[str(reference_path[-1])]
            logger.info(f'  Setting node type of all columns on reference path {[str(c) for c in reference_path]} to {ntype}.')
            for c in reference_path:
                column_to_ntype[str(c)] = ntype

    assert target_column not in column_to_ntype, f"{target_column} mapped to node type.  Is it a primary key?"

    # Handle other columns
    logger.info('Handling other columns...')
    for table_name, table in tables.items():
        df = dataframes[table_name]
        datetime_columns = []

        for column in table.columns:
            column_name = str(column.name)
            logger.info(f'Checking column {column_name} in table {table_name}...')
            if str(column) == target_column and len(column.foreign_keys) >= 1:
                # Is the foreign key column actually a target column?  Raise an error if so.
                logger.warning('****** TARGET COLUMN IS FOREIGN KEY ******')
                logger.warning('Currently this tool does not support link prediction tasks.')
                logger.warning('Ignoring the foreign key reference of target column.')
                logger.warning('******************************************')
                foreign_keys = []
            else:
                foreign_keys = column.foreign_keys

            assert len(foreign_keys) <= 1, f"Found more than 1 references in the same column {column!s}: {foreign_keys}."
            if len(foreign_keys) == 1:
                fk = next(iter(foreign_keys))
                child_table = str(fk.column.table.name)
                child_column = str(fk.column.name)
                ntype = column_to_ntype[str(fk.column)]
                logger.info(f'  Found foreign key to {child_table}.{child_column} that maps to node type {ntype}.')

                dataframe_dict[table_name]['nodes'][column_name] = ntype
            else:
                datatype = datatypes[table_name][column]
                if is_primary_key(column, datatype):
                    logger.info(f'Skipping primary key {column!s}...')
                    continue
                if datatype is not None:
                    if datatype == 'Text' and str(column) == target_column:
                        logger.warning('  The target column is assigned with Text datatype.')
                        logger.warning('  Switching to Categorical instead since we do not support text generation yet.')
                        datatype = 'Categorical'

                    if datatype == 'Text' and drop_text:
                        logger.info(f'  --drop-text enabled, dropping text column {column!s}.')
                    else:
                        dataframe_dict[table_name]['columns'][column_name] = datatype
                        if datatype == 'DateTime':
                            datetime_columns.append(column_name)

                if str(column) == target_column:
                    if datatype == 'Numeric':
                        metric = 'mse'
                        task_specs['target']['task_type'] = 'regression'
                    else:
                        assert datatype == 'Categorical', "Tasks other than classification is not supported."
                        num_categories = df[column_name].dropna().nunique()
                        logger.info(f'  Number of categories in target column {column!s}: {num_categories}')
                        if df[column_name].dropna().nunique() == 2:
                            metric = 'auroc'
                        else:
                            metric = 'accuracy'
                    logger.info(f'  Using {metric} as metric.')
                    task_specs['target']['evaluation_metric'] = metric

        # Handle time columns
        if len(datetime_columns) >= 2:
            # NOTE: We don't know which column to choose as timestamp column if multiple datetime column exists.
            # So we HONK instead.
            logger.warning(f'****** MULTIPLE TIMESTAMP COLUMNS FOUND ******')
            logger.warning(f'Found multiple datetime columns for table {table_name}:')
            for column_name in datetime_columns:
                logger.warning(f'  {column_name}')
            logger.warning(f'By default the first datetime column is chosen as timestamp column, which might be incorrect:')
            logger.warning(f'  {datetime_columns[0]}')
            logger.warning(f'**********************************************')
        if len(datetime_columns) >= 1:
            logger.info(f'Setting column {datetime_columns[0]} as timestamp column for {table_name}.')
            task_specs['target']['time_columns'][table_name] = datetime_columns[0]

    if not os.path.exists(output_data_path):
        os.makedirs(output_data_path)
    if not os.path.exists(output_config_path):
        os.makedirs(output_config_path)

    for table_name in tables.keys():
        source = f'{output_data_path}/{table_name}.pqt'
        logger.info(f'Exporting table {table_name} to {source}...')
        dataframes[table_name].to_parquet(source)
        dataframe_dict[table_name]['source'] = source

    os.makedirs(output_config_path, exist_ok=True)
    dataframe_specs_path = f'{output_config_path}/dataframes.yaml'
    logger.info(f'Writing dataframe specs to {dataframe_specs_path}...')
    with open(dataframe_specs_path, 'w') as f:
        yaml.dump(dataframe_specs, f)
    task_specs_path = f'{output_config_path}/task.yaml'
    logger.info(f'Writing task specs to {task_specs_path}...')
    with open(task_specs_path, 'w') as f:
        yaml.dump(task_specs, f)

if __name__ == '__main__':
    main()
