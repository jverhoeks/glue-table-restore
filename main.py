import awswrangler as wr
import argparse
import logging
from typing import List, Optional, Dict
from urllib.parse import urlparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GlueTableCreator:
    def __init__(self, s3_base_path: str, database: str, profile: Optional[str] = None, dry_run: bool = False):
        """
        Initialize the Glue Table Creator
        
        Args:
            s3_base_path: Base S3 path containing multiple tables
            database: Target Glue database name
            profile: AWS profile name (optional)
            dry_run: Whether to run in dry-run mode
        """
        self.s3_base_path = s3_base_path.rstrip('/')
        self.database = database
        self.dry_run = dry_run
        
        # Set AWS session if profile provided
        if profile:
            logger.info(f"Using AWS profile: {profile}")
            self.session = wr.Session(profile_name=profile)
            self.s3_client = self.session.client('s3')
        else:
            import boto3
            self.session = None
            self.s3_client = boto3.client('s3')

    def _get_bucket_and_prefix(self) -> tuple:
        """Extract bucket name and prefix from S3 path"""
        parsed = urlparse(self.s3_base_path)
        return parsed.netloc, parsed.path.lstrip('/')

    def discover_tables(self) -> Dict[str, str]:
        """
        Recursively discover all tables in the S3 path
        Returns dict of table_name: s3_path
        """
        bucket, prefix = self._get_bucket_and_prefix()
        tables = {}
        
        # List all objects in the bucket with the given prefix
        paginator = self.s3_client.get_paginator('list_objects_v2')
        
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            if 'Contents' not in page:
                continue
                
            for obj in page['Contents']:
                key = obj['Key']
                if not key.endswith('.parquet'):
                    continue
                
                # Extract table path by finding partition pattern
                path_parts = key.split('/')
                table_path = []
                found_partition = False
                
                for part in path_parts:
                    if '=' in part:  # Found partition
                        found_partition = True
                        break
                    table_path.append(part)
                
                if found_partition:
                    table_name = table_path[-1]  # Last component before partition
                    table_s3_path = f"s3://{bucket}/{'/'.join(table_path)}"
                    
                    if table_name not in tables:
                        tables[table_name] = table_s3_path
        
        return tables

    def _create_single_table(self, table_name: str, s3_path: str) -> None:
        """Create a single Glue table"""
        try:
            logger.info(f"Processing table: {table_name}")
            
            # Find partition columns by analyzing the path structure
            sample_files = []
            bucket, prefix = self._get_bucket_and_prefix()
            response = self.s3_client.list_objects_v2(
                Bucket=bucket,
                Prefix=s3_path.replace(f"s3://{bucket}/", ""),
                MaxKeys=1
            )
            if 'Contents' in response:
                sample_files = [f"s3://{bucket}/{obj['Key']}" for obj in response['Contents']]

            partition_cols = []
            if sample_files:
                path_diff = sample_files[0].replace(s3_path, '').lstrip('/')
                partition_parts = [p for p in path_diff.split('/') if '=' in p]
                partition_cols = [p.split('=')[0] for p in partition_parts]
            
            # Read schema from parquet
            logger.info(f"Reading schema for {table_name}...")
            df_iter = wr.s3.read_parquet(
                path=s3_path,
                dataset=True,
                use_threads=True,
                chunked=10
            )
            # Get first chunk to read schema
            df = next(df_iter)
            
            # Map pandas dtypes to Hive/Glue compatible types
            def map_dtype(dtype):
                dtype_str = str(dtype)
                if dtype_str.startswith('int') or dtype_str == 'Int64':
                    return 'bigint'
                elif dtype_str.startswith('float'):
                    return 'double'
                elif dtype_str == 'bool':
                    return 'boolean'
                elif dtype_str.startswith('datetime'):
                    return 'timestamp'
                else:
                    return 'string'
            
            # Remove partition columns from the main schema and map types
            columns_types = {col: map_dtype(dtype) for col, dtype in df.dtypes.items() 
                           if col not in partition_cols}
            
            # Create partition types dict
            partition_types = {col: 'string' for col in partition_cols}
            
            # Create or update the table
            if self.dry_run:
                logger.info(f"[DRY RUN] Would create/update table: {table_name}")
                logger.info(f"  Database: {self.database}")
                logger.info(f"  S3 Location: {s3_path}")
                logger.info(f"  Columns: {columns_types}")
                logger.info(f"  Partitions: {partition_types}")
                logger.info(f"  Compression: snappy")
                return
                
            logger.info(f"Creating/updating table {table_name}")
            wr.catalog.create_parquet_table(
                database=self.database,
                table=table_name,
                path=s3_path,
                columns_types=columns_types,
                mode='overwrite',
                partitions_types=partition_types,
                compression='snappy'
            )
            
            # Discover partitions
            if partition_cols:
                logger.info(f"Discovering partitions for {table_name}...")
                try:
                    # Get all parquet files in the table
                    all_partitions = {}
                    paginator = self.s3_client.get_paginator('list_objects_v2')
                    for page in paginator.paginate(Bucket=bucket, Prefix=s3_path.replace(f"s3://{bucket}/", "")):
                        if 'Contents' not in page:
                            continue
                            
                        for obj in page['Contents']:
                            if not obj['Key'].endswith('.parquet'):
                                continue
                                
                            # Extract partition path and values
                            key = obj['Key']
                            path_parts = key.split('/')
                            partition_values = []
                            partition_path_parts = []
                            
                            # Find the base path and collect partition values
                            for part in path_parts:
                                if '=' in part:
                                    partition_path_parts.append(part)
                                    val = part.split('=')[1]
                                    partition_values.append(val)
                            
                            if partition_values:
                                # Get the path up to the last partition
                                full_partition_path = f"s3://{bucket}/{key.split(partition_path_parts[0])[0]}{'/'.join(partition_path_parts)}/"
                                all_partitions[full_partition_path] = partition_values
                    
                    if all_partitions:
                        logger.info(f"Adding {len(all_partitions)} partitions to {table_name}")
                        wr.catalog.add_parquet_partitions(
                            database=self.database,
                            table=table_name,
                            partitions_values=all_partitions
                        )
                    else:
                        logger.warning(f"No partitions found for {table_name}")
                        
                except Exception as e:
                    logger.error(f"Error discovering partitions: {str(e)}")
                    raise
            
            logger.info(f"Successfully processed table {table_name}")
            
        except Exception as e:
            logger.error(f"Error processing table {table_name}: {str(e)}")
            raise

    def create_all_tables(self, max_workers: int = 5) -> None:
        """Create all discovered tables"""
        tables = self.discover_tables()
        logger.info(f"Discovered {len(tables)} tables")
        
        for table_name, s3_path in tables.items():
            try:
                self._create_single_table(table_name, s3_path)
            except Exception as e:
                logger.error(f"Failed to process table {table_name}: {str(e)}")
                if not self.dry_run:
                    raise

def main():
    parser = argparse.ArgumentParser(description='Create Glue tables from S3 parquet files recursively')
    parser.add_argument('s3_path', help='Base S3 path containing tables (e.g., s3://bucket/path/)')
    parser.add_argument('database', help='Target Glue database name')
    parser.add_argument('--profile', help='AWS profile name (optional)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be created without actually creating tables')
    parser.add_argument('--max-workers', type=int, default=5, help='Maximum number of concurrent table creations')
    
    args = parser.parse_args()
    
    try:
        creator = GlueTableCreator(
            s3_base_path=args.s3_path,
            database=args.database,
            profile=args.profile,
            dry_run=args.dry_run
        )
        creator.create_all_tables(max_workers=args.max_workers)
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()