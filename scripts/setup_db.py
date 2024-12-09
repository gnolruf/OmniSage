import os
import sys
import subprocess
import psycopg2
import argparse
import json
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

def run_command(command):
    """Run a shell command and return its output."""
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        print(f"Error output: {e.stderr}")
        return None

def check_postgres_installation():
    """Check if PostgreSQL is installed and running."""
    if sys.platform == 'win32':
        # Windows: Check if postgres service is running
        service_check = run_command('sc query postgresql')
        if not service_check or "RUNNING" not in service_check:
            print("PostgreSQL service is not running or not installed.")
            return False
    else:
        # Unix-like: Check using pg_isready
        if run_command('pg_isready') is None:
            print("PostgreSQL is not running or not installed.")
            return False
    return True

def update_config_file(db_config):
    """Update the database configuration file."""
    config_dir = os.path.join('configs')
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, 'database.json')
    
    try:
        with open(config_path, 'w') as f:
            json.dump(db_config, f, indent=4)
        print(f"Database configuration saved to {config_path}")
    except Exception as e:
        print(f"Error saving configuration: {e}")
        return False
    return True

def create_database(host, port, username, password, database_name):
    """Create database and necessary tables."""
    try:
        # First connect to default database to create our database
        conn = psycopg2.connect(
            dbname="postgres",
            user=username,
            password=password,
            host=host,
            port=port
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        
        with conn.cursor() as cur:
            # Check if database exists
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (database_name,))
            if not cur.fetchone():
                print(f"Creating database {database_name}...")
                cur.execute(f"CREATE DATABASE {database_name}")
            else:
                print(f"Database {database_name} already exists.")
        
        conn.close()
        
        # Connect to our database and create tables
        conn = psycopg2.connect(
            dbname=database_name,
            user=username,
            password=password,
            host=host,
            port=port
        )
        
        # Read and execute SQL file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        sql_file = os.path.join(script_dir, 'db', 'init_db.sql')
        
        with conn.cursor() as cur:
            with open(sql_file, 'r') as f:
                print("Creating tables...")
                cur.execute(f.read())
        
        conn.commit()
        print("Database setup completed successfully!")
        
    except psycopg2.Error as e:
        print(f"Database error: {e}")
        return False
    finally:
        if conn:
            conn.close()
    
    return True

def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description='Set up the OmniSage database.')
    parser.add_argument('--host', default='localhost',
                      help='PostgreSQL host (default: localhost)')
    parser.add_argument('--port', type=int, default=5432,
                      help='PostgreSQL port (default: 5432)')
    parser.add_argument('--username', required=True,
                      help='PostgreSQL username')
    parser.add_argument('--password', required=True,
                      help='PostgreSQL password')
    parser.add_argument('--database', default='llamachat',
                      help='Database name (default: llamachat)')
    
    args = parser.parse_args()
    
    print("Checking PostgreSQL installation...")
    if not check_postgres_installation():
        print("Please install and start PostgreSQL before running this script.")
        sys.exit(1)
    
    print("Setting up database...")
    if create_database(args.host, args.port, args.username, args.password, args.database):
        # Create config file
        db_config = {
            "host": args.host,
            "port": args.port,
            "database": args.database,
            "user": args.username,
            "password": args.password
        }
        
        if update_config_file(db_config):
            print("\nDatabase setup complete!")
            print("Configuration file has been updated.")
            print("You can now start using OmniSage.")
        else:
            print("\nDatabase setup complete, but configuration file creation failed.")
            print("Please check the configs directory permissions.")
    else:
        print("\nDatabase setup failed.")
        print("Please check the error messages above and try again.")

if __name__ == "__main__":
    main()