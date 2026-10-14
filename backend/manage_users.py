import sys
import argparse
import os
from dotenv import load_dotenv

# Load env variables for DB_URL
load_dotenv()

from IntegratedPostGreSQL import NSEDataPipeline
from UserManager import UserManager

def main():
    parser = argparse.ArgumentParser(description="Artha-Drishti Admin CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Reset Password Command
    reset_parser = subparsers.add_parser("reset_password", help="Forcefully reset a user's password")
    reset_parser.add_argument("username", type=str, help="Username of the account")
    reset_parser.add_argument("new_password", type=str, help="New password to set")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Initialize DB connection
    DB_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/StockDB")
    try:
        pipeline = NSEDataPipeline(DB_URL)
        user_manager = UserManager(pipeline.engine)
    except Exception as e:
        print(f"Error connecting to Database: {e}")
        sys.exit(1)

    if args.command == "reset_password":
        success = user_manager.update_password(args.username, args.new_password)
        if success:
            print(f"✅ Successfully updated password for user '{args.username}'")
        else:
            print(f"❌ Failed to update password. User '{args.username}' might not exist.")

if __name__ == "__main__":
    main()
