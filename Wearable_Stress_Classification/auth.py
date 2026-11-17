"""
=============================================================================
PULSEGUARD AI — USER AUTHENTICATION & SECURITY MODULE
Module: auth.py
Description: Secure persistent user registration, PBKDF2 password hashing,
             email validation, and session authentication.
=============================================================================
"""

import os
import re
import sqlite3
import hashlib
import secrets
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "pulseguard_users.db")


def get_db_connection():
    """Returns a SQLite database connection with row factory."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initializes the persistent user database table if not present."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL COLLATE NOCASE,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()

    # Pre-seed a default demo user if the database is empty
    cursor.execute("SELECT COUNT(*) FROM users")
    count = cursor.fetchone()[0]
    if count == 0:
        demo_name = "Demo User"
        demo_email = "demo@pulseguard.ai"
        demo_pass = "Password123!"
        salt = secrets.token_hex(16)
        pwd_hash = _hash_pwd(demo_pass, salt)
        cursor.execute("""
            INSERT INTO users (full_name, email, password_hash, salt, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (demo_name, demo_email, pwd_hash, salt, datetime.now().isoformat()))
        conn.commit()

    conn.close()


def _hash_pwd(password: str, salt: str) -> str:
    """Derives a secure 256-bit cryptographic hash using PBKDF2-HMAC-SHA256."""
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        iterations=100_000
    ).hex()


def validate_email(email: str) -> bool:
    """Validates email format using standard regex."""
    pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    return bool(re.match(pattern, email.strip()))


def validate_password(password: str) -> tuple[bool, str]:
    """Validates that password meets minimum security standards."""
    if len(password) < 6:
        return False, "Password must be at least 6 characters long."
    return True, ""


def register_user(full_name: str, email: str, password: str) -> tuple[bool, str]:
    """Registers a new user with secure password hashing."""
    full_name = full_name.strip()
    email = email.strip().lower()

    if not full_name:
        return False, "Full name cannot be empty."

    if not validate_email(email):
        return False, "Please enter a valid email address (e.g. name@example.com)."

    is_valid_pwd, pwd_err = validate_password(password)
    if not is_valid_pwd:
        return False, pwd_err

    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT id FROM users WHERE email = ?", (email,))
    if cursor.fetchone() is not None:
        conn.close()
        return False, "An account with this email address already exists."

    salt = secrets.token_hex(16)
    pwd_hash = _hash_pwd(password, salt)

    cursor.execute("""
        INSERT INTO users (full_name, email, password_hash, salt, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (full_name, email, pwd_hash, salt, datetime.now().isoformat()))
    conn.commit()
    conn.close()

    return True, "Account registered successfully! You can now log in."


def authenticate_user(email: str, password: str) -> tuple[bool, dict | None, str]:
    """Authenticates user credentials against the persistent database."""
    email = email.strip().lower()

    if not email or not password:
        return False, None, "Please enter both email and password."

    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT id, full_name, email, password_hash, salt FROM users WHERE email = ?", (email,))
    user = cursor.fetchone()
    conn.close()

    if user is None:
        return False, None, "No account found with this email address."

    expected_hash = _hash_pwd(password, user["salt"])
    if secrets.compare_digest(expected_hash, user["password_hash"]):
        user_data = {
            "id": user["id"],
            "full_name": user["full_name"],
            "email": user["email"]
        }
        return True, user_data, "Authentication successful."
    else:
        return False, None, "Incorrect password. Please try again."


if __name__ == "__main__":
    init_db()
    print("Database initialized successfully.")
    ok, u, msg = authenticate_user("demo@pulseguard.ai", "Password123!")
    print(f"Demo auth test: {ok}, User: {u}, Msg: {msg}")
