import hashlib
import secrets

from sqlalchemy.orm import Session

from database.models import User
from database.crud import (
    create_user,
    get_user_by_username,
    get_user_by_email
)


# =========================================================
# PASSWORD HASHING
# =========================================================

def hash_password(password: str) -> str:
    """
    Hash a password using PBKDF2-HMAC-SHA256.

    The stored value contains:
        algorithm
        salt
        derived key
    """

    salt = secrets.token_hex(16)

    derived_key = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        100_000
    )

    return f"pbkdf2_sha256${salt}${derived_key.hex()}"


def verify_password(
    password: str,
    stored_hash: str
) -> bool:

    try:

        algorithm, salt, stored_key = (
            stored_hash.split("$")
        )

        if algorithm != "pbkdf2_sha256":
            return False

        derived_key = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("utf-8"),
            100_000
        )

        return secrets.compare_digest(
            derived_key.hex(),
            stored_key
        )

    except (ValueError, AttributeError):
        return False


# =========================================================
# USER CREATION
# =========================================================

VALID_ROLES = {
    "patient",
    "doctor",
    "admin"
}


def register_user(
    db: Session,
    username: str,
    email: str,
    password: str,
    role: str
):

    role = role.lower().strip()

    if role not in VALID_ROLES:
        raise ValueError(
            f"Invalid role: {role}"
        )

    if len(password) < 8:
        raise ValueError(
            "Password must contain at least 8 characters."
        )

    if get_user_by_username(
        db,
        username
    ) is not None:

        raise ValueError(
            "Username already exists."
        )

    if get_user_by_email(
        db,
        email
    ) is not None:

        raise ValueError(
            "Email already exists."
        )

    password_hash = hash_password(
        password
    )

    return create_user(
        db=db,
        username=username,
        email=email,
        password_hash=password_hash,
        role=role
    )


# =========================================================
# LOGIN
# =========================================================

def authenticate_user(
    db: Session,
    username: str,
    password: str
):

    user = get_user_by_username(
        db,
        username
    )

    if user is None:
        return None

    if not user.active:
        return None

    if not verify_password(
        password,
        user.password_hash
    ):
        return None

    return user


# =========================================================
# ROLE CHECK
# =========================================================

def require_role(
    user: User,
    allowed_roles
):

    if user is None:
        raise PermissionError(
            "Authentication required."
        )

    if user.role not in allowed_roles:
        raise PermissionError(
            "You do not have permission "
            "to access this resource."
        )

    return True


# =========================================================
# SPECIFIC ROLE HELPERS
# =========================================================

def is_patient(user: User) -> bool:
    return (
        user is not None
        and user.role == "patient"
    )


def is_doctor(user: User) -> bool:
    return (
        user is not None
        and user.role == "doctor"
    )


def is_admin(user: User) -> bool:
    return (
        user is not None
        and user.role == "admin"
    )