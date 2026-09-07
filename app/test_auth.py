from database.connection import SessionLocal
from database.schema import create_tables

from app.auth import (
    register_user,
    authenticate_user,
    verify_password
)


def main():

    print()
    print("=" * 55)
    print("ONCOGUARD AI — AUTHENTICATION TEST")
    print("=" * 55)

    create_tables()

    db = SessionLocal()

    try:

        # -------------------------------------------------
        # Password hashing test
        # -------------------------------------------------

        password = "TestPassword123"

        hashed = (
            __import__("app.auth", fromlist=["hash_password"])
            .hash_password(password)
        )

        assert hashed != password

        assert verify_password(
            password,
            hashed
        )

        assert not verify_password(
            "WrongPassword",
            hashed
        )

        print("✓ Password hashing verified")


        # -------------------------------------------------
        # Create test user
        # -------------------------------------------------

        username = "auth_test_patient"

        existing = (
            db.query(
                __import__(
                    "database.models",
                    fromlist=["User"]
                ).User
            )
            .filter_by(
                username=username
            )
            .first()
        )

        if existing:

            user = existing

            print(
                "✓ Existing test user found"
            )

        else:

            user = register_user(
                db=db,
                username=username,
                email="auth_test@oncoguard.local",
                password=password,
                role="patient"
            )

            print(
                f"✓ Test user created: ID={user.id}"
            )


        # -------------------------------------------------
        # Correct login
        # -------------------------------------------------

        authenticated = authenticate_user(
            db=db,
            username=username,
            password=password
        )

        assert authenticated is not None

        print(
            "✓ Correct credentials accepted"
        )


        # -------------------------------------------------
        # Incorrect login
        # -------------------------------------------------

        rejected = authenticate_user(
            db=db,
            username=username,
            password="WrongPassword"
        )

        assert rejected is None

        print(
            "✓ Incorrect credentials rejected"
        )


        # -------------------------------------------------
        # Role
        # -------------------------------------------------

        assert authenticated.role == "patient"

        print(
            f"✓ Role verified: {authenticated.role}"
        )


        print()
        print("=" * 55)
        print("AUTHENTICATION TEST PASSED")
        print("=" * 55)
        print()

    finally:

        db.close()


if __name__ == "__main__":
    main()