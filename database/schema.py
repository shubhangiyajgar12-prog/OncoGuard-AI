from .connection import Base, engine

# Import models so SQLAlchemy knows about all tables.
from .models import (
    User,
    PatientProfile,
    ModelVersion,
    RiskAssessment,
    RiskExplanation,
    Recommendation,
    DoctorNote,
    FollowUp,
    Alert
)


def create_tables():
    """
    Create all OncoGuard AI database tables.
    """

    Base.metadata.create_all(bind=engine)

    print("==============================================")
    print("ONCOGUARD AI DATABASE")
    print("==============================================")
    print("All database tables created successfully.")
    print("==============================================")


if __name__ == "__main__":
    create_tables()