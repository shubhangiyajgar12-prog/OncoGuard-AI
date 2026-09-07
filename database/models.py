from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
)
from sqlalchemy.orm import relationship

from database.connection import Base


# ============================================================
# USER
# ============================================================

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)

    username = Column(String(100), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)

    role = Column(String(20), nullable=False, default="patient")
    active = Column(Boolean, default=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    patient_profile = relationship(
        "PatientProfile",
        foreign_keys="PatientProfile.user_id",
        back_populates="user",
        uselist=False,
    )

    assigned_patients = relationship(
        "PatientProfile",
        foreign_keys="PatientProfile.doctor_id",
        back_populates="doctor",
    )

    doctor_notes = relationship(
        "DoctorNote",
        foreign_keys="DoctorNote.doctor_id",
        back_populates="doctor",
    )

    followups = relationship(
        "FollowUp",
        foreign_keys="FollowUp.doctor_id",
        back_populates="doctor",
    )


# ============================================================
# PATIENT PROFILE
# ============================================================

class PatientProfile(Base):
    __tablename__ = "patient_profiles"

    id = Column(Integer, primary_key=True, index=True)

    user_id = Column(
        Integer,
        ForeignKey("users.id"),
        nullable=False,
        unique=True,
    )

    doctor_id = Column(
        Integer,
        ForeignKey("users.id"),
        nullable=True,
    )

    age = Column(Integer, nullable=False)
    gender = Column(Integer, nullable=False)
    bmi = Column(Float, nullable=False)

    smoking = Column(Integer, nullable=False)
    genetic_risk = Column(Integer, nullable=False)
    physical_activity = Column(Float, nullable=False)
    alcohol_intake = Column(Float, nullable=False)
    cancer_history = Column(Integer, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow)

    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    user = relationship(
        "User",
        foreign_keys=[user_id],
        back_populates="patient_profile",
    )

    doctor = relationship(
        "User",
        foreign_keys=[doctor_id],
        back_populates="assigned_patients",
    )

    assessments = relationship(
        "RiskAssessment",
        back_populates="patient",
        cascade="all, delete-orphan",
    )

    recommendations = relationship(
        "Recommendation",
        back_populates="patient",
        cascade="all, delete-orphan",
    )

    notes = relationship(
        "DoctorNote",
        back_populates="patient",
        cascade="all, delete-orphan",
    )

    followups = relationship(
        "FollowUp",
        back_populates="patient",
        cascade="all, delete-orphan",
    )

    alerts = relationship(
        "Alert",
        back_populates="patient",
        cascade="all, delete-orphan",
    )


# ============================================================
# MODEL VERSION
# ============================================================

class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, index=True)

    version = Column(
        String(100),
        unique=True,
        nullable=False,
    )

    model_path = Column(String(500), nullable=False)

    threshold = Column(Float, nullable=False)

    metrics = Column(JSON, nullable=True)

    calibration_method = Column(
        String(100),
        nullable=True,
    )

    active = Column(Boolean, default=False)

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
    )

    assessments = relationship(
        "RiskAssessment",
        back_populates="model_version",
    )


# ============================================================
# RISK ASSESSMENT
# ============================================================

class RiskAssessment(Base):
    __tablename__ = "risk_assessments"

    id = Column(Integer, primary_key=True, index=True)

    patient_id = Column(
        Integer,
        ForeignKey("patient_profiles.id"),
        nullable=False,
    )

    model_version_id = Column(
        Integer,
        ForeignKey("model_versions.id"),
        nullable=False,
    )

    timestamp = Column(
        DateTime,
        default=datetime.utcnow,
        index=True,
    )

    # IMPORTANT:
    # Existing SQLite database uses risk_probability.
    # Python code uses probability.
    probability = Column(
        "risk_probability",
        Float,
        nullable=False,
    )

    threshold = Column(
        Float,
        nullable=False,
    )

    prediction = Column(
        Integer,
        nullable=False,
    )

    risk_band = Column(
        String(100),
        nullable=False,
    )

    input_snapshot = Column(
        JSON,
        nullable=True,
    )

    patient = relationship(
        "PatientProfile",
        back_populates="assessments",
    )

    model_version = relationship(
        "ModelVersion",
        back_populates="assessments",
    )

    explanations = relationship(
        "RiskExplanation",
        back_populates="assessment",
        cascade="all, delete-orphan",
    )

    recommendations = relationship(
        "Recommendation",
        back_populates="assessment",
        cascade="all, delete-orphan",
    )


# ============================================================
# RISK EXPLANATION
# ============================================================

class RiskExplanation(Base):
    __tablename__ = "risk_explanations"

    id = Column(Integer, primary_key=True, index=True)

    assessment_id = Column(
        Integer,
        ForeignKey("risk_assessments.id"),
        nullable=False,
    )

    feature = Column(
        String(100),
        nullable=False,
    )

    feature_value = Column(
        String(255),
        nullable=True,
    )

    shap_value = Column(
        Float,
        nullable=False,
    )

    direction = Column(
        String(100),
        nullable=False,
    )

    rank = Column(
        Integer,
        nullable=True,
    )

    assessment = relationship(
        "RiskAssessment",
        back_populates="explanations",
    )


# ============================================================
# RECOMMENDATION
# ============================================================

class Recommendation(Base):
    __tablename__ = "recommendations"

    id = Column(Integer, primary_key=True, index=True)

    patient_id = Column(
        Integer,
        ForeignKey("patient_profiles.id"),
        nullable=False,
    )

    assessment_id = Column(
        Integer,
        ForeignKey("risk_assessments.id"),
        nullable=True,
    )

    feature = Column(
        String(100),
        nullable=False,
    )

    text = Column(
        Text,
        nullable=False,
    )

    reason = Column(
        Text,
        nullable=True,
    )

    status = Column(
        String(50),
        default="Not Started",
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
    )

    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    patient = relationship(
        "PatientProfile",
        back_populates="recommendations",
    )

    assessment = relationship(
        "RiskAssessment",
        back_populates="recommendations",
    )


# ============================================================
# DOCTOR NOTE
# ============================================================

class DoctorNote(Base):
    __tablename__ = "doctor_notes"

    id = Column(Integer, primary_key=True, index=True)

    patient_id = Column(
        Integer,
        ForeignKey("patient_profiles.id"),
        nullable=False,
    )

    doctor_id = Column(
        Integer,
        ForeignKey("users.id"),
        nullable=False,
    )

    note = Column(
        Text,
        nullable=False,
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
    )

    patient = relationship(
        "PatientProfile",
        back_populates="notes",
    )

    doctor = relationship(
        "User",
        foreign_keys=[doctor_id],
        back_populates="doctor_notes",
    )


# ============================================================
# FOLLOW-UP
# ============================================================

class FollowUp(Base):
    __tablename__ = "followups"

    id = Column(Integer, primary_key=True, index=True)

    patient_id = Column(
        Integer,
        ForeignKey("patient_profiles.id"),
        nullable=False,
    )

    doctor_id = Column(
        Integer,
        ForeignKey("users.id"),
        nullable=False,
    )

    followup_date = Column(
        DateTime,
        nullable=True,
    )

    status = Column(
        String(50),
        default="Pending",
    )

    notes = Column(
        Text,
        nullable=True,
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
    )

    patient = relationship(
        "PatientProfile",
        back_populates="followups",
    )

    doctor = relationship(
        "User",
        foreign_keys=[doctor_id],
        back_populates="followups",
    )


# ============================================================
# ALERT
# ============================================================

class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)

    patient_id = Column(
        Integer,
        ForeignKey("patient_profiles.id"),
        nullable=False,
    )

    alert_type = Column(
        String(100),
        nullable=False,
    )

    message = Column(
        Text,
        nullable=False,
    )

    severity = Column(
        String(50),
        default="Medium",
    )

    acknowledged = Column(
        Boolean,
        default=False,
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
    )

    patient = relationship(
        "PatientProfile",
        back_populates="alerts",
    )