from typing import Optional

from sqlalchemy.orm import Session

from database.models import (
    User,
    PatientProfile,
    ModelVersion,
    RiskAssessment,
    RiskExplanation,
    Recommendation,
    DoctorNote,
    FollowUp,
    Alert,
)


# ============================================================
# USER
# ============================================================

def create_user(
    db: Session,
    username: str,
    email: str,
    password_hash: str,
    role: str,
):
    user = User(
        username=username,
        email=email,
        password_hash=password_hash,
        role=role,
        active=True,
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    return user


def get_user_by_username(db: Session, username: str):
    return (
        db.query(User)
        .filter(User.username == username)
        .first()
    )


def get_user_by_email(db: Session, email: str):
    return (
        db.query(User)
        .filter(User.email == email)
        .first()
    )


def get_user_by_id(db: Session, user_id: int):
    return (
        db.query(User)
        .filter(User.id == user_id)
        .first()
    )


# ============================================================
# PATIENT PROFILE
# ============================================================

def create_patient_profile(
    db: Session,
    user_id: int,
    age: int,
    gender: int,
    bmi: float,
    smoking: int,
    genetic_risk: int,
    physical_activity: float,
    alcohol_intake: float,
    cancer_history: int,
    doctor_id: Optional[int] = None,
):
    profile = PatientProfile(
        user_id=user_id,
        doctor_id=doctor_id,
        age=age,
        gender=gender,
        bmi=bmi,
        smoking=smoking,
        genetic_risk=genetic_risk,
        physical_activity=physical_activity,
        alcohol_intake=alcohol_intake,
        cancer_history=cancer_history,
    )

    db.add(profile)
    db.commit()
    db.refresh(profile)

    return profile


def get_patient_profile(db: Session, user_id: int):
    return (
        db.query(PatientProfile)
        .filter(PatientProfile.user_id == user_id)
        .first()
    )


def get_patient_profile_by_id(db: Session, profile_id: int):
    return (
        db.query(PatientProfile)
        .filter(PatientProfile.id == profile_id)
        .first()
    )


def get_patients_for_doctor(db: Session, doctor_id: int):
    return (
        db.query(PatientProfile)
        .filter(PatientProfile.doctor_id == doctor_id)
        .all()
    )


# ============================================================
# MODEL VERSION
# ============================================================

def create_model_version(
    db: Session,
    version: str,
    model_path: str,
    threshold: float,
    metrics: Optional[dict] = None,
    calibration_method: Optional[str] = None,
    active: bool = False,
):
    model = ModelVersion(
        version=version,
        model_path=model_path,
        threshold=threshold,
        metrics=metrics,
        calibration_method=calibration_method,
        active=active,
    )

    db.add(model)
    db.commit()
    db.refresh(model)

    return model


def get_model_version(db: Session, version: str):
    return (
        db.query(ModelVersion)
        .filter(ModelVersion.version == version)
        .first()
    )


def get_active_model(db: Session):
    return (
        db.query(ModelVersion)
        .filter(ModelVersion.active == True)
        .order_by(ModelVersion.created_at.desc())
        .first()
    )


# ============================================================
# RISK ASSESSMENT
# ============================================================

def create_risk_assessment(
    db: Session,
    patient_id: int,
    model_version_id: int,
    risk_probability: float,
    decision_threshold: float,
    prediction: int,
    risk_band: str,
    input_snapshot: Optional[dict] = None,
):
    assessment = RiskAssessment(
        patient_id=patient_id,
        model_version_id=model_version_id,
        probability=risk_probability,
        threshold=decision_threshold,
        prediction=prediction,
        risk_band=risk_band,
        input_snapshot=input_snapshot,
    )

    db.add(assessment)
    db.commit()
    db.refresh(assessment)

    return assessment


def get_patient_assessments(
    db: Session,
    patient_id: int,
):
    return (
        db.query(RiskAssessment)
        .filter(
            RiskAssessment.patient_id == patient_id
        )
        .order_by(
            RiskAssessment.timestamp.desc()
        )
        .all()
    )


def get_latest_assessment(
    db: Session,
    patient_id: int,
):
    return (
        db.query(RiskAssessment)
        .filter(
            RiskAssessment.patient_id == patient_id
        )
        .order_by(
            RiskAssessment.timestamp.desc()
        )
        .first()
    )


# ============================================================
# RISK EXPLANATION
# ============================================================

def create_risk_explanation(
    db: Session,
    assessment_id: int,
    feature: str,
    feature_value,
    shap_value: float,
    direction: str,
    rank: Optional[int] = None,
):
    explanation = RiskExplanation(
        assessment_id=assessment_id,
        feature=feature,
        feature_value=str(feature_value),
        shap_value=shap_value,
        direction=direction,
        rank=rank,
    )

    db.add(explanation)
    db.commit()
    db.refresh(explanation)

    return explanation


def create_explanation(
    db: Session,
    assessment_id: int,
    feature: str,
    feature_value,
    shap_value: float,
    direction: str,
    rank: Optional[int] = None,
):
    return create_risk_explanation(
        db=db,
        assessment_id=assessment_id,
        feature=feature,
        feature_value=feature_value,
        shap_value=shap_value,
        direction=direction,
        rank=rank,
    )


def get_assessment_explanations(
    db: Session,
    assessment_id: int,
):
    return (
        db.query(RiskExplanation)
        .filter(
            RiskExplanation.assessment_id == assessment_id
        )
        .order_by(
            RiskExplanation.rank.asc()
        )
        .all()
    )


# ============================================================
# RECOMMENDATION
# ============================================================

def create_recommendation(
    db: Session,
    patient_id: int,
    assessment_id: Optional[int],
    feature: str,
    text: str,
    reason: Optional[str] = None,
    status: str = "Not Started",
):
    recommendation = Recommendation(
        patient_id=patient_id,
        assessment_id=assessment_id,
        feature=feature,
        text=text,
        reason=reason,
        status=status,
    )

    db.add(recommendation)
    db.commit()
    db.refresh(recommendation)

    return recommendation


def get_patient_recommendations(
    db: Session,
    patient_id: int,
):
    return (
        db.query(Recommendation)
        .filter(
            Recommendation.patient_id == patient_id
        )
        .order_by(
            Recommendation.created_at.desc()
        )
        .all()
    )


def update_recommendation_status(
    db: Session,
    recommendation_id: int,
    status: str,
):
    recommendation = (
        db.query(Recommendation)
        .filter(
            Recommendation.id == recommendation_id
        )
        .first()
    )

    if not recommendation:
        return None

    recommendation.status = status

    db.commit()
    db.refresh(recommendation)

    return recommendation


# ============================================================
# DOCTOR NOTES
# ============================================================

def create_doctor_note(
    db: Session,
    patient_id: int,
    doctor_id: int,
    note: str,
):
    doctor_note = DoctorNote(
        patient_id=patient_id,
        doctor_id=doctor_id,
        note=note,
    )

    db.add(doctor_note)
    db.commit()
    db.refresh(doctor_note)

    return doctor_note


def get_patient_notes(
    db: Session,
    patient_id: int,
):
    return (
        db.query(DoctorNote)
        .filter(
            DoctorNote.patient_id == patient_id
        )
        .order_by(
            DoctorNote.created_at.desc()
        )
        .all()
    )


# ============================================================
# FOLLOW-UP
# ============================================================

def create_followup(
    db: Session,
    patient_id: int,
    doctor_id: int,
    followup_date=None,
    status: str = "Pending",
    notes: Optional[str] = None,
):
    followup = FollowUp(
        patient_id=patient_id,
        doctor_id=doctor_id,
        followup_date=followup_date,
        status=status,
        notes=notes,
    )

    db.add(followup)
    db.commit()
    db.refresh(followup)

    return followup


def get_patient_followups(
    db: Session,
    patient_id: int,
):
    return (
        db.query(FollowUp)
        .filter(
            FollowUp.patient_id == patient_id
        )
        .order_by(
            FollowUp.followup_date.asc()
        )
        .all()
    )


# ============================================================
# ALERT
# ============================================================

def create_alert(
    db: Session,
    patient_id: int,
    alert_type: str,
    message: str,
    severity: str = "Medium",
):
    alert = Alert(
        patient_id=patient_id,
        alert_type=alert_type,
        message=message,
        severity=severity,
        acknowledged=False,
    )

    db.add(alert)
    db.commit()
    db.refresh(alert)

    return alert


def get_patient_alerts(
    db: Session,
    patient_id: int,
):
    return (
        db.query(Alert)
        .filter(
            Alert.patient_id == patient_id
        )
        .order_by(
            Alert.created_at.desc()
        )
        .all()
    )


def acknowledge_alert(
    db: Session,
    alert_id: int,
):
    alert = (
        db.query(Alert)
        .filter(
            Alert.id == alert_id
        )
        .first()
    )

    if not alert:
        return None

    alert.acknowledged = True

    db.commit()
    db.refresh(alert)

    return alert