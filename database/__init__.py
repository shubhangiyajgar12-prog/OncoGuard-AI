from datetime import datetime

from sqlalchemy.orm import Session

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


# =========================================================
# USER CRUD
# =========================================================

def create_user(
    db: Session,
    username: str,
    email: str,
    password_hash: str,
    role: str
):
    user = User(
        username=username,
        email=email,
        password_hash=password_hash,
        role=role,
        active=True
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    return user


def get_user_by_id(db: Session, user_id: int):
    return db.query(User).filter(
        User.id == user_id
    ).first()


def get_user_by_username(db: Session, username: str):
    return db.query(User).filter(
        User.username == username
    ).first()


def get_user_by_email(db: Session, email: str):
    return db.query(User).filter(
        User.email == email
    ).first()


# =========================================================
# PATIENT CRUD
# =========================================================

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
    doctor_id: int = None
):
    patient = PatientProfile(
        user_id=user_id,
        doctor_id=doctor_id,
        age=age,
        gender=gender,
        bmi=bmi,
        smoking=smoking,
        genetic_risk=genetic_risk,
        physical_activity=physical_activity,
        alcohol_intake=alcohol_intake,
        cancer_history=cancer_history
    )

    db.add(patient)
    db.commit()
    db.refresh(patient)

    return patient


def get_patient_by_id(db: Session, patient_id: int):
    return db.query(PatientProfile).filter(
        PatientProfile.id == patient_id
    ).first()


def get_patient_by_user_id(db: Session, user_id: int):
    return db.query(PatientProfile).filter(
        PatientProfile.user_id == user_id
    ).first()


def get_patients_by_doctor(db: Session, doctor_id: int):
    return db.query(PatientProfile).filter(
        PatientProfile.doctor_id == doctor_id
    ).all()


# =========================================================
# MODEL VERSION CRUD
# =========================================================

def create_model_version(
    db: Session,
    version: str,
    model_path: str,
    threshold: float,
    metrics: dict,
    calibration_method: str = None,
    active: bool = False
):
    model = ModelVersion(
        version=version,
        model_path=model_path,
        threshold=threshold,
        metrics=metrics,
        calibration_method=calibration_method,
        active=active
    )

    db.add(model)
    db.commit()
    db.refresh(model)

    return model


def get_active_model(db: Session):
    return db.query(ModelVersion).filter(
        ModelVersion.active == True
    ).first()


# =========================================================
# RISK ASSESSMENT CRUD
# =========================================================

def create_risk_assessment(
    db: Session,
    patient_id: int,
    risk_probability: float,
    decision_threshold: float,
    prediction: int,
    risk_band: str,
    input_snapshot: dict,
    model_version_id: int = None
):
    assessment = RiskAssessment(
        patient_id=patient_id,
        model_version_id=model_version_id,
        risk_probability=risk_probability,
        decision_threshold=decision_threshold,
        prediction=prediction,
        risk_band=risk_band,
        input_snapshot=input_snapshot
    )

    db.add(assessment)
    db.commit()
    db.refresh(assessment)

    return assessment


def get_assessment_by_id(
    db: Session,
    assessment_id: int
):
    return db.query(RiskAssessment).filter(
        RiskAssessment.id == assessment_id
    ).first()


def get_patient_assessments(
    db: Session,
    patient_id: int
):
    return (
        db.query(RiskAssessment)
        .filter(RiskAssessment.patient_id == patient_id)
        .order_by(RiskAssessment.assessment_date.asc())
        .all()
    )


def get_latest_assessment(
    db: Session,
    patient_id: int
):
    return (
        db.query(RiskAssessment)
        .filter(RiskAssessment.patient_id == patient_id)
        .order_by(RiskAssessment.assessment_date.desc())
        .first()
    )


# =========================================================
# SHAP EXPLANATION CRUD
# =========================================================

def create_explanation(
    db: Session,
    assessment_id: int,
    feature: str,
    feature_value: str,
    shap_value: float,
    direction: str,
    rank: int = None
):
    explanation = RiskExplanation(
        assessment_id=assessment_id,
        feature=feature,
        feature_value=feature_value,
        shap_value=shap_value,
        direction=direction,
        rank=rank
    )

    db.add(explanation)
    db.commit()
    db.refresh(explanation)

    return explanation


def get_assessment_explanations(
    db: Session,
    assessment_id: int
):
    return (
        db.query(RiskExplanation)
        .filter(
            RiskExplanation.assessment_id == assessment_id
        )
        .order_by(RiskExplanation.rank.asc())
        .all()
    )


# =========================================================
# RECOMMENDATION CRUD
# =========================================================

def create_recommendation(
    db: Session,
    patient_id: int,
    assessment_id: int,
    feature: str,
    recommendation: str,
    reason: str,
    status: str = "Not Started"
):
    item = Recommendation(
        patient_id=patient_id,
        assessment_id=assessment_id,
        feature=feature,
        recommendation=recommendation,
        reason=reason,
        status=status
    )

    db.add(item)
    db.commit()
    db.refresh(item)

    return item


def get_patient_recommendations(
    db: Session,
    patient_id: int
):
    return (
        db.query(Recommendation)
        .filter(Recommendation.patient_id == patient_id)
        .order_by(Recommendation.created_at.desc())
        .all()
    )


def update_recommendation_status(
    db: Session,
    recommendation_id: int,
    status: str
):
    item = db.query(Recommendation).filter(
        Recommendation.id == recommendation_id
    ).first()

    if item is None:
        return None

    item.status = status
    item.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(item)

    return item


# =========================================================
# DOCTOR NOTES
# =========================================================

def create_doctor_note(
    db: Session,
    patient_id: int,
    doctor_id: int,
    note: str
):
    item = DoctorNote(
        patient_id=patient_id,
        doctor_id=doctor_id,
        note=note
    )

    db.add(item)
    db.commit()
    db.refresh(item)

    return item


def get_patient_notes(
    db: Session,
    patient_id: int
):
    return (
        db.query(DoctorNote)
        .filter(DoctorNote.patient_id == patient_id)
        .order_by(DoctorNote.created_at.desc())
        .all()
    )


# =========================================================
# FOLLOW-UP
# =========================================================

def create_follow_up(
    db: Session,
    patient_id: int,
    doctor_id: int,
    scheduled_at,
    notes: str = None
):
    item = FollowUp(
        patient_id=patient_id,
        doctor_id=doctor_id,
        scheduled_at=scheduled_at,
        status="Scheduled",
        notes=notes
    )

    db.add(item)
    db.commit()
    db.refresh(item)

    return item


# =========================================================
# ALERTS
# =========================================================

def create_alert(
    db: Session,
    patient_id: int,
    alert_type: str,
    severity: str,
    message: str,
    assessment_id: int = None
):
    alert = Alert(
        patient_id=patient_id,
        assessment_id=assessment_id,
        alert_type=alert_type,
        severity=severity,
        message=message,
        acknowledged=False
    )

    db.add(alert)
    db.commit()
    db.refresh(alert)

    return alert


def get_patient_alerts(
    db: Session,
    patient_id: int
):
    return (
        db.query(Alert)
        .filter(Alert.patient_id == patient_id)
        .order_by(Alert.created_at.desc())
        .all()
    )


def acknowledge_alert(
    db: Session,
    alert_id: int
):
    alert = db.query(Alert).filter(
        Alert.id == alert_id
    ).first()

    if alert is None:
        return None

    alert.acknowledged = True

    db.commit()
    db.refresh(alert)

    return alert