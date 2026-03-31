from sqlalchemy.orm import Session
from app.models import Driver

def release_driver(session: Session, driver_id: int):
    driver = session.query(Driver).filter_by(id=driver_id).first()
    if driver:
        driver.is_available = True
        session.commit()