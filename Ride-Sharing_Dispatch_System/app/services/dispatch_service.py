from sqlalchemy.orm import Session
from app.models import Ride
from utils.logger import logger


def dispatch_ride(session: Session, rider_id: int):

    # business logic here...

    logger.info(f"Dispatching ride for rider {rider_id}")

    ride = Ride(rider_id=rider_id, fare=100)
    session.add(ride)
    session.commit()

    logger.info(f"Ride created with ID {ride.id}")

    return ride
