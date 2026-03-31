from sqlalchemy import (
    Column, Integer, String, Numeric,
    ForeignKey, DateTime, Boolean
)
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime

Base = declarative_base()


class Driver(Base):
    __tablename__ = "drivers"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    is_available = Column(Boolean, default=True)

    # One driver → Many rides
    rides = relationship("Ride", back_populates="driver")


class Rider(Base):
    __tablename__ = "riders"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)

    # One rider → Many rides
    rides = relationship("Ride", back_populates="rider")


class Ride(Base):
    __tablename__ = "rides"

    id = Column(Integer, primary_key=True)

    rider_id = Column(Integer, ForeignKey("riders.id"))
    driver_id = Column(Integer, ForeignKey("drivers.id"))

    fare = Column(Numeric)
    created_at = Column(DateTime, default=datetime.utcnow)

    # ORM relationships
    rider = relationship("Rider", back_populates="rides")
    driver = relationship("Driver", back_populates="rides")

    # One ride → One payment
    payment = relationship("Payment", back_populates="ride", uselist=False)


class Payment(Base):
    __tablename__ = "payments"

    id = Column(Integer, primary_key=True)
    ride_id = Column(Integer, ForeignKey("rides.id"))

    amount = Column(Numeric)
    status = Column(String)  # SUCCESS / FAILED
    created_at = Column(DateTime, default=datetime.utcnow)

    # ORM relationship
    ride = relationship("Ride", back_populates="payment")
