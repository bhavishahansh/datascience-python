from sqlalchemy import Column, Integer, String, Float
from app.models.base import Base, TimestampMixin
from sqlalchemy.orm import relationship

class Product(Base, TimestampMixin):
    __tablename__ = "products"

    id = Column(Integer, primary_key=True)
    name = Column(String, index=True)
    price = Column(Float)
    stock = Column(Integer)
    description = Column(String)
    order_items = relationship("OrderItem")
        