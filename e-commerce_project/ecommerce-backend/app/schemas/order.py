from pydantic import BaseModel, Field


class OrderItemCreate(BaseModel):

    product_id:int

    quantity:int = Field(gt=0)


class OrderCreate(BaseModel):

    items:list[OrderItemCreate]


class OrderItemResponse(BaseModel):

    product_id:int

    quantity:int

    class Config:
        from_attributes=True


class OrderResponse(BaseModel):

    id:int

    user_id:int

    items:list[OrderItemResponse]

    class Config:
        from_attributes=True