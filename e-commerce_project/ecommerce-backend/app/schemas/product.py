from pydantic import BaseModel, EmailStr, Field, ConfigDict


class ProductCreate(BaseModel):
    name:str = Field(min_length= 2, max_length= 100)

    description:str

    price:float = Field(gt=0)

    stock:int = Field(gt=0)

class ProductUpdate(BaseModel):

    name: str | None = None

    price: float | None = Field(default=None, gt=0)

    stock: int | None = Field(default=None, ge=0)    

    description:str

class ProductResponse(BaseModel):

    id: int

    name: str

    price: float

    stock: int

    description:str

    model_config = ConfigDict(
        from_attributes=True
    )
    