from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.db.session import get_db
from app.models.product import Product
from app.schemas.product import (
    ProductCreate,
    ProductUpdate,
    ProductResponse
)

from app.core.security import get_current_user
from app.models.user import User

router = APIRouter()

@router.post("/", response_model=ProductResponse)
async def create_product(product: ProductCreate, db: AsyncSession = Depends(get_db),current_user: User = Depends(get_current_user)):
    product_details = Product(
        name= product.name,
        description= product.description,
        price= product.price,
        stock= product.stock
    )
    db.add(product_details)
    await db.commit()
    await db.refresh(product_details)
    return product_details

@router.get("/", response_model=list[ProductResponse])
async def get_products(
    db: AsyncSession = Depends(get_db)
):

    result = await db.execute(
        select(Product)
    )

    products = result.scalars().all()

    return products

@router.get("/{product_id}", response_model=ProductResponse)
async def get_products( product_id : int, db: AsyncSession = Depends(get_db),current_user: User = Depends(get_current_user)):

    result = await db.execute(
            select(Product).where(Product.id == product_id)
        )

    product = result.scalar_one_or_none()

    if not product:

        raise HTTPException(
            status_code=404,
            detail="Product not found"
        )


    return product

@router.put("/{product_id}", response_model=ProductResponse)
async def update_product(

    product_id: int,

    product_update: ProductUpdate,

    db: AsyncSession = Depends(get_db),

    current_user: User = Depends(get_current_user)

):

    result = await db.execute(
        select(Product).where(Product.id == product_id)
    )

    product = result.scalar_one_or_none()

    if not product:

        raise HTTPException(
            status_code=404,
            detail="Product not found"
        )

    if product_update.name is not None:
        product.name = product_update.name

    if product_update.price is not None:
        product.price = product_update.price

    if product_update.stock is not None:
        product.stock = product_update.stock

    await db.commit()

    await db.refresh(product)

    return product

@router.delete("/{product_id}")
async def delete_product(

    product_id: int,

    db: AsyncSession = Depends(get_db),

    current_user: User = Depends(get_current_user)

):

    result = await db.execute(
        select(Product).where(Product.id == product_id)
    )

    product = result.scalar_one_or_none()

    if not product:

        raise HTTPException(
            status_code=404,
            detail="Product not found"
        )

    await db.delete(product)

    await db.commit()

    return {"message":"Product deleted"}