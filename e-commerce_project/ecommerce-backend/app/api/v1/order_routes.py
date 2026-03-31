from fastapi import APIRouter, Depends, HTTPException

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from sqlalchemy.orm import selectinload

from app.db.session import get_db

from app.models.order import Order
from app.models.product import Product
from app.models.user import User

from app.models.order_item import OrderItem
from app.schemas.order import (
    OrderCreate,
    OrderResponse
)
from app.core.security import get_current_user

router = APIRouter()

@router.post("/", response_model=OrderResponse)
async def create_order(

    order:OrderCreate,

    db:AsyncSession=Depends(get_db),

    current_user:User=Depends(get_current_user)

):

        db_order = Order(
            user_id=current_user.id
        )

        db.add(db_order)

        await db.flush()

        for item in order.items:

            result = await db.execute(

                select(Product)
                .where(Product.id==item.product_id)
                .with_for_update()

            )

            product = result.scalar_one_or_none()

            if not product:

                raise HTTPException(
                    status_code=404,
                    detail="Product not found"
                )

            if product.stock < item.quantity:

                raise HTTPException(
                    status_code=400,
                    detail=f"Not enough stock for product {product.id}"
                )

            product.stock -= item.quantity

            order_item = OrderItem(

                order_id=db_order.id,

                product_id=item.product_id,

                quantity=item.quantity
            )

            db.add(order_item)

            await db.commit()

            await db.refresh(db_order)
        
        result = await db.execute(
            select(Order)
            .options(selectinload(Order.items))
            .where(Order.id == db_order.id)
        )

        db_order = result.scalar_one()

        return db_order


@router.get("/my-orders", response_model=list[OrderResponse])
async def get_my_orders(

    db: AsyncSession = Depends(get_db),

    current_user: User = Depends(get_current_user)

):

    result = await db.execute(

        select(Order).where(
            Order.user_id == current_user.id
        )

    )

    orders = result.scalars().all()

    return orders




