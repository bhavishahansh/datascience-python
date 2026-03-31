from fastapi import FastAPI
from app.api.v1 import user_routes,auth_routes,product_routes,order_routes
from app.db import base

from fastapi.middleware.cors import CORSMiddleware

origins = [

"http://localhost:5173",

"http://127.0.0.1:5173"

]

app = FastAPI(title="Modular E-Commerce API")


app.add_middleware(

CORSMiddleware,

allow_origins=origins,

allow_credentials=True,

allow_methods=["*"],

allow_headers=["*"],

)

app.include_router(user_routes.router, prefix="/api/v1/users", tags=["Users"])

app.include_router(auth_routes.router, prefix= "/api/v1", tags=["Authentication"])

app.include_router(product_routes.router, prefix= "/api/v1/products", tags=["Products"])

app.include_router(order_routes.router, prefix= "/api/v1/orders", tags=["Orders"])


@app.get("/")
def home():
    return {"message": "API is working"}