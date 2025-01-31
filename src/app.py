"""
This module contains the main application setup.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from services.__base.manager import Manager

app = FastAPI()
app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],  # Allow specific origins
  allow_credentials=True,
  allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
  allow_headers=["*"],  # Allow all headers
)


manager = Manager(app)
manager.register_middlewares()
manager.register_services()
