from pydantic import BaseModel, Field
from typing import Any, Dict, Optional
import os
from typing import Union
from fastapi.responses import JSONResponse

class ResponseData(BaseModel):
    details: Any  # Accepts any data type

class CustomResponse(BaseModel):
    status: str = "success"
    message: str = ""
    data: Optional[Any] = None
    
def create_response(data: Optional[Dict] = None, message: str = "", status: str = "success") -> CustomResponse:
    response_structure = CustomResponse(
        status=status,
        message=message,
        data=data
    )
    return response_structure

# Old version of create_response
def create_response_old(data: Union[dict, None] = None, message: str = "", status: str = "success") -> JSONResponse:
    response_structure = {
        "status": status, #success | error
        "message": message, # message to display as alert if message div
        "data": data # the response of the query (review, enhance, title, proofread)
    }
    return JSONResponse(content=response_structure)