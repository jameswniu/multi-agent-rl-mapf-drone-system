# Central error handling for the API
# Ensures consistent JSON responses for errors

from fastapi.responses import JSONResponse

class APIError(Exception):
    def __init__(self, message: str, status_code: int = 400):
        self.message = message
        self.status_code = status_code

def error_handler(request, exc: APIError):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.message})
