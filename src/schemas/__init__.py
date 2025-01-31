from pydantic import BaseModel


class SystemRBAC(BaseModel):
  user_id: str
  default_role_id: str
