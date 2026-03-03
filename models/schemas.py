from pydantic import BaseModel, Field

class FraudResponse(BaseModel):
  status: str = Field(description="FRAUD, NORMAL, 또는 SUSPICIOUS")
  fraudScore: float = Field(description="0.0에서 100.0 사이의 사기 확률 ")
  description: str = Field(description="판단 근거 및 상세 리포트")