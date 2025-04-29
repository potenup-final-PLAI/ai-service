from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from uuid import UUID
from app.ai.profile import CharacterProfile

# 기본 요청 모델 - UserIdBase
class UserIdBase(BaseModel):
    user_id: str

# 프로필 생성 관련 모델
class ProfileGenerateRequest(UserIdBase):
    pass

# 대화 기록 초기화(캐릭터 리셋) 모델
class ClearConversationRequest(UserIdBase):
    pass

# 프로필 응답 모델
class CharacterProfileResponse(BaseModel):
    user_id: str
    profile: CharacterProfile

# 프로필 대화 관련 모델
class ProfileChatRequest(UserIdBase):
    question: str

class ProfileChatResponse(BaseModel):
    user_id: str
    response: str 