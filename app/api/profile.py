from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.models.profile import (
    UserIdBase,
    ProfileGenerateRequest,
    ClearConversationRequest,
    CharacterProfileResponse,
    ProfileChatRequest,
    ProfileChatResponse
)
from app.services.profile import ProfileService

router = APIRouter(prefix="/profile", tags=["profile"])
profile_service = ProfileService()

# 프로필 생성 엔드포인트
@router.post("/generate", response_model=CharacterProfileResponse)
async def generate_character_profile(request: ProfileGenerateRequest):
    """대화 기록을 기반으로 캐릭터 프로필을 생성합니다"""
    try:
        profile = await profile_service.generate_character_profile(
            user_id=request.user_id,
        )
        return CharacterProfileResponse(
            user_id=request.user_id,
            profile=profile
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"프로필 생성 실패: {str(e)}")

# NPC 대화 엔드포인트
@router.post("/chat", response_model=ProfileChatResponse)
async def chat_with_npc(request: ProfileChatRequest):
    """NPC와 대화하여 캐릭터의 특성을 드러내는 대화를 생성합니다"""
    try:
        response = profile_service.chat(
            user_id=request.user_id,
            question=request.question
        )
        return ProfileChatResponse(
            user_id=request.user_id,
            response=response
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"대화 실패: {str(e)}")

# NPC 대화 스트리밍 엔드포인트
@router.post("/chat/stream")
async def chat_with_npc_stream(request: ProfileChatRequest):
    """NPC와 대화하여 캐릭터의 특성을 드러내는 대화를 스트리밍 방식으로 생성합니다"""
    try:
        return StreamingResponse(
            profile_service.chat_stream(
                user_id=request.user_id,
                question=request.question
            ),
            media_type="text/event-stream"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"대화 스트리밍 실패: {str(e)}")

# 대화 초기화 엔드포인트
@router.post("/conversation/clear")
async def clear_conversation(request: ClearConversationRequest):
    """사용자의 대화 기록을 초기화합니다"""
    try:
        profile_service.clear_conversation(user_id=request.user_id)
        return {"message": "대화 기록이 초기화되었습니다."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"대화 기록 초기화 실패: {str(e)}") 