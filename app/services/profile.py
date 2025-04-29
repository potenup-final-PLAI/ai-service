from typing import List, Dict, Any, Optional, Generator
from uuid import UUID
import openai

from app.ai.profile import ProfileAI, CharacterProfile
from app.ai.npc_chat import NPCChatAI
from app.models.characters import CharacterStatsUpdateRequest
from app.utils.loader import prompt_profile_context

class ProfileService:
    def __init__(self):
        self.profile_ai = ProfileAI()
        self.npc_chat_ai = NPCChatAI()
        self.conversation_history = {}  # 사용자 ID별 대화 기록 저장
        
    def add_conversation(self, user_id: str, role: str, content: str):
        """사용자의 대화 기록에 메시지를 추가합니다"""
        if user_id not in self.conversation_history:
            self.conversation_history[user_id] = []
            
        self.conversation_history[user_id].append({
            "role": role,
            "content": content
        })
        
    def get_conversation_history(self, user_id: str) -> List[Dict[str, str]]:
        """사용자의 대화 기록을 반환합니다"""
        return self.conversation_history.get(user_id, [])
        
    def clear_conversation(self, user_id: str):
        """사용자의 대화 기록을 초기화합니다"""
        if user_id in self.conversation_history:
            self.conversation_history[user_id] = []
    
    def has_conversation(self, user_id: str) -> bool:
        """대화 기록이 있는지 확인합니다"""
        return user_id in self.conversation_history and len(self.conversation_history[user_id]) > 0
            
    async def generate_character_profile(self, user_id: str) -> CharacterProfile:
        """사용자의 대화 기록을 기반으로 캐릭터 프로필을 생성합니다"""
        conversation_history = self.get_conversation_history(user_id)
        if not conversation_history:
            raise ValueError("대화 기록이 없습니다. 먼저 NPC와 대화를 시작하세요.")
            
        profile = await self.profile_ai.generate_character_profile(conversation_history)
        return profile
    
    def get_character_stats(self, profile: CharacterProfile) -> CharacterStatsUpdateRequest:
        """AI가 생성한 프로필에서 캐릭터 스탯을 추출합니다"""
        return self.profile_ai.convert_to_character_stats(profile)
    
    def _get_context(self, user_input: str):
        """컨텍스트를 가져옵니다"""
        class LorelessStoryRetriever:
            def invoke(self, query):
                return [type('obj', (object,), {'page_content': prompt_profile_context})]
        
        retriever = LorelessStoryRetriever()
        docs = retriever.invoke(user_input)
        return "\n\n".join(d.page_content for d in docs)
    
    def _format_chat_history(self, history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """대화 기록을 LLM 메시지 형식으로 변환합니다"""
        messages = []
        for msg in history:
            role = "assistant" if msg["role"] == "npc" else "user"
            messages.append({"role": role, "content": msg["content"]})
        return messages
    
    def chat(self, user_id: str, question: str) -> str:
        """프로필 생성을 위한 NPC 대화 - 대화 내용을 저장하고 응답을 반환합니다"""
        # 컨텍스트 가져오기
        context = self._get_context(question)
        
        # 커스텀 시스템 프롬프트
        system_prompt = """
        당신은 판타지 RPG 게임 '로어리스(Loreless)'의 NPC입니다.
        당신의 역할은 플레이어(주인공)와 대화하면서 그들의 성격, 가치관, 선호하는 행동 방식 등을 알아내는 것입니다.
        플레이어의 응답을 분석하여 그들의 특성을 파악하고, 다음 질문으로 이어가세요.
        항상 질문 형태로 대화를 이어가서 플레이어가 자신의 특성을 드러낼 수 있도록 유도하세요.
        이전 대화 내용을 참고하여 연속성 있는 대화를 만들어주세요.
        질문은 항상 성격, 가치관, 선호하는 행동 방식 등을 알아내기 위한 답변하기 쉬운 하나의 질문만 제시하세요.
        """
        
        # 기본 메시지 구성 (시스템 프롬프트)
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # 사용자의 이전 대화 기록 가져오기
        history = self.get_conversation_history(user_id)
        
        # 이전 대화 기록이 있으면 추가
        if history:
            # 첫 메시지로 컨텍스트 정보 제공
            messages.append({"role": "system", "content": f"대화 컨텍스트: {context}"})
            
            # 이전 대화 기록 추가 (최대 10개의 메시지로 제한)
            formatted_history = self._format_chat_history(history[-10:])
            messages.extend(formatted_history)
        
        # 현재 사용자 질문 추가
        messages.append({"role": "user", "content": question})
        
        # 응답 생성
        resp = openai.chat.completions.create(
            model="gpt-4.1-nano",
            messages=messages,
            temperature=0.7
        )
        response = resp.choices[0].message.content
        
        # 사용자 질문 저장
        self.add_conversation(
            user_id=user_id,
            role="user",
            content=question
        )
        
        # NPC 응답 저장
        self.add_conversation(
            user_id=user_id,
            role="npc",
            content=response
        )
        
        return response
    
    def chat_stream(self, user_id: str, question: str) -> Generator[str, None, None]:
        """프로필 생성을 위한 NPC 대화 - 스트리밍 응답"""
        # 컨텍스트 가져오기
        context = self._get_context(question)
        
        # 커스텀 시스템 프롬프트
        system_prompt = """
        당신은 판타지 RPG 게임 '로어리스(Loreless)'의 NPC입니다.
        당신의 역할은 플레이어(주인공)와 대화하면서 그들의 성격, 가치관, 선호하는 행동 방식 등을 알아내는 것입니다.
        플레이어의 응답을 분석하여 그들의 특성을 파악하고, 다음 질문으로 이어가세요.
        항상 질문 형태로 대화를 이어가서 플레이어가 자신의 특성을 드러낼 수 있도록 유도하세요.
        이전 대화 내용을 참고하여 연속성 있는 대화를 만들어주세요.
        """
        
        # 기본 메시지 구성 (시스템 프롬프트)
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # 사용자의 이전 대화 기록 가져오기
        history = self.get_conversation_history(user_id)
        
        # 이전 대화 기록이 있으면 추가
        if history:
            # 첫 메시지로 컨텍스트 정보 제공
            messages.append({"role": "system", "content": f"대화 컨텍스트: {context}"})
            
            # 이전 대화 기록 추가 (최대 10개의 메시지로 제한)
            formatted_history = self._format_chat_history(history[-10:])
            messages.extend(formatted_history)
        
        # 현재 사용자 질문 추가
        messages.append({"role": "user", "content": question})
        
        # 사용자 질문 저장
        self.add_conversation(
            user_id=user_id,
            role="user",
            content=question
        )
        
        # 응답을 모으기 위한 변수
        full_response = []
        
        # 스트리밍 응답 생성
        response = openai.chat.completions.create(
            model="gpt-4.1-nano",
            messages=messages,
            temperature=0.7,
            stream=True
        )
        
        for chunk in response:
            token = chunk.choices[0].delta.content
            if token:
                full_response.append(token)
                yield token
            
        # 완성된 응답 저장
        self.add_conversation(
            user_id=user_id,
            role="npc",
            content="".join(full_response)
        )
    