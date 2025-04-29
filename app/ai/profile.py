from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import json

from app.utils.loader import traits, skills, prompt_profile_generation
from app.models.characters import CharacterStatsUpdateRequest

# 스토리 설정 정보
LORELESS_SETTING = """
## Loreless 세계관 요약

Loreless는 '기억의 검'이라는 부제를 가진 판타지 RPG로, 기억을 잃은 주인공 '카엔'의 여정을 다룹니다.

- **배경**: 카엔은 안개 낀 마을 '카일름'에서 깨어나며, 유일한 단서는 검은 매 문양이 새겨진 천 조각입니다.
- **세계**: 고대 문명 '아르벨란'의 흔적이 남아있는 판타지 세계로, 다양한 지역(파란디온 폐허, 브라에 사막, 나르센 숲 등)이 존재합니다.
- **주제**: 기억과 정체성, 선택과 책임에 관한 이야기입니다.

캐릭터 프로필에서는 카엔의 잃어버린 과거를 재구성할 수 있으며, 이는 NPC와의 대화에서 드러난 성격과 선호도를 기반으로 합니다.
"""

class CharacterProfile(BaseModel):
    """AI 모델이 생성한 캐릭터 프로필 정보"""
    name: str = Field(..., description="캐릭터의 이름")
    gender: str = Field(..., description="캐릭터의 성별 (M 또는 F)")
    traits: List[str] = Field(..., description="캐릭터의 특성 (최대 2개)")
    stats: Dict[str, Any] = Field(..., description="캐릭터의 스탯 정보")
    skills: List[str] = Field(..., description="캐릭터가 사용할 수 있는 초기 스킬 (최대 3개)")
    backstory: str = Field(..., description="캐릭터의 간략한 배경 이야기")
    reason: str = Field(..., description="특성, 스탯, 스킬 선택의 이유 (내부 로깅용)")

class ProfileAI:
    def __init__(
        self, 
        model_name="gpt-4o-mini", 
        temperature=0.7
    ):
        self.parser = PydanticOutputParser(pydantic_object=CharacterProfile)
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.prompt = PromptTemplate.from_template(prompt_profile_generation).partial(
            format=self.parser.get_format_instructions(),
            world_setting=LORELESS_SETTING
        )
        
        self.chain = self.prompt | self.llm | self.parser
        
    def get_available_traits(self) -> List[Dict[str, Any]]:
        """사용 가능한 특성 목록과 설명을 반환합니다"""
        trait_list = []
        for name, data in traits.items():
            trait_list.append({
                "name": name,
                "description": data.get("description", ""),
                "stat_changes": data.get("stat_cng", {})
            })
        return trait_list
    
    def get_available_skills(self) -> List[Dict[str, Any]]:
        """사용 가능한 스킬 목록과 설명을 반환합니다"""
        skill_list = []
        for name, data in skills.items():
            skill_list.append({
                "name": name,
                "description": data.get("description", ""),
                "ap": data.get("ap", 0),
                "range": data.get("range", 1),
                "dmg_mult": data.get("dmg_mult", 0),
                "effects": data.get("effects", [])
            })
        return skill_list
    
    def format_traits_for_prompt(self) -> str:
        """특성 정보를 프롬프트에 사용할 형식으로 변환합니다"""
        formatted_traits = []
        for name, data in traits.items():
            stat_changes = data.get("stat_cng", {})
            stat_str = ", ".join([f"{k}: {v:+.1f}" for k, v in stat_changes.items()])
            formatted_traits.append(f"- {name}: {data.get('description', '')} [변화: {stat_str}]")
        return "\n".join(formatted_traits)
    
    def format_skills_for_prompt(self) -> str:
        """스킬 정보를 프롬프트에 사용할 형식으로 변환합니다"""
        formatted_skills = []
        for name, data in skills.items():
            effects = ", ".join(data.get("effects", []))
            if not effects:
                effects = "효과 없음"
            formatted_skills.append(
                f"- {name}: {data.get('description', '')} "
                f"[AP: {data.get('ap', 0)}, 사거리: {data.get('range', 1)}, "
                f"공격력: {data.get('dmg_mult', 0)}, 효과: {effects}]"
            )
        return "\n".join(formatted_skills)
    
    async def generate_character_profile(self, conversation_history: List[Dict[str, str]]) -> CharacterProfile:
        """대화 기록을 기반으로 캐릭터 프로필을 생성합니다"""
        
        # 대화 기록을 텍스트로 변환
        conversation_text = ""
        for message in conversation_history:
            role = message.get("role", "unknown")
            content = message.get("content", "")
            conversation_text += f"{role}: {content}\n\n"
        
        # 특성 및 스킬 정보 포맷팅
        traits_text = self.format_traits_for_prompt()
        skills_text = self.format_skills_for_prompt()
        
        # AI 모델에 전달할 컨텍스트 생성
        result = await self.chain.ainvoke({
            "conversation_history": conversation_text,
            "traits_info": traits_text,
            "skills_info": skills_text
        })
        
        return result
    
    def convert_to_character_stats(self, profile: CharacterProfile) -> CharacterStatsUpdateRequest:
        """생성된 프로필을 CharacterStatsUpdateRequest 형식으로 변환합니다"""
        stats = profile.stats
        
        return CharacterStatsUpdateRequest(
            character_id=None,  # 이 부분은 서비스 레이어에서 설정
            hp=stats.get("hp", 100),
            attack=stats.get("attack", 10),
            defense=stats.get("defense", 10),
            resistance=stats.get("resistance", 10),
            critical_rate=stats.get("critical_rate", 0.05),
            critical_damage=stats.get("critical_damage", 1.5),
            move_range=stats.get("move_range", 4),
            speed=stats.get("speed", 10),
            points=stats.get("points", 0)
        )
