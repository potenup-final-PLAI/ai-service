from pydantic import BaseModel, Field
from typing import List, Optional, Tuple, Literal

# 캐릭터 공통 필드
class CharacterBase(BaseModel):
    id: str

# 전투 시작 요청용
class CharacterConfig(CharacterBase):
    name: str
    type: Literal["monster", "player"]
    traits: List[str]
    skills: List[str]

class BattleInitRequest(BaseModel):
    characters: List[CharacterConfig]
    terrain: str
    weather: str

# 전투 판단 요청용
class CharacterState(CharacterBase):
    position: Tuple[int, int] = Field(description="캐릭터의 좌표")
    hp: int = Field(description="캐릭터의 생명력")
    ap: int = Field(description="캐릭터의 행동 포인트")
    mov: int = Field(description="캐릭터의 이동력")
    status_effects: List[str] = Field(description="캐릭터의 상태 효과")

class BattleState(BaseModel):
    characters: List[CharacterState]
    cycle: int
    turn: int
    current_character_id: str

# 전투 판단 응답용 (LLM → 행동 판단)
class CharacterAction(BaseModel):
    move_to: Tuple[int, int] = Field(description="이동 할 좌표(이동이 하지 않는 경우 현재 좌표)")
    skill: Optional[str] = Field(default=None, description="사용할 스킬의 이름")
    target_character_id: Optional[str] = Field(default=None, description="스킬을 사용할 대상의 ID (이동 후 스킬의 사거리 내에 있어야 함)")
    reason: Optional[str] = Field(default=None, description="행동 선택 이유")
    dialogue: Optional[str] = Field(default=None, description="행동을 하는 캐릭터의 대사")
    remaining_ap: int = Field(default=None, description="남은 AP")
    remaining_mov: int = Field(default=None, description="남은 MOV")

class BattleActionResponse(BaseModel):
    current_character_id: str = Field(description="현재 행동 대상 캐릭터의 ID")
    # actions: List[CharacterAction] = Field(description="해당 턴에 사용하는 캐릭터의 행동 목록 (최대한 많은 행동을 수행하는 것이 중요)")
    action: CharacterAction = Field(description="해당 턴에 사용하는 캐릭터의 행동")
    

# AI 판단 용 모델
class CharacterForAI(CharacterConfig, CharacterState):
    distance: int = Field(description="현재 캐릭터와 대상 캐릭터 사이의 거리")

class BattleStateForAI(BaseModel):
    characters: List[CharacterForAI]
    cycle: int
    turn: int
    current_character_id: str
    terrain: str
    weather: str

# LLM 전략 결정 모델
class StrategyDecision(BaseModel):
    strategy_type: str = Field(description="전략 유형 (공격적/방어적/지원형/기동형)")
    priority_target: str = Field(description="우선순위 타겟의 ID ('없음'인 경우도 가능)")
    action_plan: str = Field(description="구체적인 행동 계획 설명")
    reason: str = Field(description="이 전략을 선택한 이유")
    