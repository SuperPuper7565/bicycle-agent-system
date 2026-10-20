"""Pydantic-схемы структурированных ответов агентов."""
from typing import Literal

from pydantic import BaseModel, Field

Category = Literal["mountain", "road", "city", "hybrid", "kids"]


class Requirements(BaseModel):
    """Требования клиента, извлечённые аналитиком."""

    intent: Literal["order", "consult", "other"] = Field(
        description="order: подобрать/купить велосипед; consult: вопрос о выборе, размерах, "
        "доставке, оплате, гарантии; other: приветствие или не по теме"
    )
    category: Category | None = Field(default=None, description="Тип велосипеда или null")
    quantity: int = Field(default=1, description="Количество велосипедов")
    max_price: int | None = Field(default=None, description="Максимум за ОДИН велосипед, руб.")
    wheel_size: str | None = Field(default=None, description='Диаметр колёс в дюймах, например "29"')
    must_have: list[str] = Field(default_factory=list, description="Обязательные требования")
    wishes: list[str] = Field(default_factory=list, description="Необязательные пожелания")
    question: str | None = Field(default=None, description="Уточняющий вопрос клиенту или null")


class Route(BaseModel):
    """Решение supervisor."""

    next: str = Field(description="Имя следующего исполнителя из списка допустимых")
    reason: str = Field(description="Одно предложение: почему выбран этот исполнитель")


class ProductCheck(BaseModel):
    url: str = Field(description="Ссылка товара (как во входных данных)")
    suitable: bool = Field(description="false, только если не выполнено обязательное требование")
    met: list[str] = Field(default_factory=list, description="Выполненные требования и пожелания")
    unmet: list[str] = Field(default_factory=list, description="Невыполненные требования и пожелания")
    comment: str = Field(default="", description="Короткое пояснение")
    score: int = Field(default=5, description="Оценка соответствия от 0 до 10")


class CheckReport(BaseModel):
    checks: list[ProductCheck]


class OfferText(BaseModel):
    intro: str = Field(description="1-2 предложения вступления")
    recommendation: str = Field(description="2-3 предложения: какой вариант лучше и почему")
    conditions: str = Field(description="Условия доставки, сборки, оплаты, гарантии из базы знаний")
