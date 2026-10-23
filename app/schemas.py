"""Pydantic-схемы структурированных ответов агентов."""
from typing import Literal

from pydantic import BaseModel, Field

Category = Literal["mountain", "road", "city", "hybrid", "kids"]


class Requirements(BaseModel):
    """Данные из ПОСЛЕДНЕГО сообщения клиента (что переносить из прошлых, решает код)."""

    intent: Literal["order", "consult", "other"] = Field(
        description="order: подобрать/купить велосипед; consult: вопрос о выборе, размерах, "
        "доставке, оплате, гарантии; other: приветствие или не по теме"
    )
    category: Category | None = Field(
        default=None,
        description="Тип велосипеда, только если назван или однозначно следует из слов клиента; иначе null",
    )
    for_child: bool = Field(default=False, description="true, если велосипед для ребёнка (сын, дочь, внук, ребёнок)")
    child_age: int | None = Field(default=None, description="Возраст ребёнка в годах, только если назван")
    child_height_cm: int | None = Field(default=None, description="Рост ребёнка в см, только если назван")
    quantity: int | None = Field(default=None, description="Количество велосипедов, только если названо")
    max_price: int | None = Field(default=None, description="Максимум за ОДИН велосипед, руб., только если назван")
    wheel_size: str | None = Field(default=None, description='Диаметр колёс в дюймах, например "29", только если назван')
    must_have: list[str] = Field(default_factory=list, description="Обязательные требования клиента")
    wishes: list[str] = Field(default_factory=list, description="Необязательные пожелания клиента")
    refines_previous: bool = Field(
        default=False,
        description="true, если сообщение меняет или дополняет прошлую заявку («а подешевле», "
        "«а с дисковыми тормозами»), а не начинает новую",
    )


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
    advice: str = Field(description="2-4 коротких пункта «на что обратить внимание при выборе» по справке из базы знаний; пустая строка, если справки нет")
