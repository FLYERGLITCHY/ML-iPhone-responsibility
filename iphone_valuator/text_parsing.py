"""Regex extraction of iPhone attributes and junk signals from Russian-language listing text.

Every public function accepts arbitrary objects (``None``, ``NaN`` from pandas, raw strings) and
returns ``None``/``False`` instead of raising when nothing trustworthy can be extracted.
"""

from __future__ import annotations

import math
import numbers
import re
from collections.abc import Mapping
from enum import StrEnum
from typing import Final

from iphone_valuator.config import MAX_BATTERY_HEALTH, MIN_BATTERY_HEALTH, UNKNOWN_REGION
from iphone_valuator.domain import Condition, get_model


class JunkReason(StrEnum):
    """Why a listing must be excluded from the fair-market training data."""

    LOCKED = "locked"
    FOR_PARTS = "for_parts"
    BROKEN_SCREEN = "broken_screen"
    WATER_DAMAGE = "water_damage"
    FAKE = "fake"
    ACCESSORY = "accessory"
    NOT_FOR_SALE = "not_for_sale"


_HORIZONTAL_SPACE_RE: Final = re.compile(r"[^\S\n]+")
_NEWLINES_RE: Final = re.compile(r"\s*\n\s*")


def normalize_text(text: object) -> str:
    """Lower-case, unify ``ё``/``е`` and collapse whitespace while keeping line breaks."""
    if not isinstance(text, str) or not text:
        return ""
    unified = text.replace("ё", "е").replace("Ё", "Е").lower()
    unified = _HORIZONTAL_SPACE_RE.sub(" ", unified)
    return _NEWLINES_RE.sub("\n", unified).strip()


_CLAUSE_BREAK_RE: Final = re.compile(r"[.!?;\n]|\s(?:но|однако|but)\s")
_NEAR_NEGATION_RE: Final = re.compile(r"(?:^|[\s,(])(?:не|ни|not|never)\s+(?:\S+\s+)?$")
_FAR_NEGATION_RE: Final = re.compile(
    r"(?:^|[\s,(])(?:без|нет|no|without)\s+(?P<span>(?:\S+\s+){0,4})$"
)
_AFFIRMATION_RE: Final = re.compile(r"\b(?:есть|имеется|имеются|присутству\w*|but)\b")
_POST_NEGATION_RE: Final = re.compile(
    r"^(?:\s*(?:,|\bи\b|\bили\b)\s*[^\s,]+)*\s*[,:\-–—]?\s*(?:нет|отсутству\w*|no|none)(?!\w)"
)
_NEGATION_WINDOW: Final = 60


def _is_negated(text: str, start: int, end: int) -> bool:
    before = _CLAUSE_BREAK_RE.split(text[max(0, start - _NEGATION_WINDOW) : start])[-1]
    after = _CLAUSE_BREAK_RE.split(text[end : end + _NEGATION_WINDOW], maxsplit=1)[0]
    far = _FAR_NEGATION_RE.search(before)
    return bool(
        _NEAR_NEGATION_RE.search(before)
        or (far is not None and not _AFFIRMATION_RE.search(far.group("span")))
        or _POST_NEGATION_RE.search(after)
    )


def _search_unnegated(pattern: re.Pattern[str], text: str) -> bool:
    return any(not _is_negated(text, m.start(), m.end()) for m in pattern.finditer(text))


_MODEL_REPLACEMENTS: Final[tuple[tuple[re.Pattern[str], str], ...]] = tuple(
    (re.compile(pattern), replacement)
    for pattern, replacement in (
        (r"\bа[йи]фон\w*", "iphone"),
        (r"\bпро\s*-?\s*макс\b|\bпромакс\b", "pro max"),
        (r"\bпро\b", "pro"),
        (r"\bмакс\b|\bмах\b", "max"),
        (r"\bмини\b", "mini"),
        (r"\bплюс\b", "plus"),
        (r"\bэ[йи]р\b", "air"),
        (r"\bс[еe]\b", "se"),
        (r"\b[хx][рr]\b", "xr"),
        (r"\b[хx][сsc]\b", "xs"),
        (r"\bх\b", "x"),
        (r"(?<=\d)е\b", "e"),
        (r"\bр(?=ro\b|lus\b)", "p"),
    )
)
_MODEL_GENERATION: Final = r"(?P<gen>1[0-7](?!\d)|8(?!\d)|xs|xr|x(?![a-z])|se|air)"
_MODEL_VARIANT: Final = (
    r"(?:(?P<e_suffix>e)(?![a-z-])"
    r"|\s*-?\s*(?P<variant>pro\s*-?\s*max|promax|pro|plus|mini|max|air)(?![a-z])"
    r"|\s*(?P<plus_sign>\+))?"
)
_SE_GENERATION: Final = (
    r"(?:\s*\(?\s*(?P<se_gen>20(?:20|22)"
    r"|[23](?!\d)(?:\s*-?\s*(?:nd|rd|го|gen\w*|поколени\w*))?)\s*\)?)?"
)
_PREFIXED_MODEL_RE: Final = re.compile(
    rf"(?:\bi\s*-?\s*phone|\bapple)\s*-?\s*{_MODEL_GENERATION}{_MODEL_VARIANT}{_SE_GENERATION}"
)
_BARE_MODEL_RE: Final = re.compile(
    rf"(?<![a-z0-9]){_MODEL_GENERATION}{_MODEL_VARIANT}{_SE_GENERATION}"
)
_X_FAMILY: Final[dict[str, str]] = {"10": "X", "x": "X", "xr": "XR", "xs": "XS"}
_VARIANT_SUFFIXES: Final[dict[str, str]] = {
    "": "",
    "pro max": " Pro Max",
    "promax": " Pro Max",
    "pro": " Pro",
    "plus": " Plus",
    "mini": " mini",
    "air": " Air",
}


def _canonical_model_name(match: re.Match[str]) -> str | None:
    generation = match.group("gen")
    variant = (
        "plus" if match.group("plus_sign") else " ".join((match.group("variant") or "").split())
    )
    variant = variant.replace(" - ", " ").replace("-", " ")
    if generation == "se":
        se_generation = match.group("se_gen")
        if not se_generation:
            return None
        return "iPhone SE 2022" if se_generation.startswith(("2022", "3")) else "iPhone SE 2020"
    if generation == "air" or (generation == "17" and variant == "air"):
        return "iPhone Air"
    if generation in _X_FAMILY:
        suffix = " Max" if variant == "max" else _VARIANT_SUFFIXES.get(variant, " ?")
        return f"iPhone {_X_FAMILY[generation]}{suffix}"
    if match.group("e_suffix"):
        return f"iPhone {generation}e"
    if variant == "max":
        variant = "pro max"
    return f"iPhone {generation}{_VARIANT_SUFFIXES.get(variant, ' ?')}"


def parse_model(text: object, *, require_prefix: bool = True) -> str | None:
    """Extract the canonical model name: ``"Айфон 13 про макс"`` -> ``iPhone 13 Pro Max``.

    With ``require_prefix=False`` bare inputs such as ``"14 pro max"`` or ``"XR"`` are accepted,
    which suits structured attributes and interactive user input.
    """
    normalized = normalize_text(text)
    if not normalized:
        return None
    for pattern, replacement in _MODEL_REPLACEMENTS:
        normalized = pattern.sub(replacement, normalized)
    regex = _PREFIXED_MODEL_RE if require_prefix else _BARE_MODEL_RE
    for match in regex.finditer(normalized):
        model = get_model(_canonical_model_name(match))
        if model is not None:
            return model.name
    return None


_GB_VALUES: Final = frozenset({64, 128, 256, 512, 1024})
_TB_VALUES: Final = frozenset({1, 2})
STORAGE_PATTERN: Final = re.compile(
    r"(?<![\d.,])(?P<value>64|128|256|512|1024|1|2)\s*"
    r"(?P<unit>гб|gb|гиг\w*|gig\w*|тб|tb|терабайт\w*|terabyte\w*|g|г)(?![a-zа-я])"
)
_BARE_STORAGE_RE: Final = re.compile(
    r"(?<![\d.,])(?P<value>64|128|256|512)(?![\d.,%])(?!\s+\d)"
    r"(?!\s*(?:₽|руб|р\b|т\.?\s*р|тыс|k\b|к\b|мес|шт))"
)


def parse_storage(text: object, *, allow_bare: bool = False) -> int | None:
    """Extract storage capacity in GB (``64|128|256|512|1024`` GB or ``1|2`` TB).

    ``allow_bare`` additionally accepts unit-less capacities (``"iPhone 13 Pro 256"``), which is
    only safe for short titles, never for free-form descriptions.
    """
    normalized = normalize_text(text)
    if not normalized:
        return None
    for match in STORAGE_PATTERN.finditer(normalized):
        value = int(match.group("value"))
        if match.group("unit").startswith(("т", "t")):
            if value in _TB_VALUES:
                return value * 1024
        elif value in _GB_VALUES:
            return value
    if allow_bare:
        bare = _BARE_STORAGE_RE.search(normalized)
        if bare is not None:
            return int(bare.group("value"))
    return None


_BATTERY_KEYWORD: Final = (
    r"(?:акб|аккумулятор\w*|батаре\w*|емкост\w*|battery(?:\s+health)?|batt\b|"
    r"здоровье(?:\s+(?:акб|аккумулятора|батареи))?|max(?:imum)?\s+capacity)"
)
_BATTERY_GAP: Final = r"[^\d\n.!?;]{0,30}?"
_PERCENT: Final = r"\s*(?:%|процент\w*)"
_NON_BATTERY_UNIT: Final = (
    r"(?:цикл\w*|cycle\w*|раз\w*|мес\w*|month\w*|дн\w*|day\w*|час\w*|hour\w*|ч\b|мин\w*|"
    r"mah|мач|ма\b|шт\w*|руб\w*|₽|р\b|т\.?\s*р|тыс\w*|к\b|k\b|gb|гб|г\b|лет|год\w*)"
)
BATTERY_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(rf"{_BATTERY_KEYWORD}{_BATTERY_GAP}(?<!\d)(?P<value>\d{{2,3}}){_PERCENT}"),
    re.compile(rf"(?<!\d)(?P<value>\d{{2,3}}){_PERCENT}[^\d\n.!?;]{{0,15}}?{_BATTERY_KEYWORD}"),
    re.compile(
        rf"{_BATTERY_KEYWORD}\s*[:\-–—=]?\s*(?<!\d)(?P<value>\d{{2,3}})"
        rf"(?![\d%.,])(?!\s*{_NON_BATTERY_UNIT})"
    ),
)


def parse_battery_health(text: object) -> int | None:
    """Extract battery health (АКБ %) anchored to a battery keyword, e.g. ``"АКБ 87%"`` -> 87.

    Bare percentages are ignored on purpose: ``"100% оригинал"`` is not a battery reading.
    """
    normalized = normalize_text(text)
    if not normalized:
        return None
    for pattern in BATTERY_PATTERNS:
        for match in pattern.finditer(normalized):
            value = int(match.group("value"))
            if MIN_BATTERY_HEALTH <= value <= MAX_BATTERY_HEALTH:
                return value
    return None


_LOCKED_RE: Final = re.compile(
    r"activation\s*lock|icloud\s*lock|блок\w*\s+активац\w*|"
    r"(?:заблокирован\w*|залочен\w*)\s+(?:на\s+|к\s+|по\s+|под\s+)?(?:чуж\w+\s+)?"
    r"(?:icloud|айклауд\w*|айклоуд\w*|apple\s*id|аккаунт\w*|учетн\w+\s+запис\w*|оператор\w*)|"
    r"привязан\w*\s+(?:к\s+)?(?:чуж\w+\s+(?:icloud|айклауд\w*|айклоуд\w*|apple\s*id|аккаунт\w*|"
    r"учетн\w+\s+запис\w*)|оператор\w*)|"
    r"(?:icloud|айклауд\w*|айклоуд\w*|apple\s*id)\s+(?:заблокирован\w*|залочен\w*)|"
    r"\bна\s+(?:icloud|айклауд\w*|айклоуд\w*)\b|"
    r"(?:забыл\w*|не\s+помн\w*|не\s+зна\w*)\s+(?:пароль|код|apple\s*id|айди|учетк\w*)|"
    r"режим\w*\s+пропаж\w*|lost\s*mode|\bmdm\b|\bbypass\b|\bобход\w*\s+(?:icloud|активац\w*)|"
    r"\br-?sim\b|\bрсим\w*|турбо\s*сим\w*|sim[\s-]?lock|\bзалочен\w*"
)
_FOR_PARTS_RE: Final = re.compile(
    r"(?:на|по|под)\s+(?:запчаст\w*|з/ч|зч|детал\w*|разбор\w*)|\bдонор\w*|for\s+parts|"
    r"требует\s+ремонта|нуждается\s+в\s+ремонте|под\s+ремонт\b|"
    r"не\s+включа(?:ется|ются)\b|не\s+загружа(?:ется|ются)\b|не\s+заряжа(?:ется|ются)\b|"
    r"не\s+работа(?:ет|ют)\b|не\s+(?:ловит|видит)\s+(?:сеть|сим\w*)|нет\s+сети|"
    r"(?:face\s*id|фейс\s*айди)\s+(?:не\s+работа\w*|отвал\w*)|без\s+(?:face\s*id|фейс\s*айди)|"
    r"\bсломан\w*|неисправ\w*|\bbroken\b"
)
_BROKEN_SCREEN_RE: Final = re.compile(
    r"разбит\w*|\bбит(?:ый|ое|ая|ые)\b|треснут\w*|трещин[аыу]\b|в\s+трещинах|\bcracked\b|"
    r"(?:полос\w*|пятн\w*|битые\s+пиксели)\s+на\s+(?:экран\w*|диспле\w*)"
)
_WATER_DAMAGE_RE: Final = re.compile(
    r"\bутоп\w*|\bтонул\w*|после\s+(?:воды|утоплен\w*|влаги)|"
    r"(?:был\w*|побывал\w*|попадал\w*|попал\w*|падал\w*|упал\w*|уронил\w*|искупал\w*)\s+"
    r"(?:в\s+)?(?:(?:воде|воду|вода|водой)\b|ванн\w*|унитаз\w*|море\b|бассейн\w*|лужу)|"
    r"\bзалит(?:ый|ая|о|а)?\b|water\s*damage|\bкупался\b"
)
_FAKE_RE: Final = re.compile(
    r"реплик\w*|\breplica\b|муляж\w*|подделк\w*|\bfake\b|"
    r"\bкопи[яи]\b(?!\s+(?:чек\w*|документ\w*|коробк\w*|гарант\w*))|"
    r"(?:не\s*оригинал\w*|неоригинал\w*)\s+(?:iphone|айфон\w*|телефон\w*|смартфон\w*)"
)
_ACCESSORY_TITLE_RE: Final = re.compile(
    r"^\W*(?:чехол|чехлы|стекло|стекла|пленк\w*|кабель|провод\w*|зарядк\w*|зарядное|"
    r"блок\s+питания|адаптер\w*|коробк\w*|муляж\w*|корпус\w*|дисплей|дисплеи|экран\w*|"
    r"аккумулятор\w*|батаре\w*|акб|шлейф\w*|камер\w*|плат\w*|наушник\w*|airpods|"
    r"держател\w*|подставк\w*|кейс\w*|лоток|сим-?лоток|запчаст\w*|задн\w+\s+крышк\w*|крышк\w*)\b|"
    r"\b(?:для|под)\s+(?:apple\s+)?(?:iphone|айфон\w*)|"
    r"\b(?:ipad|macbook|imac|apple\s+watch|airpods|magsafe)\b"
)
_NOT_FOR_SALE_TITLE_RE: Final = re.compile(
    r"^\W*(?:куплю|скупка|скупаю|выкуп|срочный\s+выкуп|ремонт|замена|аренда|прокат|"
    r"обмен\w*|меняю|обменяю|ищу)\b|\bв\s+аренду\b|\bпод\s+заказ\b|\bпредзаказ\w*"
)
_TEXT_JUNK_RULES: Final[tuple[tuple[JunkReason, re.Pattern[str]], ...]] = (
    (JunkReason.LOCKED, _LOCKED_RE),
    (JunkReason.WATER_DAMAGE, _WATER_DAMAGE_RE),
    (JunkReason.BROKEN_SCREEN, _BROKEN_SCREEN_RE),
    (JunkReason.FOR_PARTS, _FOR_PARTS_RE),
    (JunkReason.FAKE, _FAKE_RE),
)
_DAMAGE_REASONS: Final = frozenset(
    {JunkReason.LOCKED, JunkReason.FOR_PARTS, JunkReason.BROKEN_SCREEN, JunkReason.WATER_DAMAGE}
)


def detect_junk(title: object, description: object = "") -> JunkReason | None:
    """Return the first junk signal found in a listing, honouring negations like ``"не битый"``.

    Accessory and not-for-sale checks only look at the title, because descriptions of genuine
    phones routinely mention bundled cases, glass or exchange options.
    """
    title_text = normalize_text(title)
    if _ACCESSORY_TITLE_RE.search(title_text):
        return JunkReason.ACCESSORY
    if _NOT_FOR_SALE_TITLE_RE.search(title_text):
        return JunkReason.NOT_FOR_SALE
    full_text = f"{title_text}\n{normalize_text(description)}".strip()
    for reason, pattern in _TEXT_JUNK_RULES:
        if _search_unnegated(pattern, full_text):
            return reason
    return None


def _has_damage_signal(text: str) -> bool:
    return any(
        _search_unnegated(pattern, text)
        for reason, pattern in _TEXT_JUNK_RULES
        if reason in _DAMAGE_REASONS
    )


_PARAM_CONDITION_RULES: Final[tuple[tuple[Condition, re.Pattern[str]], ...]] = (
    (
        Condition.FOR_PARTS,
        re.compile(
            r"for[\s_]parts|требует\s+ремонта|на\s+запчаст|не\s*рабоч|неисправ|сломан|broken"
        ),
    ),
    (Condition.REFURBISHED, re.compile(r"восстановлен|refurb")),
    (
        Condition.USED,
        re.compile(r"б\s*/\s*у|\bбу\b|\bused\b|отличн|хорош|удовлетвор|как\s+нов|идеальн"),
    ),
    (Condition.NEW, re.compile(r"\bнов|\bnew\b")),
)
_REFURBISHED_TEXT_RE: Final = re.compile(r"восстановлен\w*|refurb\w*|\bреф\b|\bcpo\b")
_STRONG_NEW_TEXT_RE: Final = re.compile(
    r"запечатан\w*|\bsealed\b|нераспакован\w*|не\s+распакован\w*|не\s+активирован\w*|brand\s+new"
)
_USED_TEXT_RE: Final = re.compile(
    r"б\s*/\s*у(?![а-я])|\bбу\b|пользовал\w*|использовал\w*|в\s+использовании"
)


def _condition_from_param(value: str) -> Condition | None:
    for condition, pattern in _PARAM_CONDITION_RULES:
        if pattern.search(value):
            return condition
    return None


def parse_condition(param_value: object, text: object = "") -> Condition | None:
    """Resolve the device condition from Avito's ``Состояние`` attribute plus free text.

    Damage signals in the text override the attribute (sellers often pick "Хорошее" for locked
    phones), refurbished mentions override "Новое", and used mentions override "Новое".
    """
    param_condition = _condition_from_param(normalize_text(param_value))
    body = normalize_text(text)
    if param_condition is Condition.FOR_PARTS or _has_damage_signal(body):
        return Condition.FOR_PARTS
    if param_condition is Condition.REFURBISHED or _search_unnegated(_REFURBISHED_TEXT_RE, body):
        return Condition.REFURBISHED
    if param_condition is Condition.NEW and _search_unnegated(_USED_TEXT_RE, body):
        return Condition.USED
    if param_condition is not None:
        return param_condition
    if _search_unnegated(_STRONG_NEW_TEXT_RE, body):
        return Condition.NEW
    if _search_unnegated(_USED_TEXT_RE, body):
        return Condition.USED
    return None


_BOX_RE: Final = re.compile(
    r"коробк\w*|полн\w*\s+комплект\w*|комплект\w*\s+полн\w*|родн\w*\s+упаковк\w*|\bbox\b|full\s+set"
)
_RECEIPT_RE: Final = re.compile(
    r"\bчек(?:а|ом|и|ов|у)?\b|документ\w*|квитанци\w*|\breceipt\b|гарантийн\w*\s+талон\w*"
)
_INSTALLMENT_RE: Final = re.compile(
    r"рассрочк\w*|кредит\w*|перв\w*\s+взнос\w*|первоначальн\w*\s+(?:взнос\w*|плат\w*)|"
    r"ежемесячн\w*|/\s*мес(?:яц)?\b|в\s+месяц|аренд\w*|предоплат\w*|trade[\s-]?in|трейд[\s-]?ин|"
    r"трейдин"
)


def has_original_box(text: object) -> bool:
    """True when the text mentions the original box / full set without negating it."""
    return _search_unnegated(_BOX_RE, normalize_text(text))


def has_receipt(text: object) -> bool:
    """True when the text mentions a receipt or purchase documents without negating it."""
    return _search_unnegated(_RECEIPT_RE, normalize_text(text))


def mentions_installment(text: object) -> bool:
    """True for installment, down-payment, rental or trade-in wording."""
    return _search_unnegated(_INSTALLMENT_RE, normalize_text(text))


_PRICE_NUMBER_RE: Final = re.compile(r"\d[\d ]*")


def parse_price(value: object) -> int | None:
    """Parse a RUB price from numbers or strings such as ``"54 990 ₽"``."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, numbers.Real):
        number = float(value)
        return round(number) if math.isfinite(number) and number > 0 else None
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    try:
        number = float(stripped)
    except ValueError:
        match = _PRICE_NUMBER_RE.search(_HORIZONTAL_SPACE_RE.sub(" ", stripped))
        if match is None:
            return None
        price = int(match.group().replace(" ", ""))
        return price if price > 0 else None
    return round(number) if math.isfinite(number) and number > 0 else None


_REGION_ALIASES: Final[dict[str, str]] = {
    "мск": "Москва",
    "москва": "Москва",
    "спб": "Санкт-Петербург",
    "питер": "Санкт-Петербург",
    "санкт-петербург": "Санкт-Петербург",
    "екб": "Екатеринбург",
    "нск": "Новосибирск",
}
_NOT_A_CITY_RE: Final = re.compile(
    r"^(?:м\.|метро|р-н|район|ул\.|улица|пр-т|проспект|пр\.|пер\.|ш\.|шоссе|мкр|д\.)|\d"
)
_CITY_PREFIX_RE: Final = re.compile(r"^(?:г\.|город\s)\s*", re.IGNORECASE)


def normalize_region(location: object) -> str:
    """Reduce an Avito address such as ``"г. Москва, ул. Тверская"`` to its city/region."""
    if not isinstance(location, str):
        return UNKNOWN_REGION
    head = re.split(r"[,;\n]", location.strip(), maxsplit=1)[0]
    head = " ".join(_CITY_PREFIX_RE.sub("", head.strip()).split())
    if not head or _NOT_A_CITY_RE.search(head.lower()):
        return UNKNOWN_REGION
    return _REGION_ALIASES.get(head.lower().replace("ё", "е"), head[0].upper() + head[1:])


def find_param(params: Mapping[str, str], *candidates: str) -> str | None:
    """Find a structured attribute by exact normalized key, then by key substring."""
    normalized = {normalize_text(key).rstrip(":"): value for key, value in params.items()}
    wanted = [normalize_text(candidate) for candidate in candidates]
    for candidate in wanted:
        if candidate in normalized:
            return normalized[candidate]
    for candidate in wanted:
        for key, value in normalized.items():
            if candidate in key:
                return value
    return None
