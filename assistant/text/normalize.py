"""Normalizes text before it's handed to TTS: digits (Bengali or English),
dates, currency, percentages and common abbreviations become spoken Bengali
words, since VITS-style TTS models read digits/symbols poorly or not at all.

CAUTION: the 0-99 word table below was written from general knowledge of
Bengali numerals, not verified against a native-speaker reference or a
dictionary API (no such check was available while building this offline).
Bengali counting 1-99 is irregular (not composable like English
"twenty-one"), so a single wrong entry is easy to miss. Spot-check this
table — especially the 50-99 range — before relying on it for anything
user-facing.
"""
import re

_BN_DIGITS = "০১২৩৪৫৬৭৮৯"
_ASCII_DIGITS = "0123456789"
_BN_TO_ASCII = str.maketrans(_BN_DIGITS, _ASCII_DIGITS)
_ASCII_TO_BN = str.maketrans(_ASCII_DIGITS, _BN_DIGITS)

# 0-99, spoken Bengali words. See module caution above.
_UNITS = [
    "শূন্য", "এক", "দুই", "তিন", "চার", "পাঁচ", "ছয়", "সাত", "আট", "নয়",
    "দশ", "এগারো", "বারো", "তেরো", "চৌদ্দ", "পনেরো", "ষোলো", "সতেরো", "আঠারো", "উনিশ",
    "বিশ", "একুশ", "বাইশ", "তেইশ", "চব্বিশ", "পঁচিশ", "ছাব্বিশ", "সাতাশ", "আটাশ", "ঊনত্রিশ",
    "ত্রিশ", "একত্রিশ", "বত্রিশ", "তেত্রিশ", "চৌত্রিশ", "পঁয়ত্রিশ", "ছত্রিশ", "সাঁইত্রিশ", "আটত্রিশ", "ঊনচল্লিশ",
    "চল্লিশ", "একচল্লিশ", "বিয়াল্লিশ", "তেতাল্লিশ", "চুয়াল্লিশ", "পঁয়তাল্লিশ", "ছেচল্লিশ", "সাতচল্লিশ", "আটচল্লিশ", "ঊনপঞ্চাশ",
    "পঞ্চাশ", "একান্ন", "বাহান্ন", "তিপ্পান্ন", "চুয়ান্ন", "পঞ্চান্ন", "ছাপ্পান্ন", "সাতান্ন", "আটান্ন", "ঊনষাট",
    "ষাট", "একষট্টি", "বাষট্টি", "তেষট্টি", "চৌষট্টি", "পঁয়ষট্টি", "ছেষট্টি", "সাতষট্টি", "আটষট্টি", "ঊনসত্তর",
    "সত্তর", "একাত্তর", "বাহাত্তর", "তিয়াত্তর", "চুয়াত্তর", "পঁচাত্তর", "ছিয়াত্তর", "সাতাত্তর", "আটাত্তর", "ঊনআশি",
    "আশি", "একাশি", "বিরাশি", "তিরাশি", "চুরাশি", "পঁচাশি", "ছিয়াশি", "সাতাশি", "আটাশি", "ঊননব্বই",
    "নব্বই", "একানব্বই", "বিরানব্বই", "তিরানব্বই", "চুরানব্বই", "পঁচানব্বই", "ছিয়ানব্বই", "সাতানব্বই", "আটানব্বই", "নিরানব্বই",
]

_MONTHS = [
    "জানুয়ারি", "ফেব্রুয়ারি", "মার্চ", "এপ্রিল", "মে", "জুন",
    "জুলাই", "আগস্ট", "সেপ্টেম্বর", "অক্টোবর", "নভেম্বর", "ডিসেম্বর",
]

# Common abbreviations expanded before number handling runs.
_ABBREVIATIONS = {
    "ড.": "ডক্টর",
    "মি.": "মিস্টার",
    "মিসেস.": "মিসেস",
    "কি.মি.": "কিলোমিটার",
    "কি.মি": "কিলোমিটার",
    "সে.মি.": "সেন্টিমিটার",
    "ইঞ্জি.": "ইঞ্জিনিয়ার",
    "প্রফে.": "প্রফেসর",
    "ইত্যাদি.": "ইত্যাদি",
}


def _int_to_words(n: int) -> str:
    if n == 0:
        return _UNITS[0]
    if n < 0:
        return "ঋণাত্মক " + _int_to_words(-n)

    parts = []
    crore, n = divmod(n, 10**7)
    lakh, n = divmod(n, 10**5)
    thousand, n = divmod(n, 1000)
    hundred, n = divmod(n, 100)
    remainder = n

    if crore:
        parts.append(f"{_two_digit_words(crore)} কোটি")
    if lakh:
        parts.append(f"{_two_digit_words(lakh)} লাখ")
    if thousand:
        parts.append(f"{_two_digit_words(thousand)} হাজার")
    if hundred:
        parts.append(f"{_UNITS[hundred]}শ")
    if remainder:
        parts.append(_UNITS[remainder])

    return " ".join(parts)


def _two_digit_words(n: int) -> str:
    # crore/lakh/thousand multipliers beyond 99 aren't handled (not a
    # realistic case for spoken conversational replies).
    if 0 <= n <= 99:
        return _UNITS[n]
    return str(n)


def _digits_to_bengali(s: str) -> str:
    return s.translate(_ASCII_TO_BN)


def number_to_bengali_words(number_str: str) -> str:
    """Convert an integer or decimal numeral string (ASCII or Bengali
    digits) into spoken Bengali words."""
    ascii_str = number_str.translate(_BN_TO_ASCII)
    if "." in ascii_str:
        int_part, frac_part = ascii_str.split(".", 1)
        int_words = _int_to_words(int(int_part)) if int_part not in ("", "-") else _UNITS[0]
        digit_words = " ".join(_UNITS[int(d)] for d in frac_part if d.isdigit())
        return f"{int_words} দশমিক {digit_words}"
    return _int_to_words(int(ascii_str))


_NUMBER_RE = re.compile(rf"[{_BN_DIGITS}{_ASCII_DIGITS}]+(?:\.[{_BN_DIGITS}{_ASCII_DIGITS}]+)?")
_CURRENCY_RE = re.compile(rf"৳\s?([{_BN_DIGITS}{_ASCII_DIGITS}]+(?:\.[{_BN_DIGITS}{_ASCII_DIGITS}]+)?)")
_TAKA_SUFFIX_RE = re.compile(
    rf"([{_BN_DIGITS}{_ASCII_DIGITS}]+(?:\.[{_BN_DIGITS}{_ASCII_DIGITS}]+)?)\s?টাকা"
)
_PERCENT_RE = re.compile(rf"([{_BN_DIGITS}{_ASCII_DIGITS}]+(?:\.[{_BN_DIGITS}{_ASCII_DIGITS}]+)?)\s?%")
_DATE_RE = re.compile(
    rf"([{_BN_DIGITS}{_ASCII_DIGITS}]{{1,2}})[/\-.]([{_BN_DIGITS}{_ASCII_DIGITS}]{{1,2}})"
    rf"[/\-.]([{_BN_DIGITS}{_ASCII_DIGITS}]{{2,4}})"
)


def _expand_abbreviations(text: str) -> str:
    for abbr, expansion in _ABBREVIATIONS.items():
        text = text.replace(abbr, expansion)
    return text


def _normalize_dates(text: str) -> str:
    def repl(m: re.Match) -> str:
        day, month, year = m.groups()
        try:
            month_idx = int(month.translate(_BN_TO_ASCII)) - 1
            month_name = _MONTHS[month_idx] if 0 <= month_idx <= 11 else month
        except (ValueError, IndexError):
            return m.group(0)
        day_words = number_to_bengali_words(day)
        year_words = number_to_bengali_words(year)
        return f"{day_words} {month_name}, {year_words}"

    return _DATE_RE.sub(repl, text)


def _normalize_currency(text: str) -> str:
    text = _CURRENCY_RE.sub(lambda m: f"{number_to_bengali_words(m.group(1))} টাকা", text)
    text = _TAKA_SUFFIX_RE.sub(lambda m: f"{number_to_bengali_words(m.group(1))} টাকা", text)
    return text


def _normalize_percent(text: str) -> str:
    return _PERCENT_RE.sub(lambda m: f"{number_to_bengali_words(m.group(1))} শতাংশ", text)


def _normalize_bare_numbers(text: str) -> str:
    return _NUMBER_RE.sub(lambda m: number_to_bengali_words(m.group(0)), text)


def normalize_text(text: str) -> str:
    """Full pipeline: abbreviations -> dates -> currency -> percent -> bare
    digits. Order matters — each specific handler must consume its digits
    before the generic bare-number pass runs on what's left. English words
    embedded in an otherwise-Bengali sentence are passed through unchanged;
    Latin-script handling depends entirely on the chosen TTS engine and is
    not otherwise addressed here.
    """
    text = _expand_abbreviations(text)
    text = _normalize_dates(text)
    text = _normalize_currency(text)
    text = _normalize_percent(text)
    text = _normalize_bare_numbers(text)
    return text
