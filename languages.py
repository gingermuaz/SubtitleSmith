"""Language names (for the detected-language message) and subtitle line lengths per language."""

# Whisper's language codes and their names
NAMES = {
    "en": "English", "zh": "Chinese", "de": "German", "es": "Spanish", "ru": "Russian", "ko": "Korean",
    "fr": "French", "ja": "Japanese", "pt": "Portuguese", "tr": "Turkish", "pl": "Polish", "ca": "Catalan",
    "nl": "Dutch", "ar": "Arabic", "sv": "Swedish", "it": "Italian", "id": "Indonesian", "hi": "Hindi",
    "fi": "Finnish", "vi": "Vietnamese", "he": "Hebrew", "uk": "Ukrainian", "el": "Greek", "ms": "Malay",
    "cs": "Czech", "ro": "Romanian", "da": "Danish", "hu": "Hungarian", "ta": "Tamil", "no": "Norwegian",
    "th": "Thai", "ur": "Urdu", "hr": "Croatian", "bg": "Bulgarian", "lt": "Lithuanian", "la": "Latin",
    "mi": "Maori", "ml": "Malayalam", "cy": "Welsh", "sk": "Slovak", "te": "Telugu", "fa": "Persian",
    "lv": "Latvian", "bn": "Bengali", "sr": "Serbian", "az": "Azerbaijani", "sl": "Slovenian", "kn": "Kannada",
    "et": "Estonian", "mk": "Macedonian", "br": "Breton", "eu": "Basque", "is": "Icelandic", "hy": "Armenian",
    "ne": "Nepali", "mn": "Mongolian", "bs": "Bosnian", "kk": "Kazakh", "sq": "Albanian", "sw": "Swahili",
    "gl": "Galician", "mr": "Marathi", "pa": "Punjabi", "si": "Sinhala", "km": "Khmer", "sn": "Shona",
    "yo": "Yoruba", "so": "Somali", "af": "Afrikaans", "oc": "Occitan", "ka": "Georgian", "be": "Belarusian",
    "tg": "Tajik", "sd": "Sindhi", "gu": "Gujarati", "am": "Amharic", "yi": "Yiddish", "lo": "Lao",
    "uz": "Uzbek", "fo": "Faroese", "ht": "Haitian Creole", "ps": "Pashto", "tk": "Turkmen", "nn": "Nynorsk",
    "mt": "Maltese", "sa": "Sanskrit", "lb": "Luxembourgish", "my": "Myanmar", "bo": "Tibetan", "tl": "Tagalog",
    "mg": "Malagasy", "as": "Assamese", "tt": "Tatar", "haw": "Hawaiian", "ln": "Lingala", "ha": "Hausa",
    "ba": "Bashkir", "jw": "Javanese", "su": "Sundanese", "yue": "Cantonese",
}

# 3-letter ISO 639-2 codes, which video files use to label subtitle tracks (bibliographic form, as MKV uses)
_ISO639_2 = {
    "en": "eng", "zh": "chi", "de": "ger", "es": "spa", "ru": "rus", "ko": "kor", "fr": "fre", "ja": "jpn",
    "pt": "por", "tr": "tur", "pl": "pol", "ca": "cat", "nl": "dut", "ar": "ara", "sv": "swe", "it": "ita",
    "id": "ind", "hi": "hin", "fi": "fin", "vi": "vie", "he": "heb", "uk": "ukr", "el": "gre", "ms": "may",
    "cs": "cze", "ro": "rum", "da": "dan", "hu": "hun", "ta": "tam", "no": "nor", "th": "tha", "ur": "urd",
    "hr": "hrv", "bg": "bul", "lt": "lit", "la": "lat", "mi": "mao", "ml": "mal", "cy": "wel", "sk": "slo",
    "te": "tel", "fa": "per", "lv": "lav", "bn": "ben", "sr": "srp", "az": "aze", "sl": "slv", "kn": "kan",
    "et": "est", "mk": "mac", "br": "bre", "eu": "baq", "is": "ice", "hy": "arm", "ne": "nep", "mn": "mon",
    "bs": "bos", "kk": "kaz", "sq": "alb", "sw": "swa", "gl": "glg", "mr": "mar", "pa": "pan", "si": "sin",
    "km": "khm", "sn": "sna", "yo": "yor", "so": "som", "af": "afr", "oc": "oci", "ka": "geo", "be": "bel",
    "tg": "tgk", "sd": "snd", "gu": "guj", "am": "amh", "yi": "yid", "lo": "lao", "uz": "uzb", "fo": "fao",
    "ht": "hat", "ps": "pus", "tk": "tuk", "nn": "nno", "mt": "mlt", "sa": "san", "lb": "ltz", "my": "bur",
    "bo": "tib", "tl": "tgl", "mg": "mlg", "as": "asm", "tt": "tat", "haw": "haw", "ln": "lin", "ha": "hau",
    "ba": "bak", "jw": "jav", "su": "sun", "yue": "chi",  # ISO 639-2 has no Cantonese; Chinese is closest
}

# Characters per subtitle line. Languages written without spaces between words get shorter lines.
DEFAULT_LINE_LENGTH = 42
_LINE_LENGTHS = {"ja": 13, "zh": 16, "yue": 16, "ko": 16}


def name(code):
    """'ja' -> 'Japanese' (unknown codes are returned as they are)."""
    return NAMES.get(code, code)


def line_length(code):
    """Maximum characters per subtitle line for text in this language."""
    return _LINE_LENGTHS.get(code, DEFAULT_LINE_LENGTH)


def describe(code, probability):
    """'ja', 0.97 -> 'Japanese (97%)'"""
    return f"{name(code)} ({probability:.0%})"


def iso639_2(code):
    """'ja' -> 'jpn', for labelling subtitle tracks ('und' = undetermined, if unknown)."""
    return _ISO639_2.get(code, "und")
