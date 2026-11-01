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

# NLLB-200 translation model codes (FLORES-200). Whisper's Latin, Breton and Hawaiian have none.
_FLORES = {
    "en": "eng_Latn", "zh": "zho_Hans", "de": "deu_Latn", "es": "spa_Latn", "ru": "rus_Cyrl", "ko": "kor_Hang",
    "fr": "fra_Latn", "ja": "jpn_Jpan", "pt": "por_Latn", "tr": "tur_Latn", "pl": "pol_Latn", "ca": "cat_Latn",
    "nl": "nld_Latn", "ar": "arb_Arab", "sv": "swe_Latn", "it": "ita_Latn", "id": "ind_Latn", "hi": "hin_Deva",
    "fi": "fin_Latn", "vi": "vie_Latn", "he": "heb_Hebr", "uk": "ukr_Cyrl", "el": "ell_Grek", "ms": "zsm_Latn",
    "cs": "ces_Latn", "ro": "ron_Latn", "da": "dan_Latn", "hu": "hun_Latn", "ta": "tam_Taml", "no": "nob_Latn",
    "th": "tha_Thai", "ur": "urd_Arab", "hr": "hrv_Latn", "bg": "bul_Cyrl", "lt": "lit_Latn", "mi": "mri_Latn",
    "ml": "mal_Mlym", "cy": "cym_Latn", "sk": "slk_Latn", "te": "tel_Telu", "fa": "pes_Arab", "lv": "lvs_Latn",
    "bn": "ben_Beng", "sr": "srp_Cyrl", "az": "azj_Latn", "sl": "slv_Latn", "kn": "kan_Knda", "et": "est_Latn",
    "mk": "mkd_Cyrl", "eu": "eus_Latn", "is": "isl_Latn", "hy": "hye_Armn", "ne": "npi_Deva", "mn": "khk_Cyrl",
    "bs": "bos_Latn", "kk": "kaz_Cyrl", "sq": "als_Latn", "sw": "swh_Latn", "gl": "glg_Latn", "mr": "mar_Deva",
    "pa": "pan_Guru", "si": "sin_Sinh", "km": "khm_Khmr", "sn": "sna_Latn", "yo": "yor_Latn", "so": "som_Latn",
    "af": "afr_Latn", "oc": "oci_Latn", "ka": "kat_Geor", "be": "bel_Cyrl", "tg": "tgk_Cyrl", "sd": "snd_Arab",
    "gu": "guj_Gujr", "am": "amh_Ethi", "yi": "ydd_Hebr", "lo": "lao_Laoo", "uz": "uzn_Latn", "fo": "fao_Latn",
    "ht": "hat_Latn", "ps": "pbt_Arab", "tk": "tuk_Latn", "nn": "nno_Latn", "mt": "mlt_Latn", "sa": "san_Deva",
    "lb": "ltz_Latn", "my": "mya_Mymr", "bo": "bod_Tibt", "tl": "tgl_Latn", "mg": "plt_Latn", "as": "asm_Beng",
    "tt": "tat_Cyrl", "ln": "lin_Latn", "ha": "hau_Latn", "ba": "bak_Cyrl", "jw": "jav_Latn", "su": "sun_Latn",
    "yue": "yue_Hant",
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


def nllb_code(code):
    """'bn' -> 'ben_Beng', the code the NLLB-200 translation model uses (None if it can't translate it)."""
    return _FLORES.get(code)
