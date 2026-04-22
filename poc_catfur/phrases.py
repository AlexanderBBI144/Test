"""Phrase bank: diverse short English phrases used for semantic decoding.

Generated procedurally from a hand-curated POS-tagged lexicon and a bank
of sentence templates, plus a small set of natural hand-picked phrases.
The resulting phrases are embedded with OpenAI text-embedding-3-large at
native 24-d (Matryoshka truncation), so decoded vectors can be looked up
as whole phrases.
"""

import random
import re

NOUNS_CONCRETE: list[str] = [
    # animals
    "cats", "dogs", "birds", "horses", "fish", "ducks", "geese", "swans",
    "wolves", "bears", "eagles", "hawks", "lions", "tigers", "rabbits",
    "sheep", "cows", "frogs", "snakes", "owls", "deer", "foxes", "mice",
    "bees", "butterflies", "spiders", "whales", "dolphins", "crabs",
    "turtles", "doves", "crows", "sparrows", "pigeons", "cranes", "ravens",
    # nature
    "sky", "sun", "moon", "stars", "clouds", "wind", "rain", "snow",
    "rivers", "mountains", "forests", "oceans", "deserts", "lakes",
    "valleys", "meadows", "fields", "gardens", "flowers", "trees",
    "leaves", "roses", "lilies", "storms", "fire", "ice", "thunder",
    "waves", "shadows", "stones", "rocks", "sand", "dust",
    # places
    "home", "city", "village", "streets", "bridges", "towers", "castles",
    "beach", "islands", "hills", "caves", "roads", "paths",
    # body/person
    "mother", "father", "friend", "lover", "child", "brother", "sister",
    "baby", "heart", "eyes", "hands", "voice", "soul", "smile", "face",
    # objects / culture
    "books", "music", "songs", "stories", "dreams", "letters", "words",
    "pictures", "films", "windows", "doors", "mirrors", "candles",
    "bread", "wine", "tea", "coffee", "apples", "honey", "salt",
]

NOUNS_ABSTRACT: list[str] = [
    "love", "hope", "fear", "joy", "sadness", "anger", "peace", "war",
    "truth", "lies", "memory", "time", "silence", "beauty", "freedom",
    "justice", "faith", "hate", "pain", "pleasure", "courage", "magic",
    "luck", "life", "death", "birth", "youth", "kindness", "mercy",
    "laughter", "tears", "secrets",
]

NOUNS: list[str] = NOUNS_CONCRETE + NOUNS_ABSTRACT

ADJECTIVES: list[str] = [
    "beautiful", "lovely", "wonderful", "strange", "dark", "bright",
    "cold", "warm", "hot", "silent", "loud", "soft", "gentle",
    "wild", "tame", "happy", "sad", "sweet", "bitter", "young", "old",
    "fast", "slow", "small", "big", "tiny", "huge", "deep", "high",
    "low", "empty", "full", "lost", "found", "free", "alone", "quiet",
    "fierce", "proud", "humble", "mysterious", "terrible",
    "endless", "golden", "silver", "red", "blue", "green", "white", "black",
    "broken", "perfect", "hidden", "forgotten", "ancient", "new",
]

VERBS_PRESENT: list[str] = [
    "love", "like", "adore", "hate", "miss", "want", "need", "watch",
    "see", "hear", "feel", "remember", "forget", "find", "lose", "follow",
    "chase", "catch", "hold", "keep", "give", "take", "bring", "send",
    "call", "sing", "dance", "write", "read", "dream", "think",
    "imagine", "believe", "trust", "fear", "hope", "kiss", "save",
]

VERBS_GERUND: list[str] = [
    "loving", "missing", "chasing", "watching", "singing", "dancing",
    "dreaming", "falling", "flying", "running", "walking", "sleeping",
    "waking", "laughing", "crying", "whispering", "listening", "drifting",
]

TEMPLATES: list[str] = [
    # love / preference
    "i love {n}",
    "i adore {n}",
    "i miss {n}",
    "i like {n}",
    "i hate {n}",
    "i need {n}",
    "i want {n}",
    "we love {n}",
    "she loves {n}",
    "he loves {n}",
    # description
    "{n} are {adj}",
    "{n} is {adj}",
    "the {adj} {n}",
    "a {adj} {n}",
    "so {adj}",
    "very {adj}",
    # action
    "{v} the {n}",
    "{v} my {n}",
    "i {v} the {n}",
    "i am {g} the {n}",
    "{n} {g}",
    # combinations
    "{n} and {n2}",
    "{adj} {n} and {adj2} {n2}",
    "between {n} and {n2}",
    "from {n} to {n2}",
    # negation / contrast / equivalence
    "{n} are not {n2}",
    "{n} are not {n2} but {n3}",
    "{n} are {n2}",
    "{n} but {n2}",
    "not {n} but {n2}",
    "{n} not {n2}",
    "{n}, {n2}, and {n3}",
    "{n} and {n2} are {adj}",
    "both {n} and {n2}",
    "neither {n} nor {n2}",
    # poetic
    "{n} in the sky",
    "{n} in the forest",
    "{n} by the sea",
    "under the {n}",
    "beyond the {n}",
    "the sound of {n}",
    "the colour of {n}",
    "the song of {n}",
    # exclamations / fragments
    "oh {n}",
    "beautiful {n}",
    "lost {n}",
    "{n} forever",
    "forever {n}",
]

# A small hand-curated list of natural phrases added on top of templates
# to seed the bank with common idiomatic expressions.
NATURAL: list[str] = [
    "good morning", "good night", "sweet dreams", "thank you",
    "i miss you", "i love you", "happy birthday", "see you soon",
    "take care", "be careful", "don't worry", "time flies",
    "life is short", "love is blind", "birds of a feather",
    "the sky is blue", "the grass is green", "the sea is deep",
    "the night is dark", "the morning light", "the evening sun",
    "birds are beautiful", "children are playing", "dogs are barking",
    "cats are sleeping", "people are talking", "the world is round",
    "music is playing", "flowers are blooming", "leaves are falling",
    "snow is falling", "rain is falling", "rivers are flowing",
    "swans on the lake", "geese in the sky", "fish in the sea",
    "stars in the night", "sun in the morning", "moon over the mountains",
    "a quiet evening", "a long journey", "a short walk",
    "an old friend", "a new beginning", "a last goodbye",
    "i think of you", "i dream of you", "i wait for you",
    "come back home", "stay with me", "walk with me",
    "let it go", "hold me close", "never again", "once more",
    # negation / contrast / category
    "geese are not cats but birds", "geese are birds", "cats are not birds",
    "dogs are not cats", "birds are not fish", "whales are not fish",
    "the sun is not the moon", "day is not night", "love is not hate",
    "both birds and fish", "neither day nor night", "neither sun nor moon",
    "not cats but dogs", "not birds but bees", "not love but fear",
    "birds, geese, and swans", "cats, dogs, and birds",
    "sun, moon, and stars", "mother, father, and child",
]


_POOLS: dict[str, list[str]] = {
    "n": NOUNS, "n2": NOUNS, "n3": NOUNS,
    "adj": ADJECTIVES, "adj2": ADJECTIVES,
    "v": VERBS_PRESENT, "g": VERBS_GERUND,
}
_SLOT_RE = re.compile(r"\{(\w+)\}")


def _is_trivial(filled: str) -> bool:
    """Drop templates that filled to "X and X" / "X, X, and Y" etc."""
    for sep in (" and ", " but ", " nor ", " not "):
        if sep in filled:
            parts = [p.strip().rstrip(",") for p in filled.split(sep)]
            if any(a == b for a, b in zip(parts, parts[1:])):
                return True
    if ", " in filled and " and " in filled:
        tokens = [t.strip() for t in filled.replace(", and ", ", ").split(", ")]
        if len(tokens) != len(set(tokens)):
            return True
    return False


def generate(seed: int = 0, max_per_template: int = 140) -> list[str]:
    """Produce a deterministic, deduplicated phrase bank."""
    rng = random.Random(seed)
    out: set[str] = set(NATURAL)
    for tpl in TEMPLATES:
        slots = set(_SLOT_RE.findall(tpl))
        for _ in range(max_per_template):
            filled = tpl
            for s in slots:
                filled = filled.replace("{" + s + "}", rng.choice(_POOLS[s]))
            if "{" not in filled and not _is_trivial(filled):
                out.add(filled)
    return sorted(out)
