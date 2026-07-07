MODEL_LABELS = [
    "aom",
    "effusion",
    "normal",
    "perforation",
    "retraction",
    "tube",
    "tympanosclerosis",
]

EXTERNAL_EVAL_LABELS = [
    "effusion",
    "normal",
    "perforation",
    "retraction",
    "tube",
    "tympanosclerosis",
]

EXCLUDE_LABELS = {"cerumen", "cholesteatoma", "myringitis"}

LABEL_MAP = {
    "perforation": "perforation",
    "perf": "perforation",
    "hole": "perforation",
    "ruptured": "perforation",
    "tympanosclerosis": "tympanosclerosis",
    "sclerosis": "tympanosclerosis",
    "myringosclerosis": "tympanosclerosis",
    "retraction": "retraction",
    "retracted": "retraction",
    "atelectasis": "retraction",
    "aom": "aom",
    "acute otitis media": "aom",
    "erythema": "aom",
    "effusion": "effusion",
    "fluid": "effusion",
    "clear effusion": "effusion",
    "mucoid": "effusion",
    "tube": "tube",
    "pe tube": "tube",
    "tympanostomy": "tube",
    "cerumen": "cerumen",
    "wax": "cerumen",
    "obstruction": "cerumen",
    "normal": "normal",
    "clear": "normal",
    "cholesteatoma": "cholesteatoma",
    "myringitis": "myringitis",
    "bullous": "myringitis",
}

