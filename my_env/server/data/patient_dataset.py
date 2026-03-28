from __future__ import annotations

from collections import Counter
from typing import Final, Literal, TypedDict


class VitalsRecord(TypedDict):
    heart_rate: int
    blood_pressure: str
    oxygen_level: float


class PatientRecord(TypedDict):
    symptoms: list[str]
    vitals: VitalsRecord
    age: int
    true_esi: Literal[1, 2, 3, 4, 5]


PATIENT_DATASET: Final[list[PatientRecord]] = [
    {
        "symptoms": ["stridor", "cyanosis", "severe dyspnea"],
        "vitals": {"heart_rate": 142, "blood_pressure": "80/50", "oxygen_level": 78.0},
        "age": 54,
        "true_esi": 1,
    },
    {
        "symptoms": ["respiratory distress", "confusion", "accessory muscle use"],
        "vitals": {"heart_rate": 138, "blood_pressure": "86/54", "oxygen_level": 82.0},
        "age": 67,
        "true_esi": 1,
    },
    {
        "symptoms": ["hematemesis", "syncope", "pallor"],
        "vitals": {"heart_rate": 132, "blood_pressure": "78/46", "oxygen_level": 88.0},
        "age": 63,
        "true_esi": 1,
    },
    {
        "symptoms": ["fever", "altered mental status", "chills", "tachycardia"],
        "vitals": {"heart_rate": 128, "blood_pressure": "82/48", "oxygen_level": 90.0},
        "age": 71,
        "true_esi": 1,
    },
    {
        "symptoms": ["crushing chest pain", "diaphoresis", "nausea"],
        "vitals": {"heart_rate": 122, "blood_pressure": "84/56", "oxygen_level": 89.0},
        "age": 59,
        "true_esi": 1,
    },
    {
        "symptoms": ["polytrauma", "abdominal tenderness", "hypotension"],
        "vitals": {"heart_rate": 136, "blood_pressure": "74/42", "oxygen_level": 85.0},
        "age": 34,
        "true_esi": 1,
    },
    {
        "symptoms": ["recurrent seizures", "unresponsiveness", "gurgling respirations"],
        "vitals": {"heart_rate": 130, "blood_pressure": "90/60", "oxygen_level": 86.0},
        "age": 46,
        "true_esi": 1,
    },
    {
        "symptoms": ["facial swelling", "stridor", "diffuse urticaria"],
        "vitals": {"heart_rate": 145, "blood_pressure": "76/40", "oxygen_level": 84.0},
        "age": 29,
        "true_esi": 1,
    },
    {
        "symptoms": ["sudden coma", "unequal pupils", "vomiting"],
        "vitals": {"heart_rate": 58, "blood_pressure": "210/110", "oxygen_level": 87.0},
        "age": 62,
        "true_esi": 1,
    },
    {
        "symptoms": ["apneic episodes", "intercostal retractions", "lethargy"],
        "vitals": {"heart_rate": 170, "blood_pressure": "70/40", "oxygen_level": 79.0},
        "age": 2,
        "true_esi": 1,
    },
    {
        "symptoms": ["chest pressure", "left arm pain", "shortness of breath"],
        "vitals": {"heart_rate": 110, "blood_pressure": "150/95", "oxygen_level": 95.0},
        "age": 57,
        "true_esi": 2,
    },
    {
        "symptoms": ["slurred speech", "right-sided weakness", "facial droop"],
        "vitals": {"heart_rate": 98, "blood_pressure": "182/102", "oxygen_level": 96.0},
        "age": 74,
        "true_esi": 2,
    },
    {
        "symptoms": ["wheezing", "chest tightness", "tachypnea"],
        "vitals": {"heart_rate": 124, "blood_pressure": "148/88", "oxygen_level": 91.0},
        "age": 39,
        "true_esi": 2,
    },
    {
        "symptoms": ["fever", "altered mental status", "neutropenia", "tachycardia"],
        "vitals": {"heart_rate": 118, "blood_pressure": "102/64", "oxygen_level": 97.0},
        "age": 50,
        "true_esi": 2,
    },
    {
        "symptoms": ["suicidal ideation", "anxiety", "insomnia"],
        "vitals": {"heart_rate": 88, "blood_pressure": "126/78", "oxygen_level": 98.0},
        "age": 23,
        "true_esi": 2,
    },
    {
        "symptoms": ["lower abdominal pain", "vaginal bleeding", "dizziness"],
        "vitals": {"heart_rate": 116, "blood_pressure": "98/60", "oxygen_level": 94.0},
        "age": 31,
        "true_esi": 2,
    },
    {
        "symptoms": ["melena", "lightheadedness", "fatigue"],
        "vitals": {"heart_rate": 112, "blood_pressure": "104/66", "oxygen_level": 93.0},
        "age": 66,
        "true_esi": 2,
    },
    {
        "symptoms": ["high fever", "poor feeding", "lethargy"],
        "vitals": {"heart_rate": 160, "blood_pressure": "92/58", "oxygen_level": 95.0},
        "age": 1,
        "true_esi": 2,
    },
    {
        "symptoms": ["severe headache", "blurred vision", "nausea"],
        "vitals": {"heart_rate": 102, "blood_pressure": "224/128", "oxygen_level": 96.0},
        "age": 61,
        "true_esi": 2,
    },
    {
        "symptoms": ["somnolence", "pinpoint pupils", "recent opioid use"],
        "vitals": {"heart_rate": 94, "blood_pressure": "108/70", "oxygen_level": 92.0},
        "age": 42,
        "true_esi": 2,
    },
    {
        "symptoms": ["right lower quadrant pain", "fever", "anorexia"],
        "vitals": {"heart_rate": 104, "blood_pressure": "122/78", "oxygen_level": 97.0},
        "age": 19,
        "true_esi": 3,
    },
    {
        "symptoms": ["productive cough", "fever", "dyspnea"],
        "vitals": {"heart_rate": 108, "blood_pressure": "118/74", "oxygen_level": 92.0},
        "age": 68,
        "true_esi": 3,
    },
    {
        "symptoms": ["flank pain", "dysuria", "fever"],
        "vitals": {"heart_rate": 106, "blood_pressure": "116/72", "oxygen_level": 96.0},
        "age": 47,
        "true_esi": 3,
    },
    {
        "symptoms": ["leg deformity", "severe pain", "inability to bear weight"],
        "vitals": {"heart_rate": 102, "blood_pressure": "130/82", "oxygen_level": 98.0},
        "age": 28,
        "true_esi": 3,
    },
    {
        "symptoms": ["colicky flank pain", "hematuria", "nausea"],
        "vitals": {"heart_rate": 100, "blood_pressure": "128/80", "oxygen_level": 99.0},
        "age": 44,
        "true_esi": 3,
    },
    {
        "symptoms": ["vomiting", "diarrhea", "dry mucous membranes"],
        "vitals": {"heart_rate": 110, "blood_pressure": "110/70", "oxygen_level": 97.0},
        "age": 36,
        "true_esi": 3,
    },
    {
        "symptoms": ["polyuria", "polydipsia", "fatigue"],
        "vitals": {"heart_rate": 96, "blood_pressure": "134/84", "oxygen_level": 98.0},
        "age": 52,
        "true_esi": 3,
    },
    {
        "symptoms": ["severe headache", "photophobia", "vomiting"],
        "vitals": {"heart_rate": 92, "blood_pressure": "124/76", "oxygen_level": 99.0},
        "age": 33,
        "true_esi": 3,
    },
    {
        "symptoms": ["right upper quadrant pain", "fever", "nausea"],
        "vitals": {"heart_rate": 98, "blood_pressure": "126/78", "oxygen_level": 97.0},
        "age": 58,
        "true_esi": 3,
    },
    {
        "symptoms": ["leg redness", "swelling", "low-grade fever"],
        "vitals": {"heart_rate": 95, "blood_pressure": "120/75", "oxygen_level": 98.0},
        "age": 49,
        "true_esi": 3,
    },
    {
        "symptoms": ["ankle pain", "swelling", "twisting injury"],
        "vitals": {"heart_rate": 84, "blood_pressure": "122/76", "oxygen_level": 99.0},
        "age": 26,
        "true_esi": 4,
    },
    {
        "symptoms": ["forearm laceration", "localized bleeding", "pain"],
        "vitals": {"heart_rate": 86, "blood_pressure": "118/74", "oxygen_level": 99.0},
        "age": 41,
        "true_esi": 4,
    },
    {
        "symptoms": ["dysuria", "urinary frequency", "suprapubic discomfort"],
        "vitals": {"heart_rate": 88, "blood_pressure": "116/72", "oxygen_level": 99.0},
        "age": 30,
        "true_esi": 4,
    },
    {
        "symptoms": ["ear pain", "mild fever", "decreased hearing"],
        "vitals": {"heart_rate": 92, "blood_pressure": "114/70", "oxygen_level": 99.0},
        "age": 12,
        "true_esi": 4,
    },
    {
        "symptoms": ["mild wheeze", "cough", "medication nonadherence"],
        "vitals": {"heart_rate": 96, "blood_pressure": "120/78", "oxygen_level": 97.0},
        "age": 21,
        "true_esi": 4,
    },
    {
        "symptoms": ["wrist pain", "swelling", "fall on outstretched hand"],
        "vitals": {"heart_rate": 82, "blood_pressure": "124/80", "oxygen_level": 99.0},
        "age": 38,
        "true_esi": 4,
    },
    {
        "symptoms": ["eye pain", "tearing", "foreign body sensation"],
        "vitals": {"heart_rate": 90, "blood_pressure": "122/78", "oxygen_level": 99.0},
        "age": 27,
        "true_esi": 4,
    },
    {
        "symptoms": ["mild headache", "scalp bruise", "no loss of consciousness"],
        "vitals": {"heart_rate": 80, "blood_pressure": "128/82", "oxygen_level": 99.0},
        "age": 17,
        "true_esi": 4,
    },
    {
        "symptoms": ["tooth pain", "gum swelling", "halitosis"],
        "vitals": {"heart_rate": 94, "blood_pressure": "126/80", "oxygen_level": 98.0},
        "age": 45,
        "true_esi": 4,
    },
    {
        "symptoms": ["hand burn", "blistering", "pain"],
        "vitals": {"heart_rate": 88, "blood_pressure": "118/76", "oxygen_level": 99.0},
        "age": 35,
        "true_esi": 4,
    },
    {
        "symptoms": ["needs antihypertensive refill"],
        "vitals": {"heart_rate": 76, "blood_pressure": "138/84", "oxygen_level": 99.0},
        "age": 64,
        "true_esi": 5,
    },
    {
        "symptoms": ["scheduled suture removal"],
        "vitals": {"heart_rate": 72, "blood_pressure": "120/74", "oxygen_level": 99.0},
        "age": 32,
        "true_esi": 5,
    },
    {
        "symptoms": ["sneezing", "itchy eyes", "nasal congestion"],
        "vitals": {"heart_rate": 78, "blood_pressure": "118/72", "oxygen_level": 99.0},
        "age": 25,
        "true_esi": 5,
    },
    {
        "symptoms": ["chronic low back pain", "no weakness", "no bowel changes"],
        "vitals": {"heart_rate": 80, "blood_pressure": "126/78", "oxygen_level": 99.0},
        "age": 55,
        "true_esi": 5,
    },
    {
        "symptoms": ["asymptomatic blood pressure concern"],
        "vitals": {"heart_rate": 74, "blood_pressure": "142/88", "oxygen_level": 99.0},
        "age": 48,
        "true_esi": 5,
    },
    {
        "symptoms": ["requesting work clearance", "resolved upper respiratory illness"],
        "vitals": {"heart_rate": 77, "blood_pressure": "116/70", "oxygen_level": 99.0},
        "age": 29,
        "true_esi": 5,
    },
    {
        "symptoms": ["mild eczema flare", "itching", "dry skin"],
        "vitals": {"heart_rate": 79, "blood_pressure": "114/68", "oxygen_level": 99.0},
        "age": 14,
        "true_esi": 5,
    },
    {
        "symptoms": ["decreased hearing", "ear fullness", "no pain"],
        "vitals": {"heart_rate": 75, "blood_pressure": "122/76", "oxygen_level": 99.0},
        "age": 60,
        "true_esi": 5,
    },
    {
        "symptoms": ["mild medication-related nausea", "no vomiting"],
        "vitals": {"heart_rate": 82, "blood_pressure": "124/80", "oxygen_level": 99.0},
        "age": 43,
        "true_esi": 5,
    },
    {
        "symptoms": ["requests smoking cessation counseling"],
        "vitals": {"heart_rate": 73, "blood_pressure": "118/74", "oxygen_level": 99.0},
        "age": 37,
        "true_esi": 5,
    },
]


def get_patient_dataset() -> list[PatientRecord]:
    return [
        {
            "symptoms": list(record["symptoms"]),
            "vitals": {
                "heart_rate": record["vitals"]["heart_rate"],
                "blood_pressure": record["vitals"]["blood_pressure"],
                "oxygen_level": record["vitals"]["oxygen_level"],
            },
            "age": record["age"],
            "true_esi": record["true_esi"],
        }
        for record in PATIENT_DATASET
    ]


def _validate_dataset_balance(dataset: list[PatientRecord]) -> None:
    if len(dataset) < 50:
        raise ValueError("Dataset must contain at least 50 patients")

    counts = Counter(record["true_esi"] for record in dataset)
    expected = {1: 10, 2: 10, 3: 10, 4: 10, 5: 10}

    if counts != expected:
        raise ValueError(f"Dataset ESI distribution must be balanced: expected {expected}, got {dict(counts)}")


_validate_dataset_balance(PATIENT_DATASET)
