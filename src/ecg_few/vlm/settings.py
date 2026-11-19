"""Default experiment settings."""

ZERO_SHOT_CONDITION = "zero_shot"
ESTANDAR_CONDITION = "estandar"
BALANCED_CONDITION = "balanced"
PERMUTED_CONDITION = "permuted"
NO_SUPPORT_IMAGES_CONDITION = "no_support_images"
MORPHOLOGY_TASK = "morphology"
CLINICAL_TASK = "clinical"
CLINICAL_LABEL_NAME = "clinical_brugada"
ONLY_LEAD = "V1"
DEFAULT_MORPHOLOGY_SYSTEM_PROMPT = "package:system/qrs_huca.md"
DEFAULT_MORPHOLOGY_PROMPT = "package:qrs/right_precordial_morphology.md"
DEFAULT_CLINICAL_SYSTEM_PROMPT = "package:system/clinical_brugada_huca.md"
DEFAULT_CLINICAL_PROMPT = "package:clinical/brugada_patient.md"
DEFAULT_CLINICAL_LEADS = ("V1",)
CLINICAL_AGGREGATIONS = ("majority", "any_positive", "all_positive")
ALL_CONDITIONS = (
    ZERO_SHOT_CONDITION,
    ESTANDAR_CONDITION,
    BALANCED_CONDITION,
    PERMUTED_CONDITION,
    NO_SUPPORT_IMAGES_CONDITION,
)
