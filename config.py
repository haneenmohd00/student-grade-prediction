PROJECT_TITLE = "student performance prediction using linear regression with ai tutor(llm)"

TEXT_COLUMNS = [
    "StudentID",
    "Age",
    "Gender",
    "Ethnicity",
    "ParentalEducation",
    "StudyTimeWeekly",
    "Absences",
    "Tutoring",
    "ParentalSupport",
    "Extracurricular",
    "Sports",
    "Music",
    "Volunteering",
    "GPA",
]

TARGET_COLUMN = "GradeClass"

DATA_PATH = "data/dataset.csv"

MODEL_DIR = "models"
MODEL_PATH = f"{MODEL_DIR}/model.pkl"
PREPROCESSOR_PATH = f"{MODEL_DIR}/preprocessor.pkl"
METRICS_PATH = f"{MODEL_DIR}/metrics.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2

LOG_LEVEL = "INFO"
