
import os
import logging
import joblib
import httpx

from pathlib import Path
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FactoryGuardBackend")

app = FastAPI(title="FactoryGuard AI")

# Allow requests from your deployed frontend and local development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://factory-guard-mu.vercel.app",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

AMD_API_KEY = os.getenv("AMD_API_KEY")
LLM_API_ENDPOINT = os.getenv(
    "LLM_API_ENDPOINT", "http://localhost:8000/v1"
)

# Resolve the model path relative to this file, not the current working directory.
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(
    os.getenv("MODEL_PATH", "models/sensor_anomaly_model.pkl")
)
if not MODEL_PATH.is_absolute():
    MODEL_PATH = BASE_DIR / MODEL_PATH

ml_model = None

try:
    if MODEL_PATH.is_file():
        ml_model = joblib.load(MODEL_PATH)
        logger.info("ML model loaded successfully from %s", MODEL_PATH)
    else:
        logger.error("ML model file not found: %s", MODEL_PATH)
except Exception:
    logger.exception("Failed to load ML model from %s", MODEL_PATH)


class SensorDataRequest(BaseModel):
    temperature: float = Field(..., description="Sensor temperature input")
    vibration: float = Field(..., description="Sensor vibration input")
    pressure: float = Field(..., description="Sensor pressure input")


class FullWorkflowRequest(BaseModel):
    object_type: str = Field(..., min_length=1)
    authorized: bool
    sensor_data: SensorDataRequest


@app.get("/")
def home():
    return {"message": "FactoryGuard AI backend is running!"}


@app.get("/health")
def health():
    return {
        "status": "online",
        "model_loaded": ml_model is not None,
        "model_path": str(MODEL_PATH),
        "amd_config": bool(AMD_API_KEY),
    }


@app.post("/predict-anomaly")
def predict_anomaly(sensor_data: SensorDataRequest):
    if ml_model is None:
        logger.error("Prediction requested, but the ML model is not loaded.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML model is not loaded. Check the model file and Render logs.",
        )

    try:
        features = [[
            sensor_data.temperature,
            sensor_data.vibration,
            sensor_data.pressure,
        ]]

        prediction = ml_model.predict(features)[0]

        return {
            "status": "ANOMALY" if int(prediction) == 1 else "NORMAL",
            "prediction": int(prediction),
        }

    except Exception:
        # Full traceback appears in Render logs for debugging.
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing sensor prediction. Check server logs.",
        )


@app.post("/workflow")
async def run_full_workflow(request: FullWorkflowRequest):
    try:
        obj_type = request.object_type.strip().lower()

        if request.authorized:
            obj_status = "NORMAL"
        elif obj_type in ["wildlife", "unknown_person"]:
            obj_status = "ISOLATE"
        else:
            obj_status = "WARNING"

        sensor_status = "NORMAL"

        if ml_model is not None:
            features = [[
                request.sensor_data.temperature,
                request.sensor_data.vibration,
                request.sensor_data.pressure,
            ]]

            prediction = ml_model.predict(features)[0]

            if int(prediction) == 1:
                sensor_status = "ANOMALY_DETECTED"
        else:
            sensor_status = "MODEL_UNAVAILABLE"

        llm_guidance = "Standard monitoring recommended."

        if AMD_API_KEY:
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.post(
                        f"{LLM_API_ENDPOINT.rstrip('/')}/chat",
                        headers={
                            "Authorization": f"Bearer {AMD_API_KEY}"
                        },
                        json={
                            "prompt": (
                                f"Status: {obj_status}, "
                                f"Sensor: {sensor_status}. "
                                "Suggest response."
                            )
                        },
                    )

                    if response.status_code == 200:
                        llm_guidance = response.json().get(
                            "guidance", llm_guidance
                        )
                    else:
                        logger.warning(
                            "LLM service returned HTTP %s",
                            response.status_code,
                        )

            except Exception:
                logger.exception("LLM API request failed")
                llm_guidance = (
                    "AI guidance service unavailable. "
                    "Follow standard operating procedures."
                )

        return {
            "object_status": obj_status,
            "sensor_status": sensor_status,
            "guidance": llm_guidance,
            "system_overall": (
                "CRITICAL"
                if obj_status == "ISOLATE"
                or sensor_status == "ANOMALY_DETECTED"
                else "STABLE"
            ),
        }

    except Exception:
        logger.exception("Workflow execution failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during workflow execution.",
        )
