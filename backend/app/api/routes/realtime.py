from __future__ import annotations

import base64
import json
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from crop_ai.cv.decode import ImageDecodeError, decode_image
from crop_ai.cv.preprocessing import preprocess_image
from crop_ai.models.mock_classifier import MockClassifier

router = APIRouter(tags=["realtime"])


@router.websocket("/realtime/ws")
async def realtime(websocket: WebSocket) -> None:
    await websocket.accept()
    classifier = MockClassifier()
    try:
        while True:
            message = await websocket.receive()
            started = time.perf_counter()
            if message.get("bytes") is not None:
                data = message["bytes"]
            else:
                text = message.get("text", "")
                try:
                    payload = json.loads(text)
                    encoded = payload.get("image", "").split(",")[-1]
                    data = base64.b64decode(encoded, validate=True)
                except (ValueError, TypeError, json.JSONDecodeError):
                    await websocket.send_json(
                        {"error": "Send binary image bytes or JSON with base64 image."}
                    )
                    continue
            try:
                image = decode_image(data)
                stage = time.perf_counter()
                processed = preprocess_image(image, "standard")
                preprocessing_ms = (time.perf_counter() - stage) * 1000
                stage = time.perf_counter()
                prediction = classifier.predict(processed.image)
                inference_ms = (time.perf_counter() - stage) * 1000
                await websocket.send_json(
                    {
                        "is_mock": True,
                        "crop": prediction.crop,
                        "disease": prediction.disease,
                        "confidence": prediction.confidence,
                        "preprocessing_ms": round(preprocessing_ms, 2),
                        "inference_ms": round(inference_ms, 2),
                        "total_ms": round((time.perf_counter() - started) * 1000, 2),
                        "warning": "DEMO MODE: trained model not loaded.",
                    }
                )
            except ImageDecodeError as exc:
                await websocket.send_json({"error": str(exc)})
    except WebSocketDisconnect:
        return
