#!/usr/bin/env python3
import base64
import json
import logging
import os
import sys
import time
from io import BytesIO

import boto3
from awsiot.greengrasscoreipc.clientv2 import GreengrassCoreIPCClientV2
from awsiot.greengrasscoreipc.model import SubscribeToTopicRequest
from facenet_pytorch import MTCNN
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ipc_client = GreengrassCoreIPCClientV2()
sqs_client = boto3.client("sqs")
processed_requests = set()

mtcnn_detector = MTCNN(image_size=240, margin=0, min_face_size=20)

def handle_incoming_message(message_str: str):
    payload = json.loads(message_str)
    request_id = payload.get("request_id", "<unknown>")
    logger.info(f"Start processing request_id={request_id}")
    if request_id in processed_requests:
        return
    processed_requests.add(request_id)

    raw_data = base64.b64decode(payload["encoded"])

    image = Image.open(BytesIO(raw_data)).convert("RGB")
    logger.info(f"Decoded image bytes for request_id={request_id}")
    face_tensor, probability = mtcnn_detector(image, return_prob=True)

    if face_tensor is None:
        logger.info(f"No face detected for request_id={request_id}, sending No-Face result")
        response = {"request_id": request_id, "result": "No-Face"}
        sqs_client.send_message(QueueUrl=response_queue_url, MessageBody=json.dumps(response))
        return

    normalized = (face_tensor - face_tensor.min()) / (face_tensor.max() - face_tensor.min())
    array_ = (normalized * 255).byte().permute(1, 2, 0).numpy()
    face_image = Image.fromarray(array_, mode="RGB")

    buf = BytesIO()
    face_image.save(buf, format="JPEG")
    encoded_face = base64.b64encode(buf.getvalue()).decode()
    logger.info(f"Encoded cropped face to base64 for request_id={request_id}")
    response = {
        "request_id": request_id,
        "face_image": encoded_face,
        "filename": payload.get("filename", "")
    }
    sqs_client.send_message(QueueUrl=request_queue_url, MessageBody=json.dumps(response))
    logger.info(f"Response message sent for request_id={request_id}")

def _on_stream_event(event):
    logger.debug(f"Received event from Greengrass on topic {mqtt_topic}: {event}")
    try:
        handle_incoming_message(event.binary_message.message.decode("utf-8"))
    except Exception as err:
        logger.error(f"Error processing stream event for topic {mqtt_topic}: {err}")

def _on_error(error):
    logger.error(f"Subscription error on topic {mqtt_topic}: {error}")

def _on_closed():
    logger.info(f"Greengrass subscription for topic {mqtt_topic} has been closed")

def main():
    _, _ = ipc_client.subscribe_to_topic(
        topic=mqtt_topic,
        on_stream_event=_on_stream_event,
        on_stream_error=_on_error,
        on_stream_closed=_on_closed
    )
    logger.info(f"Subscribed to Greengrass topic {mqtt_topic}, awaiting incoming images")

    try:
        while True:
            time.sleep(10)
    except InterruptedError:
        logger.info("Subscription loop interrupted, shutting down")

if __name__ == "__main__":
    mqtt_topic, request_queue_url, response_queue_url, aws_region = sys.argv[1:5]
    main()
