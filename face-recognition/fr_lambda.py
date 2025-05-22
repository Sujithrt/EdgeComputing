import os
import json
import boto3
import torch
import base64
import numpy as np
from PIL import Image
from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class FaceRecognition:
    def __init__(self, resnet):
        self.resnet = resnet

    def recognize_face(self, model_wt_path, face_img_path):
        face_pil = Image.open(face_img_path).convert("RGB")
        face_numpy = np.array(face_pil, dtype=np.float32)
        face_numpy /= 255.0
        face_numpy = np.transpose(face_numpy, (2, 0, 1))
        face_tensor = torch.tensor(face_numpy, dtype=torch.float32)

        saved_data = torch.load(model_wt_path)

        emb = self.resnet(face_tensor.unsqueeze(0)).detach()
        embedding_list = saved_data[0]
        name_list = saved_data[1]
        dist_list = []

        for idx, emb_db in enumerate(embedding_list):
            dist = torch.dist(emb, emb_db).item()
            dist_list.append(dist)

        idx_min = dist_list.index(min(dist_list))
        logger.debug(f"Minimum distance index: {idx_min}, Distance: {dist_list[idx_min]:.4f}")
        return name_list[idx_min]

global_resnet = InceptionResnetV1(pretrained='vggface2').eval()
recognizer = FaceRecognition(global_resnet)
sqs_client = boto3.client("sqs")

def lambda_handler(event, context):
    try:
        model_wt_path = os.environ.get("MODEL_WT_PATH")
        response_queue_url = os.environ.get("SQS_RESPONSE_QUEUE_URL")
        if not model_wt_path or not response_queue_url:
            logger.error("Environment variables MODEL_WT_PATH or SQS_RESPONSE_QUEUE_URL not set")
            return {
                "statusCode": 500,
                "body": json.dumps({"error": "Environment variables MODEL_WT_PATH or SQS_RESPONSE_QUEUE_URL not set"})
            }

        for record in event.get("Records", []):
            body = record.get("body", {})
            data = json.loads(body)
            request_id = data.get("request_id")
            logger.debug(f"Processing record: {request_id}")
            face_base64 = data.get("face_image")
            if not request_id or not face_base64:
                logger.debug("Missing required fields in the record body")
                continue

            face_bytes = base64.b64decode(face_base64)
            face_img_path = f"/tmp/{request_id}_face.jpg"
            with open(face_img_path, "wb") as f:
                f.write(face_bytes)
            logger.debug(f"Face image saved at: {face_img_path}")

            result = recognizer.recognize_face(model_wt_path, face_img_path)

            response_message = {
                "request_id": request_id,
                "result": result
            }
            logger.debug(f"Recognition result: {response_message}")

            sqs_client.send_message(
                QueueUrl=response_queue_url,
                MessageBody=json.dumps(response_message)
            )
            logger.debug(f"Sent response message to SQS: {response_message}")

        return {
            "statusCode": 200,
            "body": json.dumps({"message": "Face recognition completed successfully"})
        }

    except Exception as e:
        logger.error(f"Error in lambda_handler: {str(e)}", exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
