import os
import json
import boto3
import base64
from io import BytesIO
from facenet_pytorch import MTCNN
from PIL import Image
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class FaceDetection:
    def __init__(self, mtcnn):
        self.mtcnn = mtcnn

    def detect_face(self, image):
        face, prob = self.mtcnn(image, return_prob=True, save_path=None)
        if face is not None:
            logger.debug(f"Face detected with probability: {prob:.2f}")
            face_img = face - face.min()
            face_img = face_img / face_img.max()
            face_img = (face_img * 255).byte().permute(1, 2, 0).numpy()
            return Image.fromarray(face_img, mode="RGB")
        else:
            logger.debug("No face detected")
            return None

global_mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20)
detector = FaceDetection(global_mtcnn)

def lambda_handler(event, context):
    try:
        body = event.get("body")
        if not body:
            logger.debug("No event body")
            return {"statusCode": 400, "body": json.dumps({"error": "Missing request body"})}
        if isinstance(body, str):
            body = json.loads(body)
            logger.debug(f"Received file: {body['filename']}")

        content = body.get("content")
        request_id = body.get("request_id")
        filename = body.get("filename")

        if not content or not request_id or not filename:
            logger.error("Missing required fields in the request body")
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Missing one or more required fields: content, request_id, filename"})
            }

        image_data = base64.b64decode(content)
        image = Image.open(BytesIO(image_data)).convert("RGB")

        face_img = detector.detect_face(image)

        if face_img is None:
            return {
                "statusCode": 200,
                "body": json.dumps({"message": "No face detected", "request_id": request_id})
            }

        face_path = f"/tmp/{request_id}_face.jpg"
        face_img.save(face_path)
        logger.debug(f"Face image saved at: {face_path}")

        with open(face_path, "rb") as f:
            face_bytes = f.read()
        face_base64 = base64.b64encode(face_bytes).decode("utf-8")

        message = {
            "request_id": request_id,
            "filename": filename,
            "face_image": face_base64
        }
        logger.debug(f"Sending message to SQS: {message['filename']}")

        sqs_queue_url = os.environ.get("SQS_REQUEST_QUEUE_URL")
        if not sqs_queue_url:
            logger.error("SQS_REQUEST_QUEUE_URL environment variable not set")
            return {
                "statusCode": 500,
                "body": json.dumps({"error": "SQS_REQUEST_QUEUE_URL environment variable not set"})
            }

        sqs_client = boto3.client("sqs")
        response = sqs_client.send_message(
            QueueUrl=sqs_queue_url,
            MessageBody=json.dumps(message)
        )
        logger.debug(f"SQS message sent: {response}")
        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": "Face detected and message sent to SQS",
                "request_id": request_id,
                "sqs_message_id": response.get("MessageId")
            })
        }
    except Exception as e:
        logger.error(f"Error in lambda_handler: {str(e)}", exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }