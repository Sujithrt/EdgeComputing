# Edge Computing using AWS IoT Services

This project repository offers two ways to do face detection at scale:

1. **Serverless (Cloud)**  
   • Containerized AWS Lambda (from ECR) behind a Function URL or API Gateway  
   • Incoming Base64 images trigger MTCNN inference in Lambda  
   • Results enqueued in Amazon SQS for downstream processing  

2. **On‑Device (Greengrass V2)**  
   • Lightweight Python component on your Greengrass Core  
   • Local MQTT subscription + Greengrass IPC for secure pub/sub  
   • MTCNN runs on CPU; outputs sent to SQS without ever leaving the edge  

## Key Technologies

- **AWS IoT Greengrass V2** (edge deployment & IPC)  
- **AWS Lambda & ECR** (cloud‑native containers)  
- **Python 3** with **facenet‑pytorch (MTCNN)** and **Pillow**  
- **Greengrass IPC & MQTT** for local messaging  
- **boto3** + **Amazon SQS** for reliable queuing  

## Processing Flow (Edge Mode)

1. Subscribe to a configured MQTT topic (e.g. `clients/<ThingName>`).  
2. Receive JSON payloads with:
   - `encoded`: Base64‑JPEG/PNG  
   - `request_id`: unique ID  
   - `filename`: original name  
3. Decode to a PIL image and run MTCNN.  
4. If a face is found, normalize, re‑encode as Base64‑JPEG, and send  
   ```json
   { "request_id": "...", "filename": "...", "face": "<Base64‑JPEG>" }
   ```
   to the “request” SQS queue.
5. If no face is detected, send
   ```
   { "request_id": "...", "filename": "...", "result": "No‑Face" }
   ```
   to the “response” SQS queue.

## System Architecture
### Edge Computing
![image](https://github.com/user-attachments/assets/89e304ee-1629-436e-b629-c3a2c416f070)

### Serverless
![image](https://github.com/user-attachments/assets/231a01a6-efc7-4687-b3bb-f7e0f5a60aad)

