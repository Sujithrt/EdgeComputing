# Edge Computing using AWS IoT Services

This project brings real‑time face detection to your AWS IoT Greengrass V2 core. It subscribes to a local MQTT topic, decodes incoming Base64 images, runs MTCNN on the device, and dispatches the results into Amazon SQS for downstream processing.

## Key Technologies

- **AWS IoT Greengrass V2**: Edge deployment and secure IPC.  
- **Python 3**: Component implementation.  
- **facenet‑pytorch (MTCNN)**: Lightweight on‑edge face detector.  
- **Pillow**: Image decoding and formatting.  
- **Boto3**: Integrates with Amazon SQS.  
- **Greengrass IPC**: Local publish/subscribe without cloud credentials.  
- **Amazon SQS**: Decoupled, reliable message queue.

## Processing Flow

1. **Listen** on a configured MQTT topic (e.g. `clients/<ThingName>`).  
2. **Receive** JSON payloads containing:  
   - `encoded`: Base64‑encoded image  
   - `request_id`: unique identifier  
   - `filename`: original image name  
3. **Decode** and convert to a PIL image.  
4. **Run MTCNN** to find a face crop.  
5. **Publish** outcome to SQS:  
   - If a face is found, normalize and re‑encode the crop, then send `{ request_id, filename, face: <Base64‑JPEG> }` to the “request” queue.  
   - If no face is detected, send `{ request_id, filename, result: "No‑Face" }` to the “response” queue.

## System Architecture
![image](https://github.com/user-attachments/assets/cf20d484-cce4-4b72-9d94-56fb583addf1)
