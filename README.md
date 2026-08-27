# NOAA NMFS Optics Iceseals SAHI DetectorModel Deployment (Ultralytics YOLO)

Welcome! This repository provides a template for deploying custom **Ultralytics YOLO** Computer Vision models into the NOAA NMFS Optics SI Airflow ecosystem on Google Cloud Platform (GCP).

Our cloud infrastructure runs models inside isolated Docker containers, exposes an HTTP endpoint, and interacts directly with Google Cloud Storage (GCS). This template automates downloading inputs, running Ultralytics inference, and formatting model outputs into standard CSV / KWCOCO formats.

---

## 🟢 Phase 1: Local Setup & Testing

Before deploying to Google Cloud, bake your custom model weights into the container image and test the execution locally.

### 🏁 Step 1: Add Weights

1. Clone this repositorty to your local machine or Google Cloud Workstation.

```bash
git clone https://github.com/eem1/optics-models-ultralytics-detection-SAHI.git
cd optics-models-ultralytics-detection_SAHI
```

2. **CRITICAL:** Place your default YOLO weights file in the root directory and name it exactly `model.pt`: 

```bash
cp /path/to/your/best.pt ./model.pt
```

### 💻 Step 2: Test Locally

We recommend using the Google Cloud workstations for testing, as `docker` and `gcloud` come pre-installed.

**1. Authenticate with Google Cloud**
Ensure you have local credentials so the container can download test files from GCS:
```bash
gcloud auth application-default login

chmod +r ~/.config/gcloud/application_default_credentials.json
```

**2. Build the Docker Image**
```bash
docker build -t optics-yolo-sahi-model:latest .
```

**3. Run the Container**

*(This maps your local GCP credentials into the container so it can access buckets)*

Linux / macOS / Cloud Workstation:

```bash
docker run -p 8080:8080 \
  -v ~/.config/gcloud:/tmp/.config/gcloud \
  -e GOOGLE_APPLICATION_CREDENTIALS=/tmp/.config/gcloud/application_default_credentials.json \
  -e GOOGLE_CLOUD_PROJECT=ggn-nmfs-osi-dev-1 \
  optics-yolo-sahi-model:latest
```

Windows (PowerShell):

```bash
docker run -p 8080:8080 `
  -v ${env:APPDATA}\gcloud:/tmp/.config/gcloud `
  -e GOOGLE_APPLICATION_CREDENTIALS=/tmp/.config/gcloud/application_default_credentials.json `
  -e GOOGLE_CLOUD_PROJECT=ggn-nmfs-osi-dev-1 `
  optics-yolo-sahi-model:latest
```

**4. Send a Test Request**

With your container running, open a new terminal and send a JSON payload to test it. Ensure you have updated the GCS paths in your test payload to point to actual media files you have access to.


```bash
curl -X POST http://localhost:8080/predict -H "Content-Type: application/json" -d @test_payloads/test_payload_sahi_images.json
```

---

#### ⚙️ Configuration & Features

#### Passing SAHI (Slicing Aided Hyper Inference)
```json
"config": {
    "slice_width": 6384,
    "slice_height": 4782, 
    "overlap_ratio": 0.2,

    "options": {}
    }
```

See `test_payloads\test_payload_sahi_images.json` for an example.

#### YOLO Inference Options (Kwargs Passthrough)
Any key-value pairs you place inside the `"options"` object of your config will be passed directly to the `YOLO.predict()` method as `**kwargs`. This means you can control confidence, IOU, image size, and more, directly from the Airflow UI without changing code.

```json
"config": {
    "options": {
        "conf": 0.25,
        "iou": 0.45,
        "imgsz": 1280
    }
}
```

#### Dynamic Weights Override
Although the container bakes in a default `model.pt`, you can supply custom weights at runtime via a GCS URI in the `"weights"` key:

```json
"config": {
    "weights": "gs://my-bucket/path/to/experimental_weights.pt",
    "options": { }
}
```
**⚠️ Performance Warning:** Using dynamic weights provides great flexibility for A/B testing, but it has performance tradeoffs. Downloading large `.pt` files from GCS at runtime will increase the latency of the job startup and consume more network bandwidth. For highly scaled production jobs, baking the weights into the container image is preferred.


#### Format & KWCOCO Support

**CSV Output**: Currently configured output target.

**KWCOCO Support**: Refer to the  [NOAA NMFS Optics Model Deployment Template](https://github.com/csbrown-noaa/optics-models-ultralytics-detection) documentation. Handles both single images and video sequences with bounding boxes reformatted from YOLO xyxy to COCO [x, y, width, height].

### 💻 Step 3: Deploy to Google Cloud
**1. Push Image to Artifact Registry**

```batch
docker tag optics-yolo-sahi-model:latest us-central1-docker.pkg.dev/ggn-nmfs-osi-dev-1/nmfs-dev-uc1-docker-repository/optics-yolo-sahi-model:latest 

docker push us-central1-docker.pkg.dev/ggn-nmfs-osi-dev-1/nmfs-dev-uc1-docker-repository/optics-yolo-sahi-model:latest

```

**2. Update Airflow Runtime Definitions**

To register or update container definitions in Google Cloud Composer / Airflow:

1. Download `/configs/model_runtime_definitions.json` from GCS.

2. Search for `optics-yolo-sahi-model` and update its configuration (image tag, container flags, runtime limits).

3. Re-upload the updated model_runtime_definitions.json back to GCS.

**3. Trigger Pipeline via Airflow UI**

1. Upload your target YAML configuration file (e.g., test_airflow_dag/nmfs-optics-yolo-sahi-config.yaml) to GCS.

    See yaml file for details on passing SAHI (Slicing Aided Hyper Inference) params to the model.

2. Upload your target dataset test json file (e.g., test_airflow_dag/yolo-sahi-input-images) to GCS.

3. Open Google Cloud Console, project ID `ggn-nmfs-osi-dev-1`, search Bar `Managed Airflow`> select `composer-env1`> `Open Airflow UI` > select `nmfs-optics-pipeline-longrunning-dag`> Trigger DAG arrow

4. Fill out the trigger form:

   Model Type: `optics-yolo-sahi-model`

   YAML Config File: `gs://bucket/your_folder/configs/nmfs-optics-yolo-sahi-config.yaml`

   Output Folder: `your_folder/your-output-folder/`

5. Click Trigger and monitor execution logs until completion. 

6. Monitor DAG Progress 

    Select `Managed Airflow` >  `composer-env1` > `DAGs`
    Click `nmfs-optics-pipeline-longrunning-dag` to see the list DAG runs.

7. Check GCP Batch Job Status

    Open Google Cloud Console and search for Batch in the top search bar.

    When the job is scheduled, it will appear in the Job List.
    
    Select your job and click the Logs tab to view live execution details.
