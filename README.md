# NOAA NMFS Optics Custom Model Deployment (Ultralytics YOLO)

Welcome! This repository is a starting template for deploying custom **Ultralytics YOLO** Computer Vision models into the Optics SI Airflow ecosystem on Google Cloud. 

Our infrastructure requires models to run inside isolated Docker containers, expose an HTTP endpoint, and communicate with Google Cloud Storage (GCS). We have pre-written the heavy lifting for you. This repository expects you to provide a `.pt` weights file, and it will automatically handle downloading inputs, running inference using Ultralytics, and formatting the output into CSV.

## 🟢 Phase 1: Local Setup & Testing

Before deploying to the cloud, you should bake your custom model into the container and test it locally.

### 🏁 Step 1: Clone and Add Weights

1. Clone it to your local machine (or Google Cloud Workstation).
2. **CRITICAL:** You must place your default YOLO weights file in the root of the repository and name it exactly `model.pt`. 

```bash
git clone https://github.com/eem1/optics-models-ultralytics-detection.git
cd optics-models-ultralytics-detection
# Copy your weights in...
cp /path/to/your/best.pt ./model.pt
```

### 💻 Step 2: Test Locally

We recommend using the Google Cloud workstations for testing, as they already have `docker` and `gcloud` installed.

**1. Authenticate with Google Cloud**
Ensure you have local credentials so the container can download test files from GCS:
```bash
gcloud auth application-default login
```

```bash
chmod +r ~/.config/gcloud/application_default_credentials.json
```

**2. Build the Docker Container**
```bash
docker build -t optics-yolo-sahi-model:latest .
```

**3. Run the Container**

*(This maps your local GCP credentials into the container so it can access buckets)*

```bash
docker run -p 8080:8080 \
  -v ~/.config/gcloud:/tmp/.config/gcloud \
  -e GOOGLE_APPLICATION_CREDENTIALS=/tmp/.config/gcloud/application_default_credentials.json \
  -e GOOGLE_CLOUD_PROJECT=ggn-nmfs-osi-dev-1 \
  optics-yolo-sahi-model:latest
```

For Windows (using PowerShell):

```bash
docker run -p 8080:8080 `
  -v ${env:APPDATA}\gcloud:/tmp/.config/gcloud `
  -e GOOGLE_APPLICATION_CREDENTIALS=/tmp/.config/gcloud/application_default_credentials.json `
  -e GOOGLE_CLOUD_PROJECT=ggn-nmfs-osi-dev-1 `
 optics-yolo-sahi-model:latest
```

**4. Send a Test Request**

With your container running, open a new terminal and send a JSON payload to test it. 

(Ensure you have updated the GCS paths in your `test_payload_sahi_images.json` to point to actual media files you have access to)

```bash
curl -X POST http://localhost:8080/predict \
     -H "Content-Type: application/json" \
     -d @test_payloads/test_payload_sahi_images.json
```

### ⚙️ Configuration & Features

This template supports several advanced features through the JSON Airflow payload.

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
While the container is built with a default `model.pt` baked in, you can instruct it to download a different set of weights at runtime by providing a GCS URI in the `"weights"` key.

```json
"config": {
    "weights": "gs://my-bucket/path/to/experimental_weights.pt",
    "options": { ... }
}
```
**⚠️ Performance Warning:** Using dynamic weights provides great flexibility for A/B testing, but it has performance tradeoffs. Downloading large `.pt` files from GCS at runtime will increase the latency of the job startup and consume more network bandwidth. For highly scaled production jobs, baking the weights into the container image is preferred.

### Model Output 
Currently SAHI model formats the output in CSV.
You can add codes to the `model.py` to support **KWCOCO (Kitware COCO)** JSON. 

See the template for formatting: https://github.com/csbrown-noaa/optics-models-ultralytics-detection

## ☁️ Phase 2: Deploy to Cloud

Once you are happy with local testing, you will push this container to the Google Artifact Registry.

***!!!!NB!!!!*** The Artifact Registry is where everyone's models lives.  By pushing your docker image to the registry, there is a risk that you may overwrite existing docker images.  Please be careful here.

**1. Authenticate with Google Cloud**

Since the docker image is doing work **on your behalf**, it needs its own set of credentials.  This login is so that you can interact directly with gcloud, which is what we're doing now.
```bash
gcloud auth login
```

Let's list the existing images in the registry first:

```bash
gcloud artifacts packages list \
  --project=ggn-nmfs-osi-dev-1 \
  --location=us-central1 \
  --repository=nmfs-dev-uc1-docker-repository
```

**2. Register your Docker image to the Google Cloud Artifact Registry**

```bash
# Tag your image for the registry
docker tag optics-yolo-sahi-model:latest us-central1-docker.pkg.dev/ggn-nmfs-osi-dev-1/nmfs-dev-uc1-docker-repository/optics-yolo-sahi-model:latest

# Push it
docker push us-central1-docker.pkg.dev/ggn-nmfs-osi-dev-1/nmfs-dev-uc1-docker-repository/optics-yolo-sahi-model:latest
```

**3. Hook it into Airflow to run the Pipeline**

To make your model available in the system, you must register it in the Airflow DAG.

Download GCS `ggn-nmfs-osi-dev-1-data/configs/model_runtime_definitions.json` 

The entries in the json are for the various models. It's just a dictionary containing a hard-coded list of all of the available models and the relevant configuration for that model. See `ultralytics` as an example.

Add your model to the json file and save it.

```json
    "optics-yolo-sahi-model": {
        "region": "us-central1",
        "image": "us-central1-docker.pkg.dev/ggn-nmfs-osi-dev-1/nmfs-dev-uc1-docker-repository/optics-yolo-sahi-model:latest",
        "cpu": 4,
        "memory": "16Gi",
        "gpu": 0,                        
        "gpu_type": null,                
        "machine_type": "c2-standard-4",
        "timeout": 360000,               
        "command": ["python"],
        "args": ["/workspace/inference_runner.py"]
    }
```

Upload the file back to the original GCS folder `ggn-nmfs-osi-dev-1-data/configs/model_runtime_definitions.json` 


**4. Prepare the Input Files to Trigger the Pipeline in Airflow DAG**

See /dag_files folder in this repo:
1. Upload your YAML Config file to GCS (e.g., `gs://ggn-nmfs-osi-dev-1-data/my-folder/nmfs-optics-yolo-sahi-config.yaml`).
2. Upload your input JSON file to GCS (e.g., `gs://ggn-nmfs-osi-dev-1-data/my-folder/yolo-sahi-input-images.json`).

**5: Triggering Pipeline**
1. Go to Google Cloud console, on search bar `Airflow`, select `Managed Airflow` ->  `composer-env1` -> `Open Airflow UI` tab
2. Locate the `nmfs-optics-pipeline-longrunning-dag`, click **Trigger**.
3. Select the `Model Type` parameter to `optics-yolo-sahi-model`.
4. Set the `YAML Config File Path` parameter to  `gs://ggn-nmfs-osi-dev-1-data/my-folder/nmfs-optics-yolo-sahi-config.yaml`
5. Set the `Input File` parameter to `gs://ggn-nmfs-osi-dev-1-data/my-folder/yolo-sahi-input-images.json`
6. Hit **Trigger** and monitor your job's progress in the logs!

**6. Monitor the DAG Progress and Job Status**
1. To monitor the DAG progress, select `Managed Airflow` ->  `composer-env1` -> `DAGs`. Click `nmfs-optics-pipeline-longrunning-dag` to see the list DAG runs.
2. To view the Job Status and logs, go to `Google Cloud console`, search `Batch`. When the job is actually scheduled to run, your job will appear on the Job list, select the job, click `Logs` tab for log details

**7. Check the Output Folder for Results**

