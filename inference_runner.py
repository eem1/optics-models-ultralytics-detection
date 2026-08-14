import os, sys, json, time, subprocess, yaml
import urllib.request
from urllib.error import URLError
from google.cloud import storage
import util

HOST = "http://localhost:8080"
HEALTH_ENDPOINT = f"{HOST}/isalive"
PREDICT_ENDPOINT = f"{HOST}/predict"

MAX_STARTUP_WAIT_SECONDS = 120
# 360000 seconds = 100 hours. Extreme timeout duration to accommodate 
# multi-hour, highly intensive computer vision pipelines running against long videos.
MAX_PREDICTION_WAIT_SECONDS = 360000 

def download_json_from_gcs(gcs_uri: str) -> dict:
    """
    Downloads a JSON payload file from GCS and returns it as a dictionary.
    
    Parameters
    ----------
    gcs_uri : str
        The full gs:// URI of the JSON file to download.
        
    Returns
    -------
    dict
        The parsed JSON dictionary.
        
    Raises
    ------
    ValueError
        If the URI does not start with gs://
    FileNotFoundError
        If the file cannot be located in the specified bucket.
    """
    print(f"[RUNNER] Fetching payload from {gcs_uri}...", flush=True)
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Input file URI must start with gs://. Got: {gcs_uri}")
    
    client = storage.Client()
    path_parts = gcs_uri.replace("gs://", "").split("/")
    bucket_name = path_parts[0]
    blob_path = "/".join(path_parts[1:])
    
    blob = client.bucket(bucket_name).blob(blob_path)
    if not blob.exists():
        raise FileNotFoundError(f"Input file not found in GCS: {gcs_uri}")
    
    return json.loads(blob.download_as_text())

def wait_for_server():
    """
    Polls the /isalive endpoint until the Flask server is ready.
    
    Raises
    ------
    TimeoutError
        If the server fails to report health within MAX_STARTUP_WAIT_SECONDS.
    """
    print("[RUNNER] Waiting for API server to start...", flush=True)
    start_time = time.time()
    while time.time() - start_time < MAX_STARTUP_WAIT_SECONDS:
        try:
            response = urllib.request.urlopen(HEALTH_ENDPOINT)
            if response.getcode() == 200:
                print(f"[RUNNER] Server is healthy! (Took {int(time.time() - start_time)}s)", flush=True)
                return
        except (URLError, ConnectionResetError):
            pass
        time.sleep(2)
    raise TimeoutError(f"Server did not become healthy within {MAX_STARTUP_WAIT_SECONDS} seconds.")

def execute_prediction(payload: dict):
    """
    Sends the JSON payload to the local /predict endpoint to trigger inference.
    
    Parameters
    ----------
    payload : dict
        The Vertex AI-style prediction payload containing input configurations.
    """
    print(f"[RUNNER] Sending payload to /predict endpoint. Timeout set to {MAX_PREDICTION_WAIT_SECONDS}s...", flush=True)
    req = urllib.request.Request(
        PREDICT_ENDPOINT,
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    
    try:
        response = urllib.request.urlopen(req, timeout=MAX_PREDICTION_WAIT_SECONDS)
        response_body = response.read().decode('utf-8')
        if response.getcode() == 200:
            print("[RUNNER] Prediction completed successfully.", flush=True)
        else:
            print(f"[RUNNER] Warning: Server returned status {response.getcode()}", flush=True)
    except Exception as e:
        print(f"[RUNNER] Failed to execute prediction: {str(e)}", flush=True)
        sys.exit(1)

def main():
    print("==================================================", flush=True)
    print("  Starting Cloud Batch Inference Wrapper          ", flush=True)
    print("==================================================", flush=True)

    config_yaml_file = os.environ.get("YAML_CONFIG_PATH")
    output_bucket = os.environ.get("OUTPUT_BUCKET")
    output_folder = os.environ.get("OUTPUT_FOLDER")
    input_file = os.environ.get("INPUT_FILE")

    print(f'[RUNNER] config_yaml_file:{config_yaml_file}')
    print(f'[RUNNER] input_file:{input_file}')
    print(f'[RUNNER] output_bucket:{output_bucket}')
    print(f'[RUNNER] output_folder:{output_folder}')
    
    if not input_file:
        print("[RUNNER] Error: INPUT_FILE environment variable is not set.", flush=True)
        sys.exit(1)
        
    print("[RUNNER] Spawning app.py as a background process...", flush=True)
    server_process = subprocess.Popen([sys.executable, "app.py"])
    
    try:
        wait_for_server()
        json_input = download_json_from_gcs(input_file)

        bucket_name = config_yaml_file.replace("gs://", "").split("/")[0]
        blob_path = "/".join(config_yaml_file.replace("gs://", "").split("/")[1:])
        client = storage.Client()
        blob = client.bucket(bucket_name).blob(blob_path)
    
        if not blob.exists():
            raise Exception(f"Config YAML not found at: {config_yaml_file}")

        config_yaml = yaml.safe_load(blob.download_as_text())

        # Use config_yaml to bridge the gap between the DAG and the payload before sending it to the model

        if(not json_input.get("instances", [{}])[0].get("output_file", None)):
           json_input["instances"][0]["output_file"] = f"gs://{output_bucket}/{output_folder.strip('/')}/results_sahi.csv"

        if(not json_input.get("instances", [{}])[0].get("config", None)):
            config = {} 
            params = config_yaml.get("params", {})
            if params:
                config["slice_width"] = params.get("slice_width")
                config["slice_height"] = params.get("slice_height")
                config["overlap_ratio"] = params.get("overlap_ratio")
                
            payload = config_yaml.get("payload", {})
            if payload:
                config["options"] = payload

            weights = config_yaml.get("weights",None)
            if weights:
                config["weights"] = weights

            json_input["instances"][0]["config"] = config
               

        execute_prediction(json_input)
        if config_yaml.get("upload_dataset_to_output_folder", False):
            print("[RUNNER] Starting copy input dataset to GCS output folder.", flush=True)
            input_uris = json_input.get("instances", [{}])[0].get("input_files", [])
            util.copy_dataset_to_gcs_output_folder(input_uris, output_bucket, output_folder)

    finally:
        print("[RUNNER] Shutting down background server...", flush=True)
        server_process.terminate()
        try:
            server_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server_process.kill()
            
    print("[RUNNER] Job complete. Exiting.", flush=True)

if __name__ == "__main__":
    main()
