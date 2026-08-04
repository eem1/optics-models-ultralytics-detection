from google.cloud import storage

def download_gcs_uri(uri: str, destination: str):
    """
    Downloads a single GCS file to a local destination.
    
    Parameters
    ----------
    uri : str
        The full gs:// URI of the file to download.
    destination : str
        The local file path where the file should be saved.
    """
    if not uri.startswith("gs://"):
        raise ValueError(f"URI must start with gs://. Got: {uri}")
        
    client = storage.Client()
    bucket_name = uri.replace("gs://", "").split("/")[0]
    blob_path = "/".join(uri.replace("gs://", "").split("/")[1:])
    
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.download_to_filename(destination)
