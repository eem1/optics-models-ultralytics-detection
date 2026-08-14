import os
from google.cloud import storage
from concurrent.futures import ThreadPoolExecutor

def uri_extract(uri):
    u_parts = uri.replace('gs://', '').split('/', 1)
    filename = os.path.basename(uri.strip('/'))
    bucket_name = u_parts[0]
    blob = u_parts[1]
    return bucket_name, blob, filename


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
    

def copy_dataset_to_gcs_output_folder(uri_list, destination_bucket, destination_folder):
    """
    Upload files from /tmp
    """
    if not uri_list:
        return []

    client = storage.Client()

    dest_folder = f'{destination_folder.rstrip("/")}'

    def copy_single_blob(uri):
        source_bucket, blob, filename = uri_extract(uri)
       
        source_blob = client.bucket(source_bucket).blob(blob)
        source_bucket = client.bucket(source_bucket)
        dest_bucket = client.bucket(destination_bucket)
        dest_blob = f'{dest_folder}/{filename}'
        source_bucket.copy_blob(source_blob, dest_bucket, dest_blob)
        
        return f"Successfully copied {uri} to {destination_bucket}/{dest_blob}"

    with ThreadPoolExecutor(max_workers=16) as executor:
        list(executor.map(copy_single_blob, uri_list))