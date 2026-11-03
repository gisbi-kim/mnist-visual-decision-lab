"""Download the versioned public weight archive and verify all SHA-256 hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.request
import zipfile

ROOT=Path(__file__).resolve().parent

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def verify(directory,manifest):
    for name,meta in manifest['files'].items():
        path=directory/name
        if not path.is_file() or path.stat().st_size!=meta['bytes'] or digest(path)!=meta['sha256']:
            return False
    return True

def download_base(manifest):
    from huggingface_hub import snapshot_download
    print('Downloading/checking the pinned Qwen base model...',flush=True)
    folder=Path(snapshot_download(manifest['base_model'],revision=manifest['base_revision'],
        allow_patterns=list(manifest['base_files'])))
    if not verify(folder,{'files':manifest['base_files']}):
        raise RuntimeError('Base model checksum mismatch. Check the Hugging Face cache before retrying.')
    print('Verified base model:',folder)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--directory',default=str(ROOT/'weights'))
    args=parser.parse_args()
    manifest_path=ROOT/'release_manifest.json'
    if not manifest_path.exists():raise SystemExit('A verified weight release has not been published yet.')
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    target=Path(args.directory).resolve()
    if target.exists():
        if verify(target,manifest):
            print('Verified cached weights:',target)
            download_base(manifest)
            return
        raise SystemExit('Existing weight directory does not match this release. Choose a new --directory.')
    target.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='mnist-download-',dir=target.parent) as tmp:
        temp=Path(tmp);archive=temp/'weights.zip'
        def progress(block,size,total):
            if block%128==0:print(f'Weights: {min(block*size,total)/1024**2:.1f}/{total/1024**2:.1f} MiB',flush=True)
        urllib.request.urlretrieve(manifest['url'],archive,progress)
        if digest(archive)!=manifest['sha256']:raise RuntimeError('Archive SHA-256 mismatch')
        unpacked=temp/'unpacked';unpacked.mkdir()
        with zipfile.ZipFile(archive) as z:
            if set(z.namelist())!=set(manifest['files']):raise RuntimeError('Archive contents do not match manifest')
            for name in z.namelist():
                if Path(name).name!=name or '\\' in name:raise RuntimeError('Unsafe archive member')
                (unpacked/name).write_bytes(z.read(name))
        if not verify(unpacked,manifest):raise RuntimeError('Extracted weight checksum mismatch')
        unpacked.rename(target)
    print('Verified weights:',target)
    download_base(manifest)

if __name__=='__main__':main()
