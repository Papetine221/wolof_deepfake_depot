from huggingface_hub import snapshot_download
for repo, d in [('facebook/wav2vec2-xls-r-300m','facebook__wav2vec2-xls-r-300m'),('facebook/mms-300m','facebook__mms-300m')]:
    p = snapshot_download(repo, local_dir='pretrained_models/'+d, allow_patterns=['*.json','*.safetensors','*.bin','*.txt'] )
    print(repo, p, flush=True)
