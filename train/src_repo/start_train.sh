project_root_dir=/project/train/src_repo
dataset_dir=/home/data
log_file=/project/train/log/log.txt

echo "Prepare environment..."
pip install -i https://mirrors.cloud.tencent.com/pypi/simple -r ${project_root_dir}/BiSeNet/requirements.txt
echo "Prepare dataset..."
cd ${project_root_dir} && python split_dataset.py ${dataset_dir} | tee -a ${log_file}

echo "Start training..."
cd ${project_root_dir}/BiSeNet
# bisenetv2 expressage
export CUDA_VISIBLE_DEVICES=0
cfg_file=configs/bisenetv2_expressage.py
NGPUS=1
python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_amp.py --config $cfg_file | tee -a ${log_file}

echo "Convert model to onnx..."
python tools/export_onnx.py --config $cfg_file --weight-path /project/train/models/model_final.pth --outpath /project/train/models/model.onnx
python -m onnxsim /project/train/models/model.onnx /project/train/models/model_sim.onnx


