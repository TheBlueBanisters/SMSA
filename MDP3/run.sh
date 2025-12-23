#CUDA_VISIBLE_DEVICES=0 python run.py --data Video-MME --model MiniCPM-V-2_6 --nframe 128
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 torchrun --standalone --nproc-per-node 7 run.py --data Video-MME --model MiniCPM-V-2_6 --nframe 128
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 torchrun --standalone --nproc-per-node 7 run.py --data Video-MME --model MiniCPM-V-2_6 --nframe 128 --use-subtitle
