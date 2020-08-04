Test task
Fire detectron on Facebook Detectron2 software system

## Quick Start
Run Demo
```
python demo --output_dir ./reports --model_path {model_path} --input_dir ./data/train --metadata_dir ./data \
  --thresh_tes 0.85 --opts {add_params} --mode image
```

## Train
Run trainer
```
python demo --batch_size 32 --num_workers 4 --max_iter 10000 --lr 0.0001 \
  --output_dir ../reports --model_path COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml \
  --input_dir ../data --checkpoint_step 200 --eval_step 256 \
  --img_size 256 --opts {add_params}
```


