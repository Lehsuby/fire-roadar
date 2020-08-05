import argparse
import torch
from pathlib import Path
from datetime import datetime
from src.utils import register_data

from detectron2.data import transforms
from detectron2.data import DatasetMapper
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader, build_detection_train_loader


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='number of data loading workers')
    parser.add_argument('--max_iter', type=int, default=10000, help='number of iterations to train for')
    parser.add_argument('--lr', type=float, default=0.0001, help='start learning rate')
    parser.add_argument('--output_dir', type=str, default='../reports', help='output folder')
    parser.add_argument('--model_path', type=str,
                        default="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
                        help='model path')
    parser.add_argument('--input_dir', type=str, default='../data',
                        help='input data folder')
    parser.add_argument('--checkpoint_step', type=int, default=200, help="save period")
    parser.add_argument('--eval_step', type=int, default=1000, help="evaluation period")
    parser.add_argument('--img_size', type=int, default=256, help="min image side size")
    parser.add_argument('--opts', default=[], help='additional params')
    return parser


class Learner:
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Data
        print("Load data")
        dataset_names, self.num_classes, _ = register_data("fire", self.args.input_dir)
        self.train_data_name = dataset_names["train"]
        self.val_data_name = dataset_names["validation"]

        # Model
        print("Load model")
        self.cfg = self.create_cfg()

        self.trainer = DetectronTrainer(self.cfg)
        self.trainer.resume_or_load(resume=False)

    def create_cfg(self):
        # Create config
        cfg = get_cfg()
        cfg.merge_from_list(self.args.opts)

        cfg.merge_from_file(model_zoo.get_config_file(self.args.model_path))

        output_dir = Path(self.args.output_dir, "logs_{:%Y-%m-%d_%H:%M:%S}".format(datetime.now()))
        output_dir.mkdir(parents=True, exist_ok=True)
        cfg.OUTPUT_DIR = str(output_dir)

        cfg.INPUT.MIN_SIZE_TRAIN = (self.args.img_size,)
        cfg.INPUT.MIN_SIZE_TEST = self.args.img_size

        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.args.model_path)
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = self.args.batch_size
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.num_classes
        cfg.MODEL.DEVICE = self.device

        cfg.DATASETS.TRAIN = (self.train_data_name,)
        cfg.DATASETS.TEST = (self.val_data_name,)

        cfg.DATALOADER.NUM_WORKERS = self.args.num_workers

        cfg.SOLVER.CHECKPOINT_PERIOD = self.args.checkpoint_step
        cfg.SOLVER.MAX_ITER = self.args.max_iter

        cfg.TEST.EVAL_PERIOD = self.args.eval_step
        cfg.freeze()
        print(cfg)
        return cfg

    def eval_test_dataset(self):
        evaluater = self.trainer.build_evaluator(self.cfg, self.val_data_name)
        val_loader = self.trainer.build_test_loader(self.cfg, self.val_data_name)
        results = inference_on_dataset(self.trainer.model, val_loader, evaluater)

        return results

    def train(self):
        print("Train model")
        self.trainer.train()


class DetectronTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        print(cfg.INPUT.MIN_SIZE_TRAIN)
        mapper = DatasetMapper(cfg, is_train=True, augmentations=[
            transforms.Resize(cfg.INPUT.MIN_SIZE_TEST),
            transforms.RandomFlip()
        ])
        return build_detection_train_loader(cfg, mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        mapper = DatasetMapper(cfg, is_train=False, augmentations=[
            transforms.Resize(cfg.INPUT.MIN_SIZE_TEST)
        ])
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        output_folder = cfg.OUTPUT_DIR

        return COCOEvaluator(dataset_name, cfg, False, output_folder)


def run():
    parser = get_parser()
    args = parser.parse_args()
    print(args)

    learner = Learner(args)
    learner.train()
    learner.eval_test_dataset()


if __name__ == '__main__':
    run()