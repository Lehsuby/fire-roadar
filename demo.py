import argparse
from pathlib import Path
import cv2
import torch
from tqdm import tqdm
from datetime import datetime

from src.utils import register_data, get_video_params, video_capture, video_writer
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.data.detection_utils import read_image
from detectron2.data import MetadataCatalog


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='./reports', help='output folder')
    parser.add_argument('--model_path', type=str,
                        default="./reports/logs_2020-08-04_21:26:36/model_0001399.pth",
                        help='model path')
    parser.add_argument('--input_dir', type=str, default='./data/test', help='input data folder')
    parser.add_argument('--thresh_test', type=float, default=0.9,
                        help='Minimum score threshold (assuming scores in a [0, 1] range), like NMS')
    parser.add_argument('--opts', type=list, default=[], help='additional params')
    parser.add_argument('--mode', type=str, default="video", help='Type of input file [image, video]')
    parser.add_argument('--visualize', type=bool, default=False, help='Flag of visualization')
    return parser


def visualize_img(window_name: str, img, mode: str):
    """
        Visualize image in window
        Args:
            window_name (str): Path to video
            img (Image): Image
            mode (str): Mode of file ["image", "video"]
    """
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, img)
    flag = 1 if mode == "video" else 0
    while True:
        if cv2.waitKey(flag) == 27:
            break


class Evaluator:
    def __init__(self, args, instance_mode=ColorMode.IMAGE):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.instance_mode = instance_mode

        self.output_dir = Path(self.args.output_dir, "results_{:%Y-%m-%d_%H:%M:%S}".format(datetime.now()))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Data
        print("Load metadata")
        self.dataset_name = "fire_val"
        MetadataCatalog.get(self.dataset_name).set(thing_classes=["fire"])
        self.metadata = MetadataCatalog.get(self.dataset_name)
        self.cfg = self.create_cfg()

        # Predictor
        print("Load predictor")
        self.predictor = DefaultPredictor(self.cfg)

    def create_cfg(self):
        # Create config
        cfg = get_cfg()
        cfg.merge_from_list(self.args.opts)

        cfg.MODEL.WEIGHTS = self.args.model_path
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.args.thresh_test
        cfg.DATASETS.TEST = (self.dataset_name,)

        cfg.MODEL.DEVICE = self.device

        cfg.freeze()
        print(cfg)
        return cfg

    def predict(self, obj, mode: str = "image"):
        # Make prediction
        if mode == "image":
            image = obj[:, :, ::-1]
            image_visualizer = Visualizer(image, metadata=self.metadata, instance_mode=self.instance_mode, scale=1.2)
            outputs = self.predictor(obj)
            instances = outputs["instances"].to("cpu")
            instances.remove('pred_classes')
            vis_output = image_visualizer.draw_instance_predictions(instances)
        elif mode == "video":
            video_visualizer = VideoVisualizer(metadata=self.metadata, instance_mode=self.instance_mode)
            outputs, vis_output = [], []
            while obj.isOpened():
                success, frame = obj.read()
                if success:
                    output = self.predictor(frame)
                    outputs.append(output)
                    instances = output["instances"].to("cpu")

                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                    vis_frame = video_visualizer.draw_instance_predictions(frame, instances)
                    vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)

                    vis_output.append(vis_frame)
                else:
                    break

        return outputs, vis_output

    def infer(self, data_path: str, mode: str = "image", visualize: bool = False):
        """
            Inference input video or image
            Args:
                data_path (str): Path to video or image
                mode (str): Mode of file ["image", "video"]
                visualize (bool): Flag of visualization
        """
        # Inference
        data_path = Path(data_path)
        file_paths = []
        if data_path.is_file():
            file_paths.append(data_path)
        elif data_path.is_dir():
            file_paths.extend([path for path in data_path.iterdir()])

        for file_path in tqdm(file_paths, total=len(file_paths)):
            output_path = Path(self.output_dir, file_path.name)

            if mode == "image":
                img = read_image(str(file_path), format="BGR")
                outputs, vis_output = self.predict(img, mode)
                img = vis_output.get_image()[:, :, ::-1]

                vis_output.save(str(output_path))

                if visualize:
                    visualize_img(data_path.name, img, mode)

            if mode == "video":
                video_info = get_video_params(file_path)
                with video_capture(str(file_path)) as video:

                    outputs, vis_output = self.predict(video, mode)
                    output_path = output_path.with_suffix(".mkv")

                    with video_writer(str(output_path), video_info) as output_file:

                        for frame in tqdm(vis_output, total=video_info['num_frames']):
                            output_file.write(frame)

                            if visualize:
                                visualize_img(video_info['name'], frame, mode)


def run():
    parser = get_parser()
    args = parser.parse_args()
    print(args)

    evaluater = Evaluator(args)
    evaluater.infer(args.input_dir, mode=args.mode, visualize=args.visualize)


if __name__ == '__main__':
    run()
