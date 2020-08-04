from collections import deque, Counter
from xml.etree import ElementTree
from pathlib import Path
import pandas as pd
from detectron2.structures import BoxMode
import itertools
from detectron2.data import DatasetCatalog, MetadataCatalog
import cv2


def change_xml_paths(dataset_path: str):
    """
        Change inner path to file in xml
        Args:
            dataset_path (str): Folder with xml files
    """
    dataset_path = Path(dataset_path)
    for xml_path in dataset_path.rglob("*.xml"):
        rel_path_parts = [part for part in xml_path.parts if part not in dataset_path.parts]

        rel_path_parts[rel_path_parts.index("annotations")] = "images"
        rel_img_path = Path(*rel_path_parts)

        xml_tree = ElementTree.parse(xml_path)
        root = xml_tree.getroot()

        img_suffix = Path(root.find("path").text).suffix
        root.find("path").text = str(rel_img_path.with_suffix(img_suffix))

        xml_tree.write(xml_path)


def get_inner_names(root: ElementTree):
    """
        Get list of all childs names
        Args:
            root (ElementTree): Root of the tree

        Returns:
            (list): list of all childs names
    """
    column_names = []
    queue = deque([root])

    while queue:
        childs, child_names = get_childs(queue.popleft())
        queue.extend(childs)
        column_names.extend(child_names)

    return column_names


def get_childs(root: ElementTree):
    """
        Get list of childs and their names
        Args:
            root (ElementTree): Root of the tree

        Returns:
            (list): list of childs by root.
            (list): list of child names
    """
    names = [child.tag for child in root]
    return list(root), names


def convert_xml_to_df(dataset_path: str):
    """
        Convert xml format to pandas Dataframe
        Args:
            dataset_path (str): Folder with xml files

        Returns:
            (pd.DataFrame): Dataframe from xml files
    """
    dataset_path = Path(dataset_path)
    dataset = []

    for xml_path in dataset_path.rglob("*.xml"):
        xml_tree = ElementTree.parse(xml_path)

        root = xml_tree.getroot()
        dataset_elem = {}

        column_names = get_inner_names(root)
        cnt = Counter(column_names)
        num_objs = cnt['object']

        uniqie_column_names = list(set(column_names))
        obj_names = get_inner_names(root.find("object"))

        for name in set(uniqie_column_names) - set(obj_names):
            dataset_elem[name] = list(root.iter(name))[0].text

        dataset_elems = [dataset_elem.copy() for i in range(num_objs)]
        for idx, obj in enumerate(root.iter("object")):
            for name in obj_names:
                dataset_elems[idx][name] = list(obj.iter(name))[0].text

        dataset.extend(dataset_elems)

    dataset = pd.DataFrame(dataset)
    return dataset


def cut_df(df: pd.DataFrame, column_names: list):
    """
        Cut Dataframe columns
        Args:
            df (pd.DataFrame): Full DataFrame
            column_names (list): List of new DataFrame

        Returns:
            (pd.DataFrame): Short version of Dataframe with current columns
    """
    df = df[column_names]
    return df


def create_dataset_dicts(df: pd.DataFrame, classes: list, data_folder: str):
    """
        Create list of dicts from dataset elements
        Args:
            df (str): Dataset DataFrame
            classes (list): List of current classes
            data_folder (str): Path to images which consist in dataset

        Returns:
            (list): List of dicts of dataset elements
    """
    dataset_dicts = []
    for image_id, img_rel_path in enumerate(df['path'].unique()):

        record = {}

        image_df = df[df["path"] == img_rel_path]

        file_path = Path(data_folder, img_rel_path)
        record["file_name"] = str(file_path)
        record["image_id"] = image_id
        record["height"] = int(image_df.iloc[0]['height'])
        record["width"] = int(image_df.iloc[0]['width'])

        objs = []
        for _, row in image_df.iterrows():
            xmin = int(row.xmin)
            ymin = int(row.ymin)
            xmax = int(row.xmax)
            ymax = int(row.ymax)

            poly = [
                (xmin, ymin), (xmax, ymin),
                (xmax, ymax), (xmin, ymax)
            ]
            poly = list(itertools.chain.from_iterable(poly))

            obj = {
                "bbox": [xmin, ymin, xmax, ymax],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": classes.index(row['name']),
                "iscrowd": 0
            }
            objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


def save_short_df(name_csv: str):
    """
        Create and write cut version of Dataframe.
        Args:
            name_csv (str): Name of output csv file
    """
    column_names = ["path", "height", "filename", "width", "xmin", "xmax", "ymin", "ymax", "name"]
    for df_path in ["../../data/train", "../../data/validation"]:
        df = convert_xml_to_df(df_path)
        df = cut_df(df, column_names)
        df.to_csv(Path(df_path, name_csv))


def remove_spaces_in_path(folder_path: str):
    """
        Remove all spaces in files paths.
        Args:
            folder_path (str): Path to folder with files
    """
    for file_path in Path(folder_path).iterdir():
        if file_path.is_file():
            new_name = file_path.name.replace(" ", "")
            file_path.rename(Path(file_path.parent, new_name))


def remove_spaces_in_xml(dataset_path: str):
    """
        Remove all spaces in xml features.
        Args:
            dataset_path (str): Folder with xml files
    """
    dataset_path = Path(dataset_path)
    for xml_path in dataset_path.rglob("*.xml"):
        xml_tree = ElementTree.parse(xml_path)
        root = xml_tree.getroot()

        img_path = root.find("path").text.replace(" ", "")
        root.find("path").text = img_path

        img_name = root.find("filename").text.replace(" ", "")
        root.find("filename").text = img_name

        xml_tree.write(xml_path)


def register_data(data_name: str, data_folder: str, csv_name: str = "data.csv"):
    """
        Register datasets in Detectron2
        Args:
            data_name (str): Name of dataset
            data_folder (str): Folder with all parts of datsets
            csv_name (str): Name of csv

        Returns:
            (dict): Dictionary of dataset names
            (int): Num classes
            (dict): Dictionary of metadata of datasets
    """
    dataset_names = {}
    metadata = {}
    for part in ["train", "validation"]:
        df = pd.read_csv(Path(data_folder, part, csv_name))
        classes = list(df['name'].unique())
        dataset_name = f"{data_name}_{part}"
        dataset_names[part] = dataset_name
        DatasetCatalog.register(dataset_name, lambda: create_dataset_dicts(df, classes, data_folder))
        MetadataCatalog.get(dataset_name).set(thing_classes=classes)

        metadata[part] = MetadataCatalog.get(dataset_name)
    return dataset_names, len(classes), metadata


def get_video_params(video_path: str):
    """
        Get video params.
        Args:
            video_path (str): Path to vidoe

        Returns:
            (dict): Dictionary with video params
    """
    video_path = Path(video_path)
    video = cv2.VideoCapture(str(video_path))
    video_info = {
        "width": int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(video.get(cv2.CAP_PROP_FPS)),
        "num_frames": int(video.get(cv2.CAP_PROP_FRAME_COUNT)),
        "name": video_path.name
    }

    return video_info


if __name__ == '__main__':
    save_short_df("data.csv")
    # remove_spaces_in_path("../../data/validation/annotations")
    # remove_spaces_in_path("../../data/validation/images")
    # remove_spaces_in_xml("../../data/validation")

