import json
from PIL import Image as PILImage
from datasets import (
    Image,
    Dataset,
    Features,
    Value,
    Sequence,
    Array2D,
    DatasetInfo,
    SplitDict,
    Split,
    DatasetDict,
    load_from_disk,
)

dataset_info = DatasetInfo(
    description="This dataset contains OCR data for text detection and recognition tasks. "
    "Each image has annotated bounding boxes, labels, and corresponding text.",
    citation="",
    license="MIT License",
    homepage="https://github.com/fcodelabs/intern-ml",
    features=Features(
        {
            "image": Image(),
            "height": Value("int32"),
            "width": Value("int32"),
            "annotations": Sequence(
                {
                    "box": Array2D(dtype="float32", shape=(4, 2)),
                    "text": Value("string"),
                    "label": Value("int32"),
                }
            ),
        }
    ),
    dataset_name="WildReceipt",
    splits=SplitDict(
        {
            "train": Split(name="train"),
            "test": Split("test"),
        }
    ),
)


def walk_through_json(file_name):
    # load the json file
    with open(file_name, "r") as fi:
        file = json.load(fi)

    # parse and reformat the data
    data = []
    for item in file:
        try:
            annotations = []
            for annotation in item["annotations"]:
                annotations.append(
                    {
                        "box": [
                            [annotation["box"][0], annotation["box"][1]],
                            [annotation["box"][2], annotation["box"][3]],
                            [annotation["box"][4], annotation["box"][5]],
                            [annotation["box"][6], annotation["box"][7]],
                        ],
                        "text": annotation["text"],
                        "label": annotation["label"],
                    }
                )
            data.append(
                {
                    "image": PILImage.open(item["file_name"]).convert("RGB"),
                    "height": item["height"],
                    "width": item["width"],
                    "annotations": annotations,
                }
            )
        except Exception as e:
            print(f"Error processing item {item['file_name']}: {e}")
    return data


train_data = walk_through_json("train.json")
test_data = walk_through_json("test.json")
train_dataset = Dataset.from_list(train_data, features=dataset_info.features)
test_dataset = Dataset.from_list(test_data, features=dataset_info.features)
dataset = DatasetDict(
    {
        "train": train_dataset,
        "test": test_dataset,
    }
)
dataset.info = dataset_info

# save the dataset locally
dataset.save_to_disk("ocr_dataset")
print("Dataset Created Successfully")

# push to the hub
loaded_dataset = load_from_disk("ocr_dataset")
loaded_dataset.push_to_hub(repo_id="fcodelabs/WildReceipt-OCR")
print(loaded_dataset)
