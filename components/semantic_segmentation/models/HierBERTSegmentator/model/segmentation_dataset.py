# coding=utf-8

import datasets


logger = datasets.logging.get_logger(__name__)

_DESCRIPTION = """\
Dataset for text segmentation task
"""


class SegmentationConfig(datasets.BuilderConfig):
    """BuilderConfig for Segmentation Dataset"""

    def __init__(self, **kwargs):
        """BBuilderConfig for Segmentation Dataset.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(SegmentationConfig, self).__init__(**kwargs)


class Segmentation(datasets.GeneratorBasedBuilder):
    """Segmentation dataset."""

    BUILDER_CONFIGS = [
        SegmentationConfig(name="segmentation", version=datasets.Version("1.0.0"), description="Segmentation dataset"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "sentences": datasets.Sequence(datasets.Value("string")),
                    "segmentation_tags": datasets.Sequence(datasets.Value("int32")),
                }
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        """The `datafiles` kwarg in load_dataset() can be a str, List[str], Dict[str,str], or Dict[str,List[str]].

        If str or List[str], then the dataset returns only the 'train' split.
        If dict, then keys should be from the `datasets.Split` enum.
        """
        if not self.config.data_files:
            raise ValueError(f"At least one data file must be specified, but got data_files={self.config.data_files}")
        data_files = dl_manager.download_and_extract(self.config.data_files)

        if isinstance(data_files, (list, tuple)):
            raise ValueError(f"Downloading dataset from list (tuple) of files for TRAIN, TEST or VALIDATION is "
                             f"not supported yet ({self.config.data_files})")

        if isinstance(data_files, str):
            return [datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": data_files})]

        splits = []
        for split_name in [datasets.Split.TRAIN, datasets.Split.VALIDATION, datasets.Split.TEST]:
            if split_name in data_files:
                file = data_files[split_name]
                if not isinstance(file, str):
                    raise ValueError(f"Downloading dataset from list (tuple) of files for {split_name} is "
                                     f"not supported yet ({file})")
                splits.append(datasets.SplitGenerator(name=split_name, gen_kwargs={"filepath": file}))
        return splits

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            guid = 0
            sentences = []
            segmentation_tags = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if sentences:
                        yield guid, {
                            "sentences": sentences,
                            "segmentation_tags": segmentation_tags,
                        }
                        guid += 1
                        sentences = []
                        segmentation_tags = []
                else:
                    # sentence embeddings and labels are tab separated
                    splits = line.split("\t")
                    sentences.append(splits[0])
                    segmentation_tags.append(int(splits[1].rstrip()))
            # last example
            yield guid, {
                "sentences": sentences,
                "segmentation_tags": segmentation_tags,
            }
