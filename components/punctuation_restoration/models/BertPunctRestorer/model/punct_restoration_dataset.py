# coding=utf-8
# Copyright 2020 HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3

import datasets
import json


logger = datasets.logging.get_logger(__name__)


_CITATION = """\
No citation yet
"""

_DESCRIPTION = """\
Dataset for punctuation and capitalization restoration in russian texts. This dataset has the same format as 
the dataset for shared task CoNLL-2003, but with other labels
"""


# TODO:
#   1. Сделать несколько датасетов с захардкожеными метками
#      (необходимый минимум меток, все метки, только пунктуация и т.д.)

# Вот это костыль, и без него пока не работает
# В колабе нужно указать абсолютный путь до файла с метками
with open('./labels.json') as json_file:
    LABELS = json.load(json_file)


class PunctRestorationConfig(datasets.BuilderConfig):
    """BuilderConfig for PunctRestoration"""

    def __init__(self, **kwargs):
        """BuilderConfig for PunctRestoration.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(PunctRestorationConfig, self).__init__(**kwargs)


class PunctRestoration(datasets.GeneratorBasedBuilder):
    """PunctRestoration dataset."""

    BUILDER_CONFIGS = [
        PunctRestorationConfig(name="punct_restoration", version=datasets.Version("1.0.0"),
                               description="Punctuation restoration dataset"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "punct_restoration_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=LABELS
                        )
                    ),
                }
            ),
            supervised_keys=None,
            homepage="",
            citation=_CITATION,
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
            tokens = []
            punct_restoration_tags = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if tokens:
                        yield guid, {
                            "tokens": tokens,
                            "punct_restoration_tags": punct_restoration_tags,
                        }
                        guid += 1
                        tokens = []
                        punct_restoration_tags = []
                else:
                    # conll2003 tokens are space separated
                    splits = line.split(" ")
                    tokens.append(splits[0])
                    punct_restoration_tags.append(splits[1].rstrip())
            # last example
            yield guid, {
                "tokens": tokens,
                "punct_restoration_tags": punct_restoration_tags,
            }
