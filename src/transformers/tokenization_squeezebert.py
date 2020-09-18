# coding=utf-8
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
"""Tokenization classes for SqueezeBERT."""

from .tokenization_bert import BertTokenizer, BertTokenizerFast
from .utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "squeezebert-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/squeezebert/squeezebert-uncased/vocab.txt",
        "squeezebert-mnli": "https://s3.amazonaws.com/models.huggingface.co/bert/squeezebert/squeezebert-mnli/vocab.txt",
        "squeezebert-mnli-headless": "https://s3.amazonaws.com/models.huggingface.co/bert/squeezebert/squeezebert-mnli-headless/vocab.txt",
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {}


PRETRAINED_INIT_CONFIGURATION = {}


class SqueezeBertTokenizer(BertTokenizer):
    r"""
    Constructs a  SqueezeBertTokenizer.

    :class:`~transformers.SqueezeBertTokenizer is identical to :class:`~transformers.BertTokenizer` and runs end-to-end
    tokenization: punctuation splitting + wordpiece.

    Refer to superclass :class:`~transformers.BertTokenizer` for usage examples and documentation concerning
    parameters.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION


class SqueezeBertTokenizerFast(BertTokenizerFast):
    r"""
    Constructs a  "Fast" SqueezeBertTokenizer (backed by HuggingFace's `tokenizers` library).

    :class:`~transformers.SqueezeBertTokenizerFast` is identical to :class:`~transformers.BertTokenizerFast` and runs end-to-end
    tokenization: punctuation splitting + wordpiece.

    Refer to superclass :class:`~transformers.BertTokenizerFast` for usage examples and documentation concerning
    parameters.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
