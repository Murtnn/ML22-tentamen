from pathlib import Path
from typing import Union

from pydantic import BaseModel, HttpUrl
from ray import tune

SAMPLE_INT = tune.search.sample.Integer
SAMPLE_FLOAT = tune.search.sample.Float

cwd = Path(__file__)
root = (cwd / "../..").resolve()


class Settings(BaseModel):
    datadir: Path
    testurl: HttpUrl
    trainurl: HttpUrl
    testfile: Path
    trainfile: Path
    modeldir: Path
    logdir: Path
    modelname: str
    batchsize: int


presets = Settings(
    datadir=root / "data/raw",
    testurl="https://archive.ics.uci.edu/ml/machine-learning-databases/00195/Test_Arabic_Digit.txt",  # noqa N501
    trainurl="https://archive.ics.uci.edu/ml/machine-learning-databases/00195/Train_Arabic_Digit.txt",  # noqa N501
    testfile=Path("ArabicTest.txt"),
    trainfile=Path("ArabicTrain.txt"),
    modeldir=root / "models",
    logdir=root / "logs",
    modelname="model.pt",
    batchsize=128,
)


class BaseSearchSpace(BaseModel):
    input: int
    output: int
    tunedir: Path

    class Config:
        arbitrary_types_allowed = True


class LinearConfig(BaseSearchSpace):
    h1: int
    h2: int
    dropout: float


class LinearSearchSpace(BaseSearchSpace):
    h1: Union[int, SAMPLE_INT] = tune.randint(16, 128)
    h2: Union[int, SAMPLE_INT] = tune.randint(16, 128)
    dropout: Union[float, SAMPLE_FLOAT] = tune.uniform(0.0, 0.5)


class GRUmodelConfig(BaseSearchSpace):
    hidden_size: int
    num_layers: int
    dropout: float


class GRUmodelSearchSpace(BaseSearchSpace):
    hidden_size: Union[int, SAMPLE_INT] = tune.qrandint(200, 260, 10)
    num_layers: Union[int, SAMPLE_INT] = tune.qrandint(4, 6, 1)
    dropout: Union[float, SAMPLE_FLOAT] = tune.quniform(0.1, 0.3, 0.05)
    batchsize: Union[int, SAMPLE_INT] = tune.qrandint(64, 182, 16)

#LET OP: onderste class toegevoegd met parameters obv 1d.
#class GRUmodelSearchSpace(BaseSearchSpace):
#    hidden_size: Union[int, SAMPLE_INT] = tune.randint(128, 256)
#    num_layers: Union[int, SAMPLE_INT] = tune.randint(2, 6)
#    dropout: Union[float, SAMPLE_FLOAT] = tune.uniform(0.0, 0.3)

# class GRUmodelSearchSpace(BaseSearchSpace):
#    hidden_size: Union[int, SAMPLE_INT] = tune.qrandint(128, 256, 32)
#    num_layers: Union[int, SAMPLE_INT] = tune.qrandint(2, 6, 2)
#    dropout: Union[float, SAMPLE_FLOAT] = tune.quniform(0.0, 0.3, 0.05)