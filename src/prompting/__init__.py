from prompting.data_utils import (
    AGNewsProcessor,
    AmazonProcessor,
    BaseProcessor,
    SST2Processor,
    TRECProcessor,
)
from prompting.models import (
    GenerationOutput,
    GPT2Wrapper,
    GPT3Wrapper,
    PromptTooLongError,
)
from prompting.strategies import (
    BaseStrategy,
    BestOfKStrategy,
    BestPermStrategy,
    GlobalEntropyOrderingStrategy,
    MaxEntropyStrategy,
    OracleStrategy,
    RandomClassBalancedStrategy,
    RandomStrategy,
)

PROCESSORS = {
    "sst-2": SST2Processor,
    "agnews": AGNewsProcessor,
    "trec": TRECProcessor,
    "amazon": AmazonProcessor,
}

STRATEGIES = {
    "random": RandomStrategy,
    "random-class-balanced": RandomClassBalancedStrategy,
    "best-perm": BestPermStrategy,
    "best-of-k": BestOfKStrategy,
    "max-entropy": MaxEntropyStrategy,
    "oracle": OracleStrategy,
    "global-entropy-ordering": GlobalEntropyOrderingStrategy,
}
MODELS = {"gpt2": GPT2Wrapper, "gpt3": GPT3Wrapper}
