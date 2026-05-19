from benchmarks.arc import load_arc
from benchmarks.gsm8k import load_gsm8k
from benchmarks.commonsenseqa import load_commonsenseqa
from benchmarks.strategyqa import load_strategyqa
from benchmarks.truthfulqa import load_truthfulqa


BENCHMARKS = {

    "arc": load_arc,
    "gsm8k": load_gsm8k,
    "commonsenseqa": load_commonsenseqa,
    "strategyqa": load_strategyqa,
    "truthfulqa": load_truthfulqa

}