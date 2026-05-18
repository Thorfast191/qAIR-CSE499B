from models.full_model import QAIRvNext


ABLATIONS = {

    "A1_baseline": {
        "use_quantum": False,
        "use_validator": False
    },

    "A8_full_hybrid": {
        "use_quantum": True,
        "use_validator": True
    },

    "A10_persistent": {
        "use_quantum": True,
        "use_validator": True,
        "persistent_steps": 5
    }
}


def run_ablation_suite():

    print("Running ablations...")

    for name, cfg in ABLATIONS.items():
        print(name, cfg)