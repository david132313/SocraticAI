#David Zhu
from pprint import pprint

# Import the adjust function from where you put it
from extractors import adjust_difficulty_based_on_history

# 1. Pick a concept name to test
test_concepts = [
    "Machine Learning",
    "Parameter Efficient Fine-Tuning",
    "Dimensionality Reduction",
    "TRetrieval-Augmented Generation",
    "LangChain Expression Language",
    "Vector Database"
]

# 2. Try different settings
similarity_thresholds = [0.75, 0.82, 0.9]
lower_bys = [1, 2]

for concept in test_concepts:
    for threshold in similarity_thresholds:
        for lower in lower_bys:
            adjusted = adjust_difficulty_based_on_history(
                concept_name=concept,
                base_difficulty=4,
                data_dir="data",  # your folder with jsons
                similarity_threshold=threshold,
                lower_by=lower
            )
            pprint({
                "concept": concept,
                "threshold": threshold,
                "lower_by": lower,
                "adjusted_difficulty": adjusted
            })
