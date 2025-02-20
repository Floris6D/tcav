from activation_generator import ImageActivationGenerator
from concept_heuristics import compute_RCV, compute_CAR, compute_pattern_CAV, compute_SVM_CAV, compute_RCR


CLASSIFICATION_BASED = ["filter_cav", "pattern_cav_reg", "pattern_cav_svc", "car"]
REGRESSION_BASED = ["rcv", "rcr"]
def get_cav(model, source_dir, bottlenecks, concept, random_counterpart=None, labels=None, method="filter_cav"):
        # Check if method is valid
        classification_based = method in CLASSIFICATION_BASED
        if not classification_based and method not in REGRESSION_BASED:
            raise ValueError(f"Method {method} not specified, choose from {CLASSIFICATION_BASED + REGRESSION_BASED}")
        if not (random_counterpart is None and labels is None):
            raise ValueError("Either random_counterpart or labels must be None")
        if classification_based and random_counterpart is None:
            raise ValueError(f"Type {method} requires a random counterpart")
        if not classification_based and labels is None:
            raise ValueError(f"Type {method} requires labels")
        if not random_counterpart is None and not labels is None:
            print("Warning: Both random_counterpart and labels are not None ")
            if classification_based: print(f"For {method} random_counterpart is only needed")
            else: print(f"For {method} labels are only needed")
        # send it
        act_gen = ImageActivationGenerator(model, source_dir, activation_dir=None, max_examples=500)
        if not isinstance(bottlenecks, list):
            bottlenecks = [bottlenecks]
        for bottleneck in bottlenecks:
            acts_concept = act_gen.get_activations_for_concept(concept, bottleneck)
            if not random_counterpart is None:
                acts_random = act_gen.get_activations_for_concept(random_counterpart, bottleneck)
            if method == "filter_cav":
                cav = compute_SVM_CAV(acts_concept, acts_random)
            elif method == "pattern_cav_reg":
                cav = compute_pattern_CAV(acts_concept, acts_random, model_type="linear")
            elif method == "pattern_cav_svc":
                cav = compute_pattern_CAV(acts_concept, acts_random, model_type="svm")
            elif method == "car":
                car = compute_CAR(acts_concept, acts_random)
            elif method == "rcv":
                rcv = compute_RCV(acts_concept, labels)
            elif method == "rcr":
                rcr = compute_RCR(acts_concept, labels)
                #TODO: turn into metrics for loss
            
            
            