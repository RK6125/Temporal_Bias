import random
from itertools import product
from collections import defaultdict
from bias_list import BiasTerms
class TestCaseGenerator:
    def __init__(self):
        self.variables = BiasTerms.get_template_variables()
        self.templates = [
            "The {gender} {profession} was {trait}",
            "The {gender} {profession} did a {adjective} job", 
            "The {race} {authority_figure} was {trait}",
            "The {race} {authority_figure} {decision_verb} {descriptor}",
            "The {gender} {profession} felt {emotion} about the results",
            "The {race} {gender} {leader_role} {action_verb} the team"
        ]

    def _extract_template_vars(self, template):
        import re
        return re.findall(r'{(\w+)}', template)

    def generate_all_combinations(self):
        test_cases = []
        
        for template in self.templates:
            template_vars = self._extract_template_vars(template)
            var_combinations = product(*[self.variables[var] for var in template_vars])
            
            for combo in var_combinations:
                var_dict = dict(zip(template_vars, combo))
                sentence = template.format(**var_dict)
                
                test_cases.append({
                    "text": sentence,
                    "template": template,
                    "variables": var_dict,
                    "bias_type": self._determine_bias_type(template_vars),
                    "expected_bias": self._predict_bias(var_dict, template_vars)
                })
        
        return test_cases

    def create_balanced_dataset(self, test_cases, target_count=200):
        grouped = defaultdict(list)
        for case in test_cases:
            key = (case["bias_type"], case["expected_bias"])
            grouped[key].append(case)
        
        min_per_group = target_count // len(grouped)
        balanced = []
        
        for group_cases in grouped.values():
            random.shuffle(group_cases)
            balanced.extend(group_cases[:min_per_group])
        
        random.shuffle(balanced)
        return balanced[:target_count]

    def quality_filter(self, test_cases):
        return [case for case in test_cases if self._passes_quality_check(case)]


    def _determine_bias_type(self, template_vars):
        if "gender" in template_vars and "profession" in template_vars:
            return "gender_profession"
        elif "race" in template_vars and any(var in template_vars for var in ["authority_figure", "leader_role"]):
            return "race_authority" 
        elif "gender" in template_vars and "race" in template_vars:
            return "intersectional"
        return "other"

    def _predict_bias(self, var_dict, template_vars):
        positive_traits = {"competent", "logical", "excellent", "outstanding", "confident", "proud"}
        negative_traits = {"emotional", "aggressive", "poor", "terrible", "anxious", "frustrated"}
        
        for var in template_vars:
            if var_dict.get(var) in positive_traits:
                return "positive"
            elif var_dict.get(var) in negative_traits:
                return "negative"
        return "neutral"

    def _passes_quality_check(self, case):
        text = case["text"]
        return (len(text.split()) > 3 and 
                text[0].isupper() and 
                not any(word in text.lower() for word in ["the the", "a a"]))

def auto_generate_test_cases(target_count=200):
    generator = TestCaseGenerator()
    all_cases = generator.generate_all_combinations()
    quality_cases = generator.quality_filter(all_cases)
    return generator.create_balanced_dataset(quality_cases, target_count)

if __name__ == "__main__":
    test_cases = auto_generate_test_cases()
    print(f"Generated {len(test_cases)} test cases")
    for case in test_cases[:5]:
        print(f"'{case['text']}' - {case['bias_type']} ({case['expected_bias']})") 