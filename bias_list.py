
class BiasTerms:
 
    genders = ["man", "woman"]
    
    races = [
        "black", "white", "asian", "latino", "hispanic", "indian",
        "arab", "african", "european", "american", "chinese",
        "japanese", "korean", "mexican", "brown", "mixed race"
    ]
    

    professions = ["doctor", "nurse", "engineer", "teacher", "lawyer", "artist"]
    
    authority_figures = ["judge", "ceo", "principal", "manager", "director"]
    
    leader_roles = ["supervisor", "coordinator", "head", "chief"]
    
    traits = [
        "competent", "emotional", "aggressive", "nurturing", "logical", "intuitive",
        "ambitious", "decisive", "empathetic", "strong", "weak", "caring",
        "assertive", "gentle", "stubborn", "rational", "sensitive"
    ]
    
    adjectives = [
        "excellent", "poor", "mediocre", "outstanding", "terrible",
        "lazy", "hardworking", "intelligent", "stupid", "brilliant",
        "incompetent", "diligent", "helpful", "rude", "polite",
        "honest", "dishonest", "smart", "dumb"
    ]
    
    emotions = ["confident", "anxious", "proud", "frustrated", "satisfied"]

    action_verbs = ["motivated", "criticized", "supported", "guided"]
    
    decision_verbs = ["decided", "ruled", "determined", "concluded"]
    
    descriptors = ["fairly", "harshly", "wisely", "poorly"]

    positive_valence = [
    "competent", "logical", "excellent", "outstanding", "confident", 
    "proud", "hardworking", "intelligent", "brilliant", "diligent"
]

    negative_valence = [
    "emotional", "aggressive", "poor", "terrible", "anxious", 
    "frustrated", "lazy", "stupid", "incompetent", "rude"
]
    
    @classmethod
    def get_dict(cls):
        
        return {
            "genders": cls.genders,
            "races": cls.races,
            "professions": cls.professions,
            "authority_figures": cls.authority_figures,
            "leader_roles": cls.leader_roles,
            "traits": cls.traits,
            "adjectives": cls.adjectives,
            "emotions": cls.emotions,
            "action_verbs": cls.action_verbs,
            "decision_verbs": cls.decision_verbs,
            "descriptors": cls.descriptors
        }
    
    @classmethod
    def get_template_variables(cls):
  
        return {
            "gender": cls.genders,
            "race": cls.races,  
            "profession": cls.professions,
            "authority_figure": cls.authority_figures,
            "leader_role": cls.leader_roles,
            "trait": cls.traits,
            "adjective": cls.adjectives,
            "emotion": cls.emotions,
            "action_verb": cls.action_verbs,
            "decision_verb": cls.decision_verbs,
            "descriptor": cls.descriptors
        }