# Objects of the MentionPair class represent the features for an extracted mention-pair.

class MentionPair:

    def __init__(self, feature_tuple):
        self.mention_i_str: str = feature_tuple[0]
        self.mention_j_str: str = feature_tuple[1]
        self.distance: int = feature_tuple[2]
        self.semantic_class_agreement = feature_tuple[3] # not very useful
        self.gender_agreement = feature_tuple[4]
        self.coreferent = feature_tuple[5]

    def __len__(self):
        return len(self.mention_i_str)

    def is_coreferent(self):
        return self.coreferent

    def as_tab_seperated_line(self):
        return "{}\t{}\t{}\t{}\t{}\t{}\t" \
                    .format(
                    self.mention_i_str,
                    self.mention_j_str,
                    self.distance,
                    '+' if self.semantic_class_agreement == True else '-' if self.semantic_class_agreement == False else 'unknown',
                    '+' if self.gender_agreement == True else '-' if self.gender_agreement == False else 'unknown',
                    '+' if self.coreferent == True else '-' if self.coreferent == False else 'unknown')

    def as_array(self):
        return [self.mention_i_str,
                self.mention_j_str,
                self.distance,
                '+' if self.semantic_class_agreement == True else '-' if self.semantic_class_agreement == False else 'unknown',
                '+' if self.gender_agreement == True else '-' if self.gender_agreement == False else 'unknown',
                '+' if self.coreferent == True else '-' if self.coreferent == False else 'unknown']

