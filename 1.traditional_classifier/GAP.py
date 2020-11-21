from typing import DefaultDict, List, Optional, Iterator, Set, Tuple
import pandas as pd

# import en_core_web_lg
# nlp = en_core_web_lg.load()
import en_core_web_sm
nlp = en_core_web_sm.load()

class GAPDocument:

    def __init__(self,
                 parag_id: str, # paragraph_id
                 text: str,
                 sentence_id: [(int,int)],
                 words: List[str],  # List[(int, str)]
                 pos_tags: List[str],
                 named_entities: List[str],
                 coref_spans: Set[Tuple[int, int]],
                 coref_spans_false: Set[Tuple[int, int]],
                 pronoun: int,
                 a: int,
                 b: int) -> None:
        self.parag_id = parag_id
        self.text = text
        self.sentence_id = sentence_id
        self.words = words
        self.pos_tags = pos_tags
        self.named_entities = named_entities
        self.coref_spans = coref_spans
        self.coref_spans_false = coref_spans_false
        self.pronoun = pronoun
        self.a = a
        self.b = b

    def as_array(self):
        return [self.parag_id,
                self.text,
                self.sentence_id,
                self.words,
                self.pos_tags,
                self.named_entities,
                self.coref_spans,
                self.coref_spans_false,
                self.pronoun,
                self.a,
                self.b]

class GAP:

    def dataset_2_documents(self, filepath: str):
        ds = pd.read_csv(filepath, sep='\t')
        print(f'\n# Data: {len(ds)}')
        # tokenFrame = pd.DataFrame(columns=['parag_id', 'sentence_id', 'words', 'pos', 'tree', 'lemma'])
        gapdocuments: List[GAPDocument] = []

        for _, row in ds.iterrows():
            parag_id: str = row['ID']
            text: List[str] = row['Text']
            sentence: [(int,int)] = []
            words: List[(int,str)] = []
            pos_tags: List[str] = []
            named_entities: List[(int,int,str)] = []  # sentence_id, position, string
            coref_span: List[(int,int)] = []
            coref_span_false: List[(int, int)] = []
            pronoun: int = row['Pronoun-offset']
            a: int = row['A-offset']
            b: int = row['B-offset']

            tokens = nlp(row[1])
            en_temp = []
            # search all the named_entities in one document
            for e in tokens.ents:
                if e.label_ == 'PERSON':
                    en_temp.append((text.index(e.text), e.text))  # idx, string_of_NER

            # add the pronoun and names into the list
            en_temp.append((int(row['Pronoun-offset']), row['Pronoun']))
            en_temp.append((int(row['A-offset']), row['A']))
            en_temp.append((int(row['B-offset']), row['B']))

            try:
                # process coref_span,coref_span_false
                p_offset = int(row['Pronoun-offset'])
                a_offset = int(row['A-offset'])
                b_offset = int(row['B-offset'])

                # only one true label
                if row['A-coref'] == True:
                    coref_span.append((min(p_offset,a_offset), max(p_offset,a_offset)))
                elif row['B-coref'] == True:
                    coref_span.append((min(p_offset,b_offset), max(p_offset,b_offset)))
                # perhaps 2 false labels
                if row['A-coref'] == False:
                    coref_span_false.append((min(p_offset,a_offset), max(p_offset,a_offset)))
                if row['B-coref'] == False:
                    coref_span_false.append((min(p_offset,b_offset), max(p_offset,b_offset)))
            except ValueError:
                pass

            sentence_id = 0
            for token in tokens:
                words.append((token.idx,token.text))
                pos_tags.append(token.pos_)

                if token.is_punct:

                    sentence.append((sentence_id,token.idx))  # idx is the punct.idx
                    sentence_id += 1
                    for i1,(idx1,string1) in enumerate(list(set(en_temp))):
                        if idx1 < token.idx:
                            stored_ne = [(idx2,string2) for (sentence_id2,idx2,string2) in named_entities]
                            if (idx1,string1) not in stored_ne:
                                named_entities.append((sentence_id,idx1,string1))

            gapdocument = GAPDocument(parag_id,
                                      text,
                                      sentence,
                                      words,
                                      pos_tags,
                                      named_entities,
                                      list(set(coref_span)),  # delete duplicates
                                      list(set(coref_span_false)),
                                      pronoun,
                                      a,
                                      b)
            gapdocuments.append(gapdocument)

        return gapdocuments

if __name__ == '__main__':
    filepath = '../data/gap-validation.tsv'
    gap= GAP()
    data = gap.dataset_2_documents(filepath)
    print(len(data))
    for i in range(len(data)):
        print(data[i].as_array())

    # train = pd.read_csv('../data/gap-development.tsv', sep='\t')
    # val = pd.read_csv('../data/gap-validation.tsv', sep='\t')
    # test = pd.read_csv('../data/gap-test.tsv', sep='\t')
    # to_test = pd.read_csv('../data/test_stage_2.tsv', sep='\t')
    #
    # print(f'\n# Train: {len(train)}'
    #       f'\n# Val: {len(val)}'
    #       f'\n# Test: {len(test)}'
    #       f'\n# To Test: {len(to_test)}')
    #
    # print(train.head())

