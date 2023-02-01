import conllu
from typing import List, Dict


def txt_to_conll(text: str, nlp):
    """Input:
    - text: the string we want to parse
    - nlp: stanza parser (initalized in the cell above)

    Output:
    - the dependency trees for each setnence in text,
      concatenated in a .conll format"""

    # text clean-up: we need to eliminate all \n and \t and not have more than one ' ' in a row anywhere
    # we do this by using string.split() method which splits by " ", \n and \t and concatenate all the pieces
    # excess spaces result in a wrong .conll that is undreadable afterwards
    text = " ".join(text.split())

    doc = nlp(text)
    return doc._.conll_str


def stanza_to_bert_tokens(sentence: conllu.models.TokenList, bert_tokenization, tokenizer):
    """
    The function finds the ranges of tokens in tokenizer's tokenization that correspond to stanza's tokenization

    Input:
    - sentence: the sentence parsed using stanza (will be of the form: TokenList<I, am, home, metadata={sent_id: "1", text: "I am home"}>)
    - bert_tokenization: the output of tokenizer(original-sentence-string)['input_ids']
    - tokenizer: the tokenizer that we will use for getting context tensors
    Output:
    - mapping: the map between stanza tok
    enization and model tokenization
    """

    bert_tokens = tokenizer.convert_ids_to_tokens(bert_tokenization)
    token_map: List[tuple(int, int)] = []  # the tuples mapping (left-to-right) the stanza tokens
    # to the range where they are in the RoBERTa tokenization

    i = 1  # to walk through bert_tokens. Starts at 1 because bert_tokens[0] is the bos token
    j = 0  # to walk through the RoBERTa token character by character.
    # This will help if words are weirdly cut and glued together

    for token in sentence:  # this will loop through all stanza tokens
        token = token['form']
        start = i
        while len(token) > 0:

            if bert_tokens[i][j] == "Ġ":  # this signifies the start of a word in RoBERTa in the pre-tokenized sentence
                j += 1

            if len(token) == len(bert_tokens[i][j:]):  # RoBERTa token corresponds exactly to the stanza token
                token_map.append((start, i + 1))
                i += 1
                j = 0
                break  # breaks the while

            elif len(token) < len(bert_tokens[i][j:]):
                token_map.append((start, i + 1))
                j += len(token)
                break

            else:  # so len(token) > len(bert_tokens[i][j:])
                token = token[len(bert_tokens[i][j:]):]
                i += 1
                j = 0

    return token_map


def get_contextual_embeddings(path: str, tokenizer, model):
    """
    Input:
    - path: the location of the .conll file containing dependency trees for each sentence of the text we are analysing
    - tokenizer, model: the tokenizer and the model we will use for contextual representation

    Output:
    - a dictionary mapping each verb lemma to its v- and v+ representations (look into the code for a more detailed explanation)
    """

    # we find negation cues by depth-first search through the tree
    # jump below the function definition
    def depth_search(root: conllu.models.TokenTree, current_verb: str, current_index: int):
        """Input:
        - root: the (sub)tree in which we are looking for negation
        - current_verb: if we encounter negation, this is the lemma of the verb that is negated.
                        if another verb is encountered, the function is recursively called with the new current_verb
        - current_index: same as above
        Passing both current_verb and current_index to omit the search for the index's lemma in the tree.
        We need the index to be able to localize the tokens of the verb in RoBERTa tokenization and the lemma to be able to fill in verb_embeddings"""
        nonlocal representations  # will be initialized for each sentence; the RoBERTa encoding
        nonlocal negation_found  # will be initialized for each sentence; dictionary that tells us
        # whether negation was found for each verb lemma (for each auxilary)
        nonlocal sentence  # this is the linearized tree (sentence in the first for loop -
        # we have "for sentence in dependency_trees")
        # only needed for localizing "no more"

        if root.token['upos'] == "VERB":  # the function is called for all children but current_verb
            # and current_index change
            negation_found[root.token['id']] = [0, 0]  # a verb is found but the negation not yet -
            # the auxilaries also haven't because we are in a tree structure
            for child in root.children:
                depth_search(child, root.token['lemma'], root.token['id'])

        else:  # we haven't found negation and root is not a verb
            # we iterate through the root's children, not changing the other parameters
            if root.token['upos'] == "AUX":  # we have found an auxiliary of the current root
                if current_index in negation_found:  # it is possible for the auxiliary to be tied
                    # to another auxiliary (like "won't" to "can")
                    # in the sentence "I can but won't do this"
                    negation_found[current_index][0] += 1

            # we check whether we have found negation:
            elif root.token['lemma'] == 'not' or root.token['lemma'] == 'never' or (
                    root.token['lemma'] == 'more' and root.token['id'] > 1 and sentence[root.token['id'] - 2]['lemma'] == 'no'):
                if current_index in negation_found:  # it is possible for the head to be something other than a verb,
                    # for example in the sentence "Martin will have no more apple sauce"
                    # where the head of negation is "sauce" - in this case we will ignore it
                    negation_found[current_index][1] += 1  # a negation found for the current verb

            # iterate though all the children (there is probably no children here)
            for child in root.children:
                depth_search(child, current_verb, current_index)

    # reading the file from path
    file = open(path)
    dependency_trees = conllu.parse_incr(file)

    # verb_embeddings is a map: lemma of the verb v -> [context_reprs_negated, context_reprs_affirmative]

    # we have List[List[torch.Tensor]] since it is possible that some verbs be split into multiple tokens in RoBERTa
    verb_embeddings: Dict[str, [List[List[torch.Tensor]], List[List[torch.Tensor]]]] = {}

    for sentence in dependency_trees:

        sentence_tree = sentence.to_tree()

        # tokenizing and encoding of the original sentence using RoBERTa
        bert_tokens = tokenizer(sentence_tree.metadata['text'], return_tensors='pt',
                                max_length=512, padding=True, truncation=True)['input_ids']
        representations = model(bert_tokens, output_attentions=True, output_hidden_states=True,
                                return_dict=True)

        # getting the stanza to RoBERTa token map
        token_mapping = stanza_to_bert_tokens(sentence, tokenizer(sentence_tree.metadata['text'])['input_ids'],
                                              tokenizer)

        negation_found: Dict[int, [int, int]] = {}  # maps the index of a verb to  a tuple (num_aux, num_negations) -
        # the number of auxiliaries and the number of negations of the verb)

        # depth first search from the tree: see function above
        depth_search(sentence_tree, sentence_tree.token['lemma'], sentence_tree.token['id'])

        # current_verbs are now filled
        for index in negation_found:
            lemma = sentence[index - 1]['lemma']
            if lemma not in verb_embeddings:
                verb_embeddings[lemma] = [[], []]

            start, end = token_mapping[index - 1]  # localizing the verb in the RoBERTa tokenization

            if negation_found[index][1] == 0:  # negation wasn't found for the verb at position index
                verb_embeddings[lemma][1].append(representations.last_hindexden_state[0, start:end, :])
            elif negation_found[index][1] >= negation_found[index][0]:  # the number of negations is
                # bigger than or equal to the number of auxiliaries
                verb_embeddings[lemma][0].append(representations.last_hidden_state[0, start:end, :])
            else:  # then negations were found but not for every auxiliary, thus we add the tensors to both sides
                verb_embeddings[lemma][0].append(representations.last_hidden_state[0, start:end, :])
                verb_embeddings[lemma][1].append(representations.last_hidden_state[0, start:end, :])

    # we have exited the first loop, everything we need is in verb_embeddings
    return verb_embeddings


def wiki_negated_clause_stats(path: str):
    """
    Input:
    - path: the location of the .conll file containing dependency trees for each sentence of the text we are analysing

    Output:
    - (number_of_phrases_with_negation, number_of_simple_phrases_with_negation)
    """

    # we find negation cues by depth-first search through the tree
    # jump below the function definition

    # reading the file from path
    file = open(path)
    dependency_trees = conllu.parse_incr(file)

    phrases_with_negation = 0
    negations_in_dependent_clauses = 0

    for sentence in dependency_trees:
        for token in sentence:
            if token['lemma'] == 'not' or token['lemma'] == 'never' or (token['lemma'] == 'more' and token['id'] > 1 and sentence[token['id'] - 2]['lemma'] == 'no'):
                token_head = sentence[token['head'] - 1]
                if token_head['upos'] == "VERB":
                    phrases_with_negation += 1
                    if token_head['head'] != 0:  # the root is negated
                        negations_in_dependent_clauses += 1

    return phrases_with_negation, negations_in_dependent_clauses


def wiki_parsing(wikipages, filepath):
    """
    Input:
    - pages: dataset from Huggingface
    - filepath: the path to the .conll file where we will write the dependency trees after
      parsing wiki pages"""
    with open(filepath, "a") as file:
        for page in wikipages:
            file.write(txt_to_conll(page))

