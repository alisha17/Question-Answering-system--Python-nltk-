# This program was written as the final assignment for
# “Introduction to Natural Language Processing” (CMPS_143) class
# at University of California Santa Cruz
# by Jeremy Chien, Ethan Seither, and Ekaterina Tcareva

#*************************************************************************
#to run:
#python3 q_a_new_chunk.py hw8_process_stories.txt

#to test using testing file:
#perl score-answers.pl chien_seither_tcareva_answers.txt hw8_dev.answers
#*************************************************************************

#FINAL RESULTS

#AVERAGE RECALL =    0.7094  (165.29 / 233)
#AVERAGE PRECISION = 0.6760  (154.80 / 229)
#AVERAGE F-MEASURE = 0.6923
#*************************************************************************

import re, nltk
from nltk.stem import SnowballStemmer
import sys, operator
from nltk.stem.wordnet import WordNetLemmatizer
import csv
from collections import defaultdict
from nltk.corpus import wordnet as wn
from operator import itemgetter
import string

file_name = ""
def load_wordnet_ids(filename):
    file = open(filename, 'r')
    if "noun" in filename: type = "noun"
    else: type = "verb"
    csvreader = csv.DictReader(file, delimiter=",", quotechar='"')
    word_ids = defaultdict()
    for line in csvreader:
        stories = line['stories']
        synset_id = line['synset_id']
        word_type = 'story_'+type
        story_word = line[word_type]
        synset_offset = line['synset_offset']
        for i in stories.split(', '):
            i=i.replace('\'', '')
            i=i.replace('.vgl', '')
            i=i.replace('{','')
            i=i.replace('}','')
            if not word_ids.get(i):
                word_ids[i] = defaultdict()
            word_ids[i][story_word] = {'synset_offset': synset_offset, 'synset_id': synset_id}
    return word_ids
###############################################################################
## Utility Functions ##########################################################
###############################################################################

noun_ids = load_wordnet_ids("Wordnet_nouns.csv")
verb_ids = load_wordnet_ids("Wordnet_verbs.csv")

TAG_WORD_SET = set(["VBD", "VBG", "VBN", "VB", "JJ", "RB", "NN", "NNS", "NNP"])
LEMMATIZED_SET = set(["VBD", "VBG", "VBN", "VB"])
# Our simple grammar from class (and the book)
GRAMMAR =   """
            N: {<PRP>|<NN.*>}
            V: {<V.*>}
            ADJ: {<JJ.*>}
            NP: {<DT>? <ADJ>* <N>+}
            PP: {<IN> <NP>}
            VP: {<TO>? <V> (<NP>|<PP>)*}
            """

LOC_PP = set(["in", "on", "at","near", "along", "under", "into", "upon"])

WHEN_PP = set(["on", "year", "ago", "next", "last", "before", "after", "previous"])
filler_verbs = ["was", "is"]

WHY_PP = set(["since", "because", "in order to", "due to", "for it"])

vp_signals = []
snowball = nltk.stem.SnowballStemmer("english")

passive_voice_signals = ['did', 'be', 'will', 'being']

def pp_filter(subtree):
    return subtree.label() == "PP"
    
def n_filter(subtree):
    return subtree.label() == "NP"

def is_location(prep):
    return prep[0] in LOC_PP
    
def is_time(prep):
    return prep[0] in WHEN_PP

def get_try_order(rel):
    if(rel == 'nsubj'):
        return ['nsubj', 'nmod', 'dobj']
    if(rel == 'dobj'):
        return ['dobj', 'nmod', 'nsubj']
    return None

def map_to_wn_tag(tag):
    if tag.startswith("N"):
        return "n"
    if tag.startswith("R"):
        return "r"
    if tag.startswith("J"):
        return "a"
    if tag.startswith("V"):
        return "v"

def find_locations(tree):
    # Starting at the root of the tree
    # Traverse each node and get the subtree underneath it
    # Filter out any subtrees who's label is not a PP
    # Then check to see if the first child (it must be a preposition) is in
    # our set of locative markers
    # If it is then add it to our list of candidate locations
    # How do we modify this to return only the NP: add [1] to subtree!
    # How can we make this function more robust?
    # Make sure the crow/subj is to the left
    locations = []
    for subtree in tree.subtrees(filter=pp_filter):
        if is_location(subtree[0]):
            locations.append(subtree)
    
    return locations

def find_time(tree):
    time = []
    for subtree in tree.subtrees(filter=pp_filter):
        if is_time(subtree[0]):
            time.append(subtree)
    
    return time
    
def find_noun(tree):
    n = []
    for subtree in tree.subtrees(filter=n_filter):
        n.append(subtree)
    return n

def detect_stype(dep_parse):
    """
    :param sentence:
    :return:
    helps learn what kind of thing the dependency parser should be answering about
    """
    word = "What"
    i = find_node(word, dep_parse)
    if i is None:
        return None
    return (i['rel'], dep_parse.nodes[i['head']]['word'])


#for "Where" questions
def find_candidates(sentences, chunker):
    candidates = []
    for sent in sentences:
        tree = chunker.parse(sent)
        locations = find_locations(tree)
        candidates.extend(locations)
    return candidates

#for "When" questions
def find_candidates2(sentences, chunker):
    candidates = []
    for sent in sentences:
        tree = chunker.parse(sent)
        locations = find_time(tree)
        candidates.extend(locations)
    return candidates

#for "Who" questions
def find_candidates3(sentences, chunker):
    candidates = []
    for sent in sentences:
        tree = chunker.parse(sent)
        n = find_noun(tree)
        candidates.extend(n)
    return candidates
        
# returns a dictionary where the question numbers are the key
# and its items are another dict of difficulty, question, type, and answer
# e.g. story_dict = {'fables-01-1': {'Difficulty': x, 'Question': y, 'Type':}, 'fables-01-2': {...}, ...}
def getQA(filename):
    content = open(filename, 'rU', encoding='latin1').read()
    question_dict = {}
    for m in re.finditer(r"QuestionID:\s*(?P<id>.*)\nQuestion:\s*(?P<ques>.*)\n(Answer:\s*(?P<answ>.*)\n){0,1}Difficulty:\s*(?P<diff>.*)\nType:\s*(?P<type>.*)\n+", content):
        qid = m.group("id")
        question_dict[qid] = {}
        question_dict[qid]['Question'] = m.group("ques")
        question_dict[qid]['Answer'] = m.group("answ")
        question_dict[qid]['Difficulty'] = m.group("diff")
        question_dict[qid]['Type'] = m.group("type")
    return question_dict

def get_data_dict(fname):
    data_dict = {}
    data_types = ["story", "sch"]
    parser_types = ["par", "dep"]
    for dt in data_types:
        data_dict[dt] = read_file(fname + "." + dt)
        for tp in parser_types:
            data_dict['{}.{}'.format(dt, tp)] = read_file(fname + "." + dt + "." + tp)
    return data_dict

# Read the file from disk
# filename can be fables-01.story, fables-01.sch, fables-01-.story.dep, fables-01.story.par
def read_file(filename):
    fh = open(filename, 'r')
    text = fh.read()
    fh.close()   
    return text

###############################################################################
## Question Answering Functions
###############################################################################

# Read the dependency parses from a file
def read_dep_parses(depfile):
    fh = open(depfile, 'r')

    # list to store the results
    graphs = []

    # Read the lines containing the first parse.
    dep = read_dep(fh)

    # While there are more lines:
    # 1) create the DependencyGraph
    # 2) add it to our list
    # 3) try again until we're done
    while dep is not None:
        test='\n'.join(dep[0].split('\n'))
        graph = nltk.parse.DependencyGraph(str(test))
        graphs.append((graph,dep[1]))
        dep = read_dep(fh)
    fh.close()

    graphs.sort(key=lambda pair: pair[1])

    return [graph[0] for graph in graphs]

def update_inconsistent_tags(old):
    return old.replace("root", "ROOT")

# Read the lines of an individual dependency parse
def read_dep(fh):
    dep_lines = []
    num = 0
    for line in fh:
        line = line.strip()
        if len(line) == 0:
            return (update_inconsistent_tags("\n".join(dep_lines)),num)
        elif re.match(r"^QuestionId:\s+(.*)$", line):
            num = int(re.findall(r"-(\d+)$", line)[0])

            continue
        dep_lines.append(line)
    if len(dep_lines) > 0:
        return (update_inconsistent_tags("\n".join(dep_lines)),num)
    else:
        return None

def find_answer(sgraph, rel, word, wn_tag):
    snode = find_node(word, sgraph, wn_tag)

    if snode:
        for node in sgraph.nodes.values():
            if node.get('head', None) == snode["address"]:
                if node['rel'] == rel:
                    deps = get_dependents(node, sgraph)
                    deps = [(node2["word"], node2["address"]) for node2 in deps]
                    deps.append((node['word'], node['address']))
                    deps.sort(key= lambda pair: pair[1])
                    return ' '.join([pair[0] for pair in deps])

    # if conj then look at the conj
        if snode['rel'] == 'conj':
            snode = sgraph.nodes[snode['head']]
            for node in sgraph.nodes.values():
                if node.get('head', None) == snode["address"]:
                    if node['rel'] == rel:
                        deps = get_dependents(node, sgraph)
                        deps = [(node2["word"], node2["address"]) for node2 in deps]
                        deps.append((node['word'], node['address']))
                        deps.sort(key= lambda pair: pair[1])
                        return ' '.join([pair[0] for pair in deps])
    return None

def find_node(word, graph, wordnettag=''):
    if word:
        lemmword = ''
        if wordnettag is not '' and wordnettag is not None:
            lemmword = WordNetLemmatizer().lemmatize(word, wordnettag)
        snowlem = snowball.stem(word)
        for node in graph.nodes.values():
            if node["word"]:

                lemmoword = ''
                if wordnettag is not '' and wordnettag is not None:
                    lemmoword = WordNetLemmatizer().lemmatize(node["word"], wordnettag)
                snowlemmo = snowball.stem(node["word"])

                if(noun_ids[file_name].get(node["word"])):
                    synset = noun_ids[file_name][node["word"]]['synset_id']
                    if check_similarity(word, synset):
                        return node

                if(verb_ids[file_name].get(node["word"])):
                    synset = verb_ids[file_name][node["word"]]['synset_id']
                    if check_similarity(word, synset):
                        return node

                if node["word"] == word or lemmoword==word \
                        or (lemmword == lemmoword and lemmword is not '') or lemmword == node["word"]\
                        or snowlem == snowlemmo:
                    return node
    return None

def check_similarity(word, synword):
    synset = wn.synset(synword)
    if word in synset.lemma_names():
        return True;

    check_syns = wn.synsets(word)
    closet = (-1,synset)
    for syncheck in check_syns:
        simil = syncheck.path_similarity(synset)
        if(simil is not None):
            if simil>.6:
                return True
    return False

def get_dependents(node, graph):
    results = []
    for item in node["deps"]:
        address = node["deps"][item][0]
        dep = graph.nodes[address]
        results.append(dep)
        results = results + get_dependents(dep, graph)

    return results
# The standard NLTK pipeline for POS tagging a document
def get_sentences(text):
    sentences = nltk.sent_tokenize(text)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]
    return sentences	

def get_bow(tokens, stopwords):
    return set([t[0].lower() for t in tokens if t[0].lower() not in stopwords])
    
def get_bow2(tokens):
    return set([t[0].lower() for t in tokens])

def baseline(qbow, sentences, stopwords, dep_sent):
    # Collect all the candidate answers
    answers = []
    for sent, deps in zip(sentences, dep_sent):
        sbow = get_bow2(sent)
        lemmas = []
        for a in sbow:
        	lemmas.append(snowball_stemmer.stem(a))
        sbow.update(lemmas)
        
        # Count the # of overlapping words between the Q and the A
        # & is the set intersection operator
        overlap = len(qbow & sbow)
        answers.append((overlap, sent, deps))
        
    # Sort the results by the first element of the tuple (i.e., the count)
    # Sort answers from smallest to largest by default, so reverse it
    answers = sorted(answers, key=operator.itemgetter(0), reverse=True)

    # Return the best answer
    best_answer = (answers[0])[1]    
    return answers



#*************************************************************************
#the entrance to the programm
#*************************************************************************
if __name__ == '__main__':

    stopwords = set(nltk.corpus.stopwords.words("english"))
    snowball_stemmer = SnowballStemmer("english")
    chunker = nltk.RegexpParser(GRAMMAR)
    lmtzr = WordNetLemmatizer()

    pyfile = open(sys.argv[1])
    text = pyfile.read()

    # Loop over the files in fables and blogs in order.
    output_file = open("chien_seither_tcareva_answers.txt", "w", encoding="utf-8")

    for fname in text.split('\n'):
        if fname != '':
            data_dict = get_data_dict(fname)
            global file_name
            file_name = fname

            questions = getQA("{}.questions".format(fname))
            depquestions = read_dep_parses("{}.questions.dep".format(fname))
            for j in range(0, len(questions)):
                qname = "{0}-{1}".format(fname, j+1)
                if qname in questions:
                    print("QuestionID: " + qname)
                    question = questions[qname]['Question']
                    print(question)
                    qtypes = questions[qname]['Type']

                    answer = None
                    # qtypes can be "Story", "Sch", "Sch | Story"
                    answers = []
                    for qt in qtypes.split("|"):
                        qt = qt.strip().lower()
                        # These are the text data where you can look for answers.
                        raw_text = data_dict[qt]
                        par_text = data_dict[qt + ".par"]
                        dep_text = data_dict[qt + ".dep"]
                        qbow = get_bow(get_sentences(question)[0], stopwords)
                        sentences = get_sentences(raw_text)
                        lemmas = []
                        for a in qbow:
                            lemmas.append(snowball_stemmer.stem(a))
                        qbow.update(lemmas)
                        if "Where" in question:
                            qbow.update(LOC_PP)
                        if "When" in question:
                            qbow.update(WHEN_PP)
                        if "Why" in question:
                            qbow.update(WHY_PP)
                        ans = baseline(qbow, sentences, stopwords, dep_text.split('\n\n'))
                        answers.append(ans[0][1])
                        answer = answers[0]
                    agraph = nltk.parse.DependencyGraph(ans[0][2])
                    if ("When" in question) or ("Where" in question) or ("Who" in question) or "What" in question \
                        or "Why" in question:
                        locations = []
                        if "What" in question:
                            q = depquestions[j]
                            rel = detect_stype(q)
                            if rel is not None:
                                #grab part of speech where it's the same
                                wn_tag = ''
                                if rel[0] != 'ROOT':
                                    qtg = get_sentences(question)[0]
                                    try:
                                        qpos = next(pair[1] for pair in qtg if pair[0]==rel[1])
                                    except StopIteration:
                                        words = [node["word"] for node in depquestions[j].nodes.values()]
                                        words.remove(None)
                                        tgs = nltk.pos_tag(words)
                                        qpos = next(pair[1] for pair in
                                                    nltk.pos_tag(words)
                                                    if pair[0]==rel[1])
                                    wn_tag = map_to_wn_tag(qpos)
                            if rel is not None:
                                tst = find_answer(agraph,rel[0], rel[1], wn_tag)
                                if tst is not None:
                                    locations.append(tst)

                            lmtzr = WordNetLemmatizer()

                            question_sep = nltk.word_tokenize(question)
                            question_sep = nltk.pos_tag(question_sep)

                            q_words = []
                            for (word, pos) in question_sep:
                                if pos in TAG_WORD_SET:
                                    q_words.append((word, pos))

                            if len(locations) < 1:
                                q_lemma_words = []
                                for (word, pos) in q_words:
                                    if pos in LEMMATIZED_SET:
                                        q_lemma_words.append(lmtzr.lemmatize(word, "v"))
                                    else:
                                        q_lemma_words.append(word)

                                ans_lemma_words = []
                                for (word, pos) in answers[0]:
                                    if pos in LEMMATIZED_SET:
                                        ans_lemma_words.append(lmtzr.lemmatize(word, "v"))
                                    else:
                                        ans_lemma_words.append(word)

                                found = False
                                answer_list = []
                                q_lemma_words = nltk.pos_tag(q_lemma_words)
                                q_lemma_words.sort(key=itemgetter(1), reverse=True)

                                if len(q_lemma_words) > 0:
                                    for (qword, pos) in q_lemma_words:
                                        i = 0
                                        if not found:
                                            for aword in ans_lemma_words:
                                                if found:
                                                    answer_list.append(answers[0][i][0])
                                                elif qword == aword:
                                                    found = True
                                                i = i + 1

                                if len(answer_list) > 0:
                                    answer_list2 = [''.join(c for c in s if c not in string.punctuation) for s in
                                                    answer_list]
                                    answer_list2 = [s for s in answer_list2 if s]
                                if len(answer_list2) > 0:
                                    answer_final = " ".join(answer_list)
                                else:
                                    answer_final = answers[0]

                                answer = answer_final

                        elif "Why" in question:
                            qt="why"
                            found = False
                            locations = []
                            for ansindex in range(0, len(ans)):
                                ans_str = " ".join([pair[0] for pair in ans[ansindex][1]])

                                for signal in WHY_PP:
                                    if signal in ans_str.lower():
                                        found = True
                                        index = ans_str.lower().index(signal)
                                        answer = ans_str[index:]
                                        locations.append(answer)
                                        break
                                if found == True:
                                    break

                        elif "When" in question:
                            qt="when"
                            locations = find_candidates2(answers, chunker)
                        elif "Who" in question:
                            qt="what"
                            locations = find_candidates3(answers, chunker)
                        elif "Where" in question: #Where
                            qt="prep"
                            locations = find_candidates(answers, chunker)

                        if len(locations) > 0:
                            answer = locations[0]
                        print("Answer: ", end = "")
                        if answer != None:
                            if isinstance(answer, str):
                                print(answer)
                            elif isinstance(answer, nltk.tree.Tree):
                                print(" ".join([token[0] for token in answer.leaves()]))
                            else:
                                print(" ".join([token[0] for token in answer]))
                        else:
                            print ("None")
                        print("")

                        # Save your results in output file.
                        if answer != None:
                            if isinstance(answer, str):
                                an = answer
                            elif isinstance(answer, nltk.tree.Tree):
                                an = " ".join([token[0] for token in answer.leaves()])
                            else:
                                an = " ".join([token[0] for token in answer])
                        else:
                            an = ""
                        output_file.write("QuestionID: {}\n".format(qname))
                        output_file.write("Answer: {}\n\n".format(an))
                    else:
                        answer = answers[0]
                        print("Answer: ", end = "")
                        print(" ".join(t[0] for t in answer))
                        print("")

                        # Save your results in output file.
                        an = " ".join(t[0] for t in answer)
                        output_file.write("QuestionID: {}\n".format(qname))
                        output_file.write("Answer: {}\n\n".format(an))

    output_file.close()