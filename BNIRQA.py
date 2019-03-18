import math
import wikipedia
import copy
import xml.etree.ElementTree as etree
import nltk
from squad_qa import fetch_samples
# BNIRQA Model
# Ali Hinnawe
# Questions and NLP, project seminar 2018

def get_paragraphs(query, pages=1):
    """
    Extract a list of paragraphs that are relevant to query from Wikipedia using Wiki search API.
    Args:
        query : SQuAD data set query
        pages : possible number of Wikipedia pages that might be related to the query.
    Returns:
            A list that contains all the paragraphs that were extracted from the extracted articles, with their corresponding Wikipedia links.
    """
    paragraphs = []
    number_of_pages = pages
    if len(query) > 300: #limit the size of the query to 300 characters
        query = query[0:300]
    try:
        # Search for the most relevant articles to the query in Wikipedia
        page_tiltes_results = wikipedia.search(query, results=number_of_pages, suggestion=False)
    except:
        return paragraphs
    for page_title in page_tiltes_results:
        try:
            # extract the page title and content from the article
            page = wikipedia.page(page_title)
            page_content = page.content
            # extract paragraphs from articles
            for paragraph in page_content.split("=="):
                if paragraph.strip() == "See also":
                    break
                # discard paragraphs that are greater than 50 terms.
                if len(paragraph) >= 50:
                    paragraphs.append({"paragraph":paragraph, "url":page.url})
        except:
            continue

    return paragraphs
def get_answers(query, num_ans, pages=1, paragraphs_objects=None):
    """
    Get a list of answers of the query, ranked in descending order (Most probable paragraph first)

    Args:
        query : SQuAD data set query.
        num_ans : number of answers to return for a give query.
        pages : number of Wikipedia pages that might be related to the query.
        paragraphs_objects : a list of all paragraphs with their corresponding web links ; default = empty list.
    Returns:
            list of answers to the guery. The paragraphs with its probabilities.
     """
    query = query.lower().replace('?','') # lowercase all words.
    if paragraphs_objects is None:
        paragraphs_objects = get_paragraphs(query, pages)
    paragraphs = [o['paragraph'] for o in paragraphs_objects] #list of paragraphs.
    word_count = paragraphs_word_count(paragraphs) #count of all words in the paragraphs.
    total_num_word = sum([d[1] for d in word_count]) #total number of words in  the collections of paragraphs.

    # calculate probability of the words in the paragraphs
    prob_dicts = calculate_prob(word_count, total_num_word)

    #apply threshold to the pairs of words in order to restrict the dependency relation between the terms in the paragraph to the most relevant ones.
    #The accuracy did not improve when applying the threshold. That's why it is disabled.
    #apply_threshold(prob_dicts, 0.1) # apply a theshold of 0.1.

    ans_index, prob = where(query, prob_dicts, total_num_word, [x[1] for x in word_count], num_ans)
    ans = []
    for (i, j) in enumerate(ans_index):
        ans.append((paragraphs_objects[j], 10**prob[i]))
    return ans
def count_words(s):
    """
    Counts the (unigrams, bigrams and trigrams) in each paragraph. / parts 4.2 a,b,c - Parametes estimation paper of Garrouch et al
    Args:
        s : a single paragraph from the collection of paragraphs
    Returns:
            a dictionary of (unigrams, bigrams and trigrams) frequencies along with their associated frequency sum.
     """
    s = s.lower()
    s = s.replace("\n"," ")
    frequency = {}
    tokens = nltk.word_tokenize(s) #word tokenization
    # 4.2 a / paper of Garrouch et al
    # frequency of unigrams
    for word in tokens:
        count = frequency.get(word, 0)
        frequency[word] = count + 1
    # 4.2 b / paper of Garrouch et al
    # frequency of bigrams
    for word in [" ".join(pair) for pair in nltk.bigrams(tokens)]:
        count = frequency.get(word, 0)
        frequency[word] = count + 1
    # 4.2 b / paper of Garrouch et al
    # frequency of trigrams
    for word in [" ".join(pair) for pair in nltk.trigrams(tokens)]:
        count = frequency.get(word, 0)
        frequency[word] = count + 1

    return (frequency, sum(frequency.values()))
def paragraphs_word_count(ps):
    """
    Counts the (unigrams, bigrams and trigrams) in each paragraph.
    Args:
        ps : collection of paragraphs extracted from Wikipedia
    Returns:
            a list that contains all the words, pair of words and triplets with their counts.
        """
    frequency = []
    for p in ps: # for each paragraph
        frequency.append(count_words(p)) # append the counts of words in the list.
    return frequency
def where(inp,prob_dicts,total_num_word,num_word_lists, ans_len = 1) :
    '''
    Find the probability of relevance of each paragraph to the given query.  part 4.3 / Gharrouch et al. paper
    Args:
        inp : SQuAD query.
        prob_dicts : list that contains the probabilities of unigrams,bigrams and trigrams in the paragraph.
        total_num_word : total number of words in the collection of data.
        num_word_lists : list that contains the number of word in each paragraph.
        ans_len : number of answers returned to a given query.
    Retrun:
         A set that contains the indexes of paragraphs along with a set that contains the probabilities of those paragraphs.
    '''

    tokens = nltk.word_tokenize(inp) # tokenize words.
    list_word_2 = [" ".join(pair) for pair in nltk.bigrams(tokens)] #list of word pairs.
    list_word_3 = [" ".join(pair) for pair in nltk.trigrams(tokens)] #list of triplets.
    list_word = tokens + list_word_2 + list_word_3 #list for all types of terms in the paragraph.
    prob_values = [0]*len(prob_dicts) #list of calculated probability of each paragraph.

    # 4.3 Probability of a paragraph given a query
    for i in range(0,len(prob_dicts)):
        prob = math.log10((num_word_lists[i]+1)/(float(total_num_word)+num_word_lists[i])) # probability of each paragraph
        for word in list_word : # if the word in the query belongs to the paragraph.
            if word in prob_dicts[i][0]:
                prob += prob_dicts[i][0][word] #Product probability to total. add up the probability of the term in the query to the paragraph  probability.
            else: #if the word belongs only to the query
                prob += math.log10(1/(float(total_num_word)+num_word_lists[i])) #add the probability of the word that belongs only to the query to the probability of paragraph.
        prob_values[i] = prob
    ans = list(reversed(sorted(range(len(prob_values)), key=lambda k: prob_values[k]))) #list that contains the indexes of paragraphs.
    prob_sorted = list(reversed(sorted(prob_values))) #list that contains the probabilities of the paragraphs.
    return ans[0:ans_len], prob_sorted[0:ans_len]
def calculate_prob(counted_dicts,total_num_word):
    """
    Probability of unigrams, bigrams and trigrams in the paragraphs / part 4.1.a Structure learning / Dependency formula #1
    Args:
        counted_dicts : count of all words in the paragraphs
        total_num_word : total number of words in the collection of paragraphs.
    Returns:
            a dictionary of (unigrams, bigrams and trigrams) probabilities along with their associated probability sum.
    """
    counted_dicts = copy.deepcopy(counted_dicts)
    # probability of each (unigram, bigram or trigram) / part 4.1.a Structure learning / Dependency formula #1
    for counted_dict in counted_dicts :
        for key in counted_dict[0] :
            counted_dict[0][key] +=1
            counted_dict[0][key] /= (counted_dict[1]+float(total_num_word))
            counted_dict[0][key] = math.log10(counted_dict[0][key])
    return counted_dicts
def apply_threshold(counted_dicts, threshold):
    """
    A threshold imposed to remove the pair of words that has a value less that 0.1

    Args:
        counted_dicts : a dictionary that contains probabilities of words in the paragraphs
        threshold : threshold value.
    """
    threshold = math.log10(threshold)
    for (d, t) in counted_dicts:
        for key, value in d.items():
            if value < threshold:
                d[key] = 1
def print_answers(answers):
    """
    print the answer(s) paragraphs of a given query, the probability of the paragraph(s), and Wikipedia link where the answer was extracted from.

    Args:
        answers : a list of answers to the query ranked in descending order (Most probable paragraph first)
        """
    if answers is None:
        return
    for (i, a) in enumerate(answers):
        print("Answer: ", i+1)
        print("Probability: ", a[1])
        print("URL: ", a[0]['url'])
        print(a[0]['paragraph'].replace("=","").replace('\n','').replace('\t',''))
def fetch_squad_data(_sort=True):
    """
    evalation part: Check if the answer of the proposed model BNIRQA contains the answer of SQuAD data set.
    Args:
        _sort : sorting boolean
    Return:
        The accuracy of the test, the total number of answers, the number of correct answers
    """

    testset = fetch_samples() #read the SQuAD training , dev file
    if _sort:
        testset = sorted(testset,
                         key=lambda x: len(nltk.word_tokenize(x.context.idx)) # a Tuple that contains the index, paragraph, the question and its answer.
                         )
    total = 0
    right = 0
    passed = 0
    for t in testset: # for each rawsample in testset
        passed += 1
        if passed <= 3179:
            continue
        total += 1
        print(t.question)
        print(t.answer.text)
        ans = get_answers(t.question, 5) #get 5 answers for each question
        print_answers(ans)
        found = False
        for a in ans: # for each answer paragraph in the list of answers
            if found:
                break
            for t1 in t.answer.text.lower().split(" "): # for each answer
                if t1 in a[0]['paragraph'].lower(): #if answer of the model contains the answer of SQuAD
                    right += 1
                    found = True
                    break
        print(passed, total, right, float(right/total))

################################################################################################################################################

# This function is not included in the project report.
# I have tested the proposed model on few questions just to check if it gets correct answers from wikipedia for some of the TREC questions.

# The idea is to extract the list of all questions and their positive answers from TREC data set and then run the proposed model,
#to get the answers of the TREC query, and then check if the answer of the model contains the answer of TREC.

def read_TREC_data(ans=1, pages = 1):

    """
TREC has had a question answering track since 1999; in each track the task was defined such that the systems were to retrieve small snippets of text that contained an answer for open-domain,
closed-class questions (i.e., fact-based, short-answer questions that can be drawn from any domain).
Answer patterns are provided below for the TREC QA collections. The questions were mostly contributed by participants or developed by NIST assessors.
This set of questions was distributed to participants so they could train their systems.
The format for each question is a question number, the text of the question, and then one or more answer strings followed by the id of a document that supports the answer.
The dataset contains test questions, top ranked documents list, judgment set, pattern set, pattern evaluation perl script and original answers.
Most of the documents in the data set are given in the xml format.
    """
    question = ""
    positive_answers = []
    for e in etree.iterparse('TREC.xml'): # parse the file
        if e[1].tag == "question":
            question = e[1].text
            question = question.split('?')[0]
            question = question.replace('\n',' ').replace('\t',' ')
            print(question)
            positive_answers = []
        if e[1].tag == "positive":
            positive = e[1].text
            positive = positive.split('.')[0]
            positive = positive.replace('\n', ' ').replace('\t', ' ')
            positive_answers.append(positive)
        if e[1].tag == "QApairs":
            answers = get_answers(question, ans, pages)
            print_answers(answers)

################################################################################################################################################


def main():
    fetch_squad_data()


if __name__ == '__main__':
    main()
