# import dependencies
import os
import re
import sys
import nltk
from nltk.stem import WordNetLemmatizer
from collections import defaultdict

nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

def _compute_m(input):
        """
        _comput_m - function to compute the measure m of a stem based on the alternate vowel-consonane sequences
        Inputs:
                - input : str
                        Words to be stemmed
        Outputs:
                - input : str
                        Stemmed words
        """
        # initialize vowel-consonant matching pattern
        pattern = r'[aeiouy][qwrtpsdfghjklzxcvbnm]'

        # count frequency of vowel-consonant sequences
        m = len(re.findall(pattern = pattern, string = input))

        return m

def _compute_m(input):
        """
        _comput_m - function to compute the measure m of a stem based on the alternate vowel-consonane sequences
        Inputs:
                - input : str
                        Words to be stemmed
        Outputs:
                - input : str
                        Stemmed words
        """
        # initialize vowel-consonant matching pattern
        pattern = r'[aeiouy][qwrtpsdfghjklzxcvbnm]'

        # count frequency of vowel-consonant sequences
        m = len(re.findall(pattern = pattern, string = input))

        return m

def _remove_plural(input):
        """
        _remove_plural - function to remove plural forms
        Inputs:
                - input : str
        Outputs:
                - input : str
        """
        # ending w/ 'es'
        if input.endswith('es'):
                return input[:-2]
        # ending w/ ss -> skip
        elif input.endswith('ss'):
                pass
        # ending w/ s only
        elif input.endswith('s'):
                return input[:-1]
        return None

def _remove_ing_n_ed(input):
        """
        _remove_ing_n_ed - function to remove -ing and -ed forms
        Inputs:
                - input : str
        Outputs:
                - input : str
        """

        vowel_pattern = r'[aeiouy]'
        cvc_pattern = r'[a-z]*[b-df-hj-np-tv-z][aeiouy][b-df-hj-np-tvz]'

        # (m>0)eed -> ee
        if input.endswith('eed') and _compute_m(input[:-3]) > 0:
                return input[:-1]
        # (*v*)ed -> null
        elif input.endswith('ed') and re.search(vowel_pattern, input[:-2]):
                input = input[:-2]
        # (*v*)ing -> null
        elif input.endswith('ing') and re.search(vowel_pattern, input[:-3]):
                input = input[:-3]

        # special cases
        # if ends with at or bl or iz
        if input.endswith('at') or input.endswith('bl') or input.endswith('iz'):
                return input + 'e'
        #end with consonant-vowel-consonant sequence
        elif _compute_m(input[:-3]) == 1 and re.search(cvc_pattern, input):
                return input + 'e'
        # special case of double consonant
        elif len(input) > 2 and input[-1] == input[-2]:
                # if double-consonant is l, s, or z
                return input if re.search(pattern = r'[lsz]', string = input[-1]) else input[:-1]

        return None

def _substitue_y(input):
        """
        _substitue_y - function to replace y with i
        Inputs:
                - input : str
        Outputs:
                - input : str
        """
        vowel_pattern = r'[aeiouy]'

        # if ending w/ y and has a vowel in stem
        if input.endswith('y') and re.search(vowel_pattern, input[:-1]):
                return input[:-1] + 'i'

        return input

def _step1(input):
        """
        _step1 - function to apply step1 rules
        Inputs:
                - input : str
                - m : int
                        Measurement m of c.v.c. sequences
        Outputs:
                - input : str
        """

        # remove plural forms
        temp = _remove_plural(input)
        if temp:
                return temp

        # remove -ed and -ing forms
        temp = _remove_ing_n_ed(input)
        if temp:
                return temp

        # substitue y w/ i
        temp = _substitue_y(input)
        if temp:
                return temp
        return input

def _step2(input):
        """
        _step2 - function to apply step2 rules
        Inputs:
                - input : str
                - m : int
                        Measurement m of c.v.c. sequences
        Outputs:
                - input : str
        """
        # ational -> ate
        if input.endswith('ational') and _compute_m(input[:-7]) > 0:
                return input[:-1 * len('ational')] + 'ate'
        # tional -> tion
        elif input.endswith('tional') and _compute_m(input[:-6]) > 0:
                return input[:-1*len('tional')] + 'tion'
        # enci -> ence
        elif input.endswith('enci') and _compute_m(input[:-4]) > 0:
                return input[:-1] + 'e'
        # anci -> ance
        elif input.endswith('anci') and _compute_m(input[:-4]) > 0:
                return input[:-1] + 'e'
        # izer -> ize
        elif input.endswith('izer') and _compute_m(input[:-4]) > 0:
                return input[:-1]
        # abli -> able
        elif input.endswith('abli') and _compute_m(input[:-4]) > 0:
                return input[:-1] + 'e'
        # alli -> al
        elif input.endswith('alli') and _compute_m(input[:-4]) > 0:
                return input[:-2]
        # entli -> ent
        elif input.endswith('entli'):
                return input[:-2]
        # eli -> e
        elif input.endswith('eli') and _compute_m(input[:-3]) > 0:
                return input[:-2]
        # ousli -> ous
        elif input.endswith('ousli') and _compute_m(input[:-5]) > 0:
                return input[:-2] + 's'
        # ization -> ize
        elif input.endswith('ization') and _compute_m(input[:-7]) > 0:
                return input[:-5] + 'e'
        # ation -> ate
        elif input.endswith('ation') and _compute_m(input[:-5]) > 0:
                return input[:-3] + 'e'
        # ator -> ate
        elif input.endswith('ator') and _compute_m(input[:-5]) > 0:
                return input[:-2] + 'e'
        # alism -> al
        elif input.endswith('alism') and _compute_m(input[:-5]) > 0:
                return input[:-3]
        # iveness -> ive
        elif input.endswith('iveness') and _compute_m(input[:-7]) > 0:
                return input[:-4]
        # fulness -> ful
        elif input.endswith('fulness') and _compute_m(input[:-7]) > 0:
                return input[:-4]
        # ousness -> ous
        elif input.endswith('ousness') and _compute_m(input[:-7]) > 0:
                return input[:-4]
        # aliti -> ali
        elif input.endswith('aliti') and _compute_m(input[:-5]) > 0:
                return input[:-3]
        # iviti -> ive
        elif input.endswith('iviti') and _compute_m(input[:-5]) > 0:
                return input[:-3] + 'e'
        # biliti -> ble
        elif input.endswith('biliti') and _compute_m(input[:-6]) > 0:
                return input[:-5] + 'le'
        return input

def _step3(input):
        """
        _step3 - function to apply step2 rules
        Inputs:
                - input : str
        Outputs:
                - input : str
        """
        # icate -> ic
        if input.endswith('icate') and _compute_m(input[:-5]) > 0:
                return input[:-3]
        # ative -> null
        elif input.endswith('ative') and _compute_m(input[:-5]) > 0:
                return input[:-5]
        # alize -> al
        elif input.endswith('alize') and _compute_m(input[:-5]) > 0:
                return input[:-3]
        # iciti -> ic
        elif input.endswith('iciti') and _compute_m(input[:-5]) > 0:
                return input[:-3]
        # ical -> ic
        elif input.endswith('ical') and _compute_m(input[:-4]) > 0:
                return input[:-2]
        # ful -> null
        elif input.endswith('ful') and _compute_m(input[:-3]) > 0:
                return input[:-3]
        # ness -> null
        elif input.endswith('ness') and _compute_m(input[:-4]) > 0:
                return input[:-4]

        return input

def _step4(input):
        """
        _step4 - function to apply step4 rules
        Inputs:
                - input : str
        Outputs:
                - input : str
        """
        # al -> null
        if input.endswith('al') and _compute_m(input[:-2]) > 0:
                return input[:-2]
        # ance -> null
        elif input.endswith('ance') and _compute_m(input[:-4]) > 0:
                return input[:-4]
        # ence -> null
        elif input.endswith('ence') and _compute_m(input[:-4]) > 0:
                return input[:-4]
        # er -> null
        elif input.endswith('er') and _compute_m(input[:-2]) > 0:
                return input[:-2]
        # ic -> null
        elif input.endswith('ic') and _compute_m(input[:-2]) > 0:
                return input[:-2]
        # able -> null
        elif input.endswith('able') and _compute_m(input[:-4]) > 0:
                return input[:-4]
        # ible -> null
        elif input.endswith('ible') and _compute_m(input[:-4]) > 0:
                return input[:-4]
        # ant -> null
        elif input.endswith('ant') and _compute_m(input[:-3]) > 0:
                return input[:-3]
        # ement -> null
        elif input.endswith('ement') and _compute_m(input[:-5]) > 0:
                return input[:-5]
        # ment -> null
        elif input.endswith('ment') and _compute_m(input[:-4]) > 0:
                return input[:-4]
        # ent -> null
        elif input.endswith('ent') and _compute_m(input[:-3]) > 0:
                return input[:-3]
        # sion/tion -> null
        elif (input.endswith('sion') or input.endswith('tion')) and _compute_m(input[:-4]) > 0:
                return input[:-4]
        # ou -> null
        elif input.endswith('ou') and _compute_m(input[:-2]) > 0:
                return input[:-2]
        # ism -> null
        elif input.endswith('ism') and _compute_m(input[:-3]) > 0:
                return input[:-3]
        # ate -> null
        elif input.endswith('ate') and _compute_m(input[:-3]) > 0:
                return input[:-3]
        # iti -> null
        elif input.endswith('iti') and _compute_m(input[:-3]) > 0:
                return input[:-3]
        # ous -> null
        elif input.endswith('ous') and _compute_m(input[:-3]) > 0:
                return input[:-3]
        # ive -> null
        elif input.endswith('ive') and _compute_m(input[:-3]) > 0:
                return input[:-3]
        # ize -> null
        elif input.endswith('ize') and _compute_m(input[:-3]) > 0:
                return input[:-3]

        return input

def _step5(input):
        """
        _step5 - function to apply step rules
        Inputs:
                - input : str
        Outputs:
                - input
        """
        cvc_pattern = r'[a-z]*[b-df-hj-np-tv-z][aeiouy][b-df-hj-np-tvz]'

        # m > 1 and ending w/ 'e'
        if _compute_m(input[:-1]) > 1and input.endswith('e'):
                return input[:-1]
        # m == 1 and not *o
        elif _compute_m(input[:-3]) == 1 and not re.search(cvc_pattern, input):
                return input[:-1]
        # m > 1 and *d and *L -> single-letter
        elif _compute_m(input[:-2]) > 1 and input[-1] == input[-2] and input[-1] == 'l':
                return input[:-1]
        return input

def stem(input):
        """
        stem - function to stem words to their roots
        Inputs:
                - input : str
                        Token
        Outputs:
                - input : str
                        Stem
        """
        # step 1
        input = _step1(input)

        # step 2
        input = _step2(input)

        # step 3
        input = _step3(input)

        # step 4
        input = _step4(input)

        # step 5
        input = _step5(input)

        return input

def tokenize(text):
        """
        tokenize - function to perform tokenization
        Inputs:
                - text : str
        Outputs:
                - text : list of tokens
        """
        # substitue puncutations with whitespace
        text = re.sub(pattern = '[,.!?\\<>]*', repl = '', string = text)

        # strip heading, tailing, and in-middle whitepsace
        text = re.sub(pattern = '\s+', repl = ' ', string = text).strip()

        # split text into tokens by whitespace
        text = text.split(' ')

        return text

def process(text, pattern = '(<\/?[a-zA-Z]*>)|(\n)'):
        """
        process - process SGML-styled text into preferred text
        Inputs:
                - text : raw text
                - pattern : matching pattern to remove SGML tags
        Outputs:
                - text : processed text
        """

        # remove SML tags
        text = re.sub(pattern = pattern, repl = ' ', string = text)

        # remove space in middle
        text = re.sub(pattern = '\s+', repl = ' ', string = text).strip()

        return text

def lem(text):
    """
    Lemmatize text
    """

    return lemmatizer.lemmatize(text)

def get_headline(text, pattern = '<TITLE>\\n[\w\\n\s\W]*</TITLE>'):
    text = re.findall(pattern, text)[0]
    return re.sub(pattern = '(<\/?[a-zA-Z]*>)|(\n)', repl = '', string = text)
