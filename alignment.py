from rapidfuzz import distance as fuzz_dist, process
import nltk
from nltk.tokenize import word_tokenize
import string


def preprocess_text(sentence, level):

    """
    Preprocess and tokenize text for alignment task. Remove punctuation.

    Args:
      sentence: targeted text
      level: level to split the text, ex. word, phoneme, character

    Return:
      list: tokenized text
    """
    
    # Remove punctuation from each word, convert to lowercase and strip whitespace
    remove_punct = str.maketrans('', '', string.punctuation)
    sentence = sentence.translate(remove_punct).lower()

    # Tokenize the sentence
    if level == "word":
        processed_words = word_tokenize(sentence)
    elif level in ["phoneme", "character"]:
        processed_words = list(sentence)
    return processed_words


def word_level_alignment(reference_sentence, hypothesis_sentence):

    """
    Align ASR transcriptions with the reference text. Refer to Jiwer 
    and use fuzz_dist.Levenshtein.editops for alignment. Each word in 
    the transcription should be marked as Correct, Substituted, or Deleted.

    Args:
      reference_sentence
      hypothesis_sentence

    Return:
      list[dict]: word-level features for each word in the reference sentence
    """

    # Tokenize sentences into words
    ref_words = preprocess_text(reference_sentence, "word")
    hyp_words = preprocess_text(hypothesis_sentence, "word")
    
    # Compute edit operations needed to transform hypothesis into reference at the word level
    edit_ops = fuzz_dist.Levenshtein.editops(ref_words, hyp_words)

    # Generate opcodes from the edit operations
    opcodes = fuzz_dist.Opcodes.from_editops(edit_ops)

    # Process each opcode 
    word_alignment_result = []
    index = 0
    for op in opcodes:

        if op.tag == 'equal':
            for i in range(op.src_start, op.src_end):
                word_alignment_result.append({
                    "HypoIndex": index,
                    "Reference Word": ref_words[i],
                    "Hypothesis Word": hyp_words[index],
                    "Status": "Correct"
                })
                index += 1
        elif op.tag == 'replace':
            for i, j in zip(range(op.src_start, op.src_end), range(op.dest_start, op.dest_end)):
                word_alignment_result.append({
                    "HypoIndex": index,
                    "Reference Word": ref_words[i],
                    "Hypothesis Word": hyp_words[index],
                    "Status": "Substituted"
                })
                index += 1
        elif op.tag == 'delete':
            for i in range(op.src_start, op.src_end):
                word_alignment_result.append({
                    "HypoIndex": None,
                    "Reference Word": ref_words[i],
                    "Hypothesis Word": None,
                    "Status": "Deleted"
                })
        elif op.tag == 'insert':

            if not word_alignment_result:
                continue

            for j in range(op.dest_start, op.dest_end):

                #FOR ERROR CHECK
                '''
                word_alignment_result.append({
                    "HypoIndex": index,
                    "Reference Word": None,
                    "Hypothesis Word": hyp_words[index],
                    "Status": "Inserted"
                })
                '''
                index += 1

    return word_alignment_result

def phoneme_level_alignment(reference_sentence, hypothesis_sentence):

    """
    Align ASR transcriptions with the reference text. Refer to Jiwer 
    and use fuzz_dist.Levenshtein.editops for alignment. Each character in 
    the transcription is marked as Correct, Substituted, or Deleted.
    Return correctness rate on the phoneme level for each word.

    Args:
      reference_sentence
      hypothesis_sentence

    Return:
      list[dict]: phoneme-level features for each word in the reference sentence
    """


    # Tokenize sentences into words
    ref_words = preprocess_text(reference_sentence, "phoneme")
    hyp_words = preprocess_text(hypothesis_sentence, "phoneme")

    # Compute edit operations needed to transform hypothesis into reference at the word level
    edit_ops = fuzz_dist.Levenshtein.editops(ref_words, hyp_words)

    # Generate opcodes from the edit operations
    opcodes = fuzz_dist.Opcodes.from_editops(edit_ops)

    # Process each opcode 
    ref_phoneme = ""
    hypo_phoneme = ""
    status_sequence = ""
    index = 0
    for op in opcodes:
        if op.tag == 'equal':
            for i in range(op.src_start, op.src_end):
                ref_phoneme = ref_phoneme + ref_words[i]
                hypo_phoneme = hypo_phoneme + hyp_words[index]
                status_sequence = status_sequence + "C"
                index += 1
        elif op.tag == 'replace':
            for i, j in zip(range(op.src_start, op.src_end), range(op.dest_start, op.dest_end)):
                ref_phoneme = ref_phoneme + ref_words[i]
                hypo_phoneme = hypo_phoneme + hyp_words[index]
                status_sequence = status_sequence + "S"
                index += 1
        elif op.tag == 'delete':
            for i in range(op.src_start, op.src_end):
                ref_phoneme = ref_phoneme + ref_words[i]
                hypo_phoneme = hypo_phoneme + "*"
                status_sequence = status_sequence + "D"
        elif op.tag == 'insert':

            if not ref_phoneme:
                continue

            for j in range(op.dest_start, op.dest_end):

                ref_phoneme = ref_phoneme + "*"
                hypo_phoneme = hypo_phoneme + hyp_words[index]
                status_sequence = status_sequence + "I"
                
                index += 1
    
    phoneme_alignment_result = []
    word_length, correctness = 0, 0
    prev_i = 0
    for i, (ref, hypo, status) in enumerate(zip(ref_phoneme, hypo_phoneme, status_sequence)):
        if ref == " " or i == len(ref_phoneme) - 1:
            if i == len(ref_phoneme) - 1:
               i += 1
            phoneme_alignment_result.append({'ref_phoneme': ref_phoneme[prev_i:i], 
                                            'hypo_phoneme': hypo_phoneme[prev_i:i], 
                                            'phoneme_correct_rate': status_sequence[prev_i:i].count("C")/(i-prev_i)}
                                            )
            prev_i = i+1
            word_length, correctness = 0, 0

    return phoneme_alignment_result