import argparse

import json
import string
import pandas as pd
from lexical import get_lexical_features
from alignment import word_level_alignment, phoneme_level_alignment
from tqdm import tqdm

# import dictionaries
json_file_path = 'arpabet_to_amirabet.json'
with open(json_file_path, 'r') as file:
    arpabet_to_amirabet = json.load(file)

word_to_arpabet = {}
with open('all_story_words.dic', 'r') as file:
    for line in file:
        parts = line.strip().split()
        word = parts[0]
        phonemes = parts[1:]
        word_to_arpabet[word] = phonemes


remove_punct = str.maketrans('', '', string.punctuation)

def convert_text_to_phonemes(text):
    """ Convert text to phonemes using the dictionaries. """
    text = text.translate(remove_punct).upper()
    words = text.split()
    phoneme_text = []
    for word in words:
        if word in word_to_arpabet:
            arpabet_phonemes = word_to_arpabet[word]
            amirabet_phonemes = [arpabet_to_amirabet[ph] for ph in arpabet_phonemes if ph in arpabet_to_amirabet]
            phoneme_text.append(''.join(amirabet_phonemes))
        else:
            phoneme_text.append('UNK')  # UNK for unknown words
    return ' '.join(phoneme_text)


def data_generation(df):
    results = []
    for index, row in tqdm(df.iterrows()):
        asr_outputs = {
            'Amazon': row['amazon_data'],
            'Kaldi': row['kaldi_data'],
            'KaldiNA': row['kaldiNa_data'],
        }
        
        story_text = row['story_text']
        word_result = {}
        try:
            # apply word-level alignment to the three asr transcriptions
            for key, asr_output in asr_outputs.items():

                asr_output = eval(asr_output)
                word_alignments = word_level_alignment(story_text, asr_output["text"])
                phoneme_alignments = phoneme_level_alignment(\
                                  convert_text_to_phonemes(story_text), \
                                  row['wav2vec_transcript_phonemes']
                              )

                for i in range(len(word_alignments)):

                    word_alignment = word_alignments[i]
                    expected_text = word_alignment['Reference Word']

                    if not expected_text: continue
                    
                    if i not in word_result: 
                        word_result[i] = {'activityId': row['activityId'],
                                          'phraseIndex': row['phrase_index'],
                                          'word_index': i,
                                          'expected_text': expected_text
                                          }

                    # word level alignment
                    if key == 'Amazon':
                        if word_alignment['HypoIndex'] is None:
                            word_result[i]['amazon_deleted'] = 1 
                        else:
                            word_result[i]['amazon_lapse'] = asr_output['lapse'][word_alignment['HypoIndex']][-1]
                            word_result[i]['amazon_confidence'] = asr_output['confidence'][word_alignment['HypoIndex']][-1]
                            word_result[i]['amazon_correct'] = 1 if word_alignment['Status'] == 'Correct' else 0
                            word_result[i]['amazon_substituted'] = 1 if word_alignment['Status'] == 'Substituted' else 0


                    if key == 'Kaldi':
                        if word_alignment['HypoIndex'] is None:
                            word_result[i]['kaldi_deleted'] = 1
                        else:
                            word_result[i]['kaldi_lapse'] = asr_output['transcription'][word_alignment['HypoIndex']]['confidence']
                            word_result[i]['kaldi_confidence'] = asr_output['transcription'][word_alignment['HypoIndex']]['end_time'] \
                                                - asr_output['transcription'][word_alignment['HypoIndex']]['start_time']
                            word_result[i]['kaldi_correct'] = 1 if word_alignment['Status'] == 'Correct' else 0
                            word_result[i]['kaldi_substituted'] = 1 if word_alignment['Status'] == 'Substituted' else 0

                    if key == 'KaldiNA':
                        if word_alignment['HypoIndex'] is None:
                            word_result[i]['kaldina_deleted'] = 1
                        else:
                            word_result[i]['kaldina_lapse'] = asr_output['transcription'][word_alignment['HypoIndex']]['confidence']
                            word_result[i]['kaldina_confidence'] = asr_output['transcription'][word_alignment['HypoIndex']]['end_time'] \
                                                - asr_output['transcription'][word_alignment['HypoIndex']]['start_time']
                            word_result[i]['kaldina_correct'] = 1 if word_alignment['Status'] == 'Correct' else 0
                            word_result[i]['kaldina_substituted'] = 1 if word_alignment['Status'] == 'Substituted' else 0
                    
                    # phoneme level alignment
                    word_result[i].update(phoneme_alignments[i])

                    # lexical features
                    word_result[i].update(get_lexical_features(expected_text))
        except:

            # print(index, row['activityId'], row['phrase_index'])

            continue

        results.extend(list(word_result.values()))

    return results


def main(args):

    labels_df = pd.read_csv(args.label_path)
    labels_df.expected_text = labels_df.expected_text\
                      .apply(lambda x: x.translate(remove_punct).lower())
    asr_data_df = pd.read_csv(args.asr_data_path)
    
    print("Data Preprocessing...")

    processed_df = data_generation(asr_data_df)
    processed_df = labels_df.merge(processed_df , on=['activityId', 'phraseIndex', 'word_index'])
    processed_df.to_csv(args.save_path, index=False)

    print("Processed dataset saved.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some strings.')
    parser.add_argument('--label_path', type=str, default="labels.csv",help='path to label dataset')
    parser.add_argument('--asr_data_path', type=str, default="asr_data.csv", help='path to label dataset')
    parser.add_argument('--save_path', type=str, default="processed_data.csv", help='path to save the processed dataset')
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(args)
