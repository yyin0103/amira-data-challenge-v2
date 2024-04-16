# Amira Data Challenge

This repository includes the files and code used for the challenge. The files include:

- `alignment.py`: Includes functions that align word-level and phoneme-level ground truth to three ASR results.
- `data_prep.py`: Code to prepare the dataset for training.
- `lexical.py`: Functions to extract word-level lexical features.
- `model.py`: Includes a simple process of model experimentation with briefly generated data.
- `result.txt`: Model results.
  
### Idea
The code focuses on data preparation to binarily-detect students' errors. I concentrated on aligning transcriptions to corresponding sentences, referencing Jiwer's method to process errors, and considering some phonological and phonetic features at both word and phoneme levels to model training. I then fed the prepared data into several binary classification models. The best performance, based on an 80/20 split dataset, achieved an F1-score of 0.91 and ROC-AUC of 0.9052823725465227. However, there are further improvements that could enhance the model's performance.

### Future Improvements
- Model Development: Explore the implementation of a sequential model and more to enhance predictive capabilities.
- Patterns Finding and Feature Generation: Further investigation is needed into the intricate relationships between errors, sounds, and linguistic patterns. This time, I didn't get to explore how inserted words in transcription could help us identify errors. Additionally, there is potential to introduce more phonetic-related features. Areas of interest include analyzing how students handle linguistic phenomena such as liaison, vowel reduction, intonation, and stop words.
- Open Source Tools: Investigate and integrate more open-source tools to streamline and enhance the development process.

### Reference
- [Pronunciation error detection model based on feature fusion (Zhu et al, 2024)](https://www.sciencedirect.com/science/article/pii/S0167639323001437?ssrnid=4470827&dgcid=SSRN_redirect_SD)
- [End-to-End Automatic Pronunciation Error Detection Based on Improved Hybrid CTC/Attention Architecture (Zhang. et al, 2020)](https://github.com/tuanio/noisy-student-training-asr)
- [ASR Error Detection via Audio-Transcript entailment (Meripo et al, 2022)](https://arxiv.org/pdf/2207.10849.pdf)



### Challenge Instructions
Dear candidate:

Thank you for accepting our data challenge! In this directory, you will find several files:

- `labels.csv`
- `asr_data.csv`
- `all_story_words.dic`
- `arpabet_to_amirabet.json`

These files contain realistic data related to many student sessions (activities). In each session, a student reads
a story out loud.  A human expert then listens to the student's reading and marks, for each word in the story, whether
the student read the word correctly or not. Reproducing this "scoring" work of such a human expert is one of the problems
we work on.

Here are more details on the data.


1. All of the human scoring data is located in `labels.csv`. Each unique session (`activityId`) involves one story being read.
There is a label for each word of the story in each session. A label of 1 means the child read the word correctly, and a label
of 0 means the child did not read the word correctly. Note that each story is broken down into phrases, denoted by
`phraseIndex`. The order of the `phraseIndex` numbers indicates the ordering of the phrases in the story. The column `word_index`
indicates the order of the words as they unfold across a phrase. The column `expected_text` indicates the target story word
to which the label applies.


2. The file `asr_data.csv` contains transcriptions and, in some cases, confidence values and/or timing information outputted by
different automatic speech recognition (ASR) systems when we feed them audio from the student reading each phrase of each story.
The ASRs include:
- Amazon (https://aws.amazon.com/transcribe/) - data found in `amazon_data` column
- A version of Kaldi (https://kaldi-asr.org/) that contains alternative pronunciations in its language model - data found in the
`kaldi_data` column
- A version of Kaldi that only contains story text words in its language model - data found in the
`kaldina_data` column
- wav2vec 2.0 (https://arxiv.org/abs/2006.11477) - transcripts found in the `wav2vec_transcript_words` column
- A version of wav2vec 2.0 that transcribes a phoneme, rather than letter, sequence - transcripts found in the
`wav2vec_transcript_phonemes` column. NOTE: these phoneme transcripts use an alphabet called AMIRABET, *not* to be confused with
the International Phonetic Alphabet (IPA; there is a lot of overlap in these two alphabets, but they are not the same).

As in the labels file, `activityId` denotes the session and `phraseIndex` denotes the index of the phrase of the story that the
transcriptions pertain to. The story text corresponding to the phrase can be found in `story_text`. 


3. The file `all_story_words.dic` contains a dictionary mapping a huge number of words (inclusive of, but not limited to, all of the
words found in the stories) to the underlying phoneme sequence representing how they ought to be said out loud. The alphabet used in
this file's mapping is called ARPABET. You can learn more about this phoneme alphabet here: http://www.speech.cs.cmu.edu/cgi-bin/cmudict
(section "Phoneme Set") and here: https://en.wikipedia.org/wiki/ARPABET.


4. The file `arpabet_to_amirabet.json` contains a one-to-one mapping between unique symbols in ARPABET and unique symbols in AMIRABET.
If you do any mapping to phonemes, you should use `all_story_words.dic` and `arpabet_to_amirabet.json`.  Do *not* convert words to IPA,
or they will not perfectly match the alphabet used by `wav2vec_phonetic`.


Your task is to make meaningful steps towards creating an automated system that accurately labels student errors at the word level.
Making a complete system like that is a daunting task; it is not doable in a few hours and we are not expecting that. We ask you to
limit the work to a reasonable amount of time, and develop a piece of methodology that would demonstrably help in solving the problem.
For example, you may choose to develop features that would be useful in training a supervised learning classifier for student errors.
Or you may choose to focus on data preparation or finding useful patterns and insights in the data. You should not feel the need to
use all of the data or features you are given (for example - you may choose not to consume all the ASRs). On the flip side, you should
feel free to pull in external, open-source data or resources if you believe they are helpful in solving the problem.

We would like to see how you work with a new and unfamiliar problem, and fairly complex and noisy data. We hope to see inventiveness,
appropriate methodological sophistication, and fluency with relevant tools.

Please submit your code and report via email, or send a link to a new github repository (not this repository - otherwise other candidates
may see your work). Please do not include the data in the repository. Please email your submission to ran.liu@amiralearning.com.

We thank you for your time, and are looking forward to learning about your findings. Good luck!

Amira team.
