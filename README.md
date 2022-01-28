# amira-data-challenge-v2

# Amira Data Challenge

Dear candidate:

Thank you for accepting our data challenge! In this directory, you will find several files:

- `labels.csv`
- `asr_transcripts.csv`
- `all_story_words.dic`
- `arpabet_to_amirabet.json`

These files contain realistic data related to many student sessions (activities). In each session, a student reads
a story out loud.  A human expert then listens to the student's reading and marks, for each word in the story, whether
the student read the word correctly or not. Reproducing this "scoring" work of such a human expert is one of the problems
we work on.

Here are more details on the data.


1. All of the human scoring data is located in `labels.csv`. Each unique session (`activityId`) involves one story being read.
There is a label for each word of the story in each session. A label of 1 means the child read the word correctly, and a label
of 2 means the child did not read the word correctly. Note that each story is broken down into phrases, denoted by
`phraseIndex`. The order of the `phraseIndex` numbers indicates the ordering of the phrases in the story. The column `word_index`
indicates the order of the words as they unfold across a phrase. The column `expected_text` indicates the target story word
to which the label applies.


2. The file `asr_transcripts.csv` contains transcriptions outputted by different automatic speech recognition (ASR) systems when
we feed them audio from the student reading each phrase of each story. The ASRs include:
- Amazon (https://aws.amazon.com/transcribe/) - transcripts found in `amazon_unaligned` column
- A version of Kaldi (https://kaldi-asr.org/) that contains alternative pronunciations in its language model - transcripts found
in `kaldi_unaligned` column
- A version of Kaldi (https://kaldi-asr.org/) that only contains story text words in its language model - transcripts found in
`kaldina_unaligned` column
- wav2vec 2.0 (https://arxiv.org/abs/2006.11477) - transcripts found in the `wav2vec_unaligned` column
- A version of wav2vec 2.0 that transcribes a phoneme, rather than letter, sequence - transcripts found in the `wav2vec_phonetic` column.
NOTE: these phoneme transcripts use an alphabet called AMIRABET, *not* to be confused with the International Phonetic Alphabet (IPA; there
is a lot of overlap in these two alphabets, but they are not the same).

As in the labels file, `activityId` denotes the session and `phraseIndex` denotes the index of the phrase of the story that the transcriptions
pertain to. The story text corresponding to the phrase can be found in `story_text`. 


3. The file `all_story_words.dic` contains a dictionary mapping a huge number of words (inclusive of, but not limited to, all of the
words found in the stories) to the underlying phoneme sequence representing how they ought to be said out loud. The alphabet used in
this file's mapping is called ARPABET. You can learn more about this phoneme alphabet here: http://www.speech.cs.cmu.edu/cgi-bin/cmudict
(section "Phoneme Set") and here: https://en.wikipedia.org/wiki/ARPABET.


4. The file `arpabet_to_amirabet.json` contains a one-to-one mapping between unique symbols in ARPABET and unique symbols in AMIRABET.
If you do any mapping to phonemes, you should use `all_story_words.dic` and `arpabet_to_amirabet.json`.  Do *not* convert words to IPA,
or they will not perfectly match the alphabet used by `wav2vec_phonetic`.


Your task is to make meaningful steps towards creating an automated system that accurately labels student errors at the word level.
Making a complete system like that is a daunting task; it is not doable in a few hours. We ask you to limit the work to a reasonable
amount of time, and develop a piece of methodology that would demonstrably help in solving the problem. For example, you may choose
to develop features that would be useful in training a supervised learning classifier for student errors. Or you may choose to focus
on data preparation or finding useful patterns and insights in the data. You should feel free to pull in external, open-source data or
resources if you believe they are helpful to solving the problem.

We would like to see how you work with a new and unfamiliar problem, and fairly complex data. We hope to see inventiveness, appropriate
methodological sophistication, and fluency with relevant tools.

Please submit your code and report via email, or send a link to a new github repository (not this repository - otherwise other candidates
may see your work). Please do not include the data in the repository. Please email your submission to ran.liu@amiralearning.com.

We thank you for your time, and are looking forward to learning about your findings. Good luck!

Amira team.
