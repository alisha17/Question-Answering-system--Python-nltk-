This program was written as the final assignment for “Introduction to Natural Language Processing” (CMPS_143) class at University of California Santa Cruz by Jeremy Chien, Ethan Seither, and Ekaterina Tcareva. 

Our goal was to design and build a question answering system  which can produce the answers to the questions about a given text. 

Our question-answering system (q_a_new_chunk_2.py) accepts a single input file as a command-line argument. This file contains a list of file StoryIDs in the order that they should be processed. We assume that we have some files which contain the stories and some files which contain the questions in the same directory.  

To run the program:
python3 q_a_new_chunk_2.py hw8_process_stories.txt

The output is the single response file (chien_seither_tcareva_answers.txt) which contains a list of answers.
Each question file contains 4 lines for each question indicating the QuestionID, the question itself, a difficulty rating, and the file we should use to find the answer. The Difficulty ratings are based upon the methods required for extracting the answer. 

For example, for  blogs-01.questions:

………

QuestionID: blogs-01-3

Question: Where did the protest happen?

Difficulty: Easy

Type: Story | Sch


QuestionID: blogs-01-4
Question: What did the people burn?
Difficulty: Easy
Type: Sch

QuestionID: blogs-01-5
Question: Who rebelled?
Difficulty: Easy
Type: Sch

QuestionID: blogs-01-6
Question: Who created a riot?
Difficulty: Easy
Type: Sch
…………..

Output:
….
QuestionID: blogs-01-3
Answer: along the street

QuestionID: blogs-01-4
Answer: a police car

QuestionID: blogs-01-5
Answer: The people

QuestionID: blogs-01-6
Answer: The people
…..

Our Q/A system assumes that for each StoryID, the directory contains:
 a story file named StoryID.story (e.g., “fables-01.story”)
 a question file named StoryID.questions (e.g., “fables-01.questions”)
We also use:
	•	two WordNet .csv files that you can use to do word sense disambiguation. One is called “Wordnet nouns.csv” and the other is called Wordnet verbs.csv”. 
 •	Scheherazade realization files named StoryID.sch (e.g., “blogs-01.sch”) 
•	constituency parses of the story named StoryID.story.par (e.g., “blogs-01.story.par”) 
	•	dependency parses of the story named StoryID.story.dep (e.g., “blogs-01.story.dep”) 
	•	constituencyparsesoftheScheherazadestorynamedStoryID.sch.par(e.g.,“blogs-01.sch.par”)  
	•	dependencyparsesoftheScheherazadestorynamedStoryID.sch.dep(e.g.,“blogs-01.sch.dep”)  
	•	constituencyparsesofthequestionsnamedStoryID.questions.par(e.g.,“blogs-01.questions.par”)  
	•	dependencyparsesofthequestionsnamedStoryID.questions.dep(e.g.,“blogs-01.questions.dep”)  
The performance of our question-answering system was evaluated using the F-measure statistic which combined recall and precision in a single evaluation metric. Since question-answering systems often produce answers that are partially but not fully correct, the test program (it was provided for us) scored each answer by computing the Word Overlap between the system’s answer and the strings in the answer key. Given an answer string from the answer key, our system’s response was scored for recall, precision, f-measure. We got:
AVERAGE RECALL =    0.7094  (165.29 / 233)
AVERAGE PRECISION = 0.6760  (154.80 / 229)
AVERAGE F-MEASURE = 0.6923
Our  result put us in the best 5 teams for this class.
