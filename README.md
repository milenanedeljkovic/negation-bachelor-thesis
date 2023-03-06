# Bachelor thesis: Impact of negation on the distribution of lexical items

## Using the scripts

### Parsing
parsing_script.py takes two integers (first, last) as arguments in the command line.  
The pages are parsed in collections of 10000 successive pages as follows:  
  
for(i = first, i < last, i += 10000):  
    parse page at index i of the dataset using stanza  
    append the output of the parsing to the file at location parsed/parsed{i}.conll  
  
  
A folder named "parsed" has to exist in the current directory, otherwise Python will throw an error.  
  
Examples:  
  
python3 parsing_script.py 0 1 : parses the pages from 0 (included) to 10000 (excluded)  
python3 parsing_script.py 0 10000 : parses the pages from 0 (included) to 10000 (excluded)  
python3 parsing_script.py 0 10001 : parses the pages from 0 (included) to 10000 (excluded)  
python3 parsing_script.py 0 30001 : parses the pages from 0 (included) to 30000 (excluded), separating the output  
                                    into 3 different files  
  
### Collecting contextual embeddings
collect_context_representations.sh is a shell script which runs collect_context_representations.py in a for loop,  
taking one by one the files in "parsed".  
  
As variables are not allowed within the for loop range in shell, in order to change the range the files to be parsed,  
one has to change the .sh directly (initially, I tried making taking the first and last indices of the pages to be  
analysed as arguments in the command line; this is still possible if .py script is run).  
  
collect_context_representations.py can be run on its own as follows:  
  
python3 parsing_script.py 0 1 : analyses the pages from 0 (included) to 10000 (excluded)  
python3 parsing_script.py 0 10000 : analyses the pages from 0 (included) to 10000 (excluded)  
python3 parsing_script.py 0 10001 : analyses the pages from 0 (included) to 10000 (excluded)  
python3 parsing_script.py 0 30001 : analyses the pages from 0 (included) to 30000 (excluded), separating the output  
                                    into 3 different files  
  
Folders named "embeddings" and "embeddings-avg" have to exist in the current directory, otherwise Python will throw an error.  
  
The overview of the page analysis is as follows:  
for(i = first, i < last, i += 10000):  
    open the file parsed/parsed{i}.conll  
    create an empty dictionary  
    for each dependency tree in the file (read using conll module):  
        one by one, calculate RoBERTa embeddings for the sentence  
        locate the verbs and whether they are negated or not by a depth-first search of the tree  
        add the verbs' contextual representations to the dictionary (using the stanza-to-bert token mapping function  
                                                                     and the verb lemma as the key in the dictionary)  
    save the dictionary into the file embeddings/embeddings{i}  
    make another dictionary from the existing one with the average representations and the number of occurrences  
    save the second dictionary into the file embeddings-avg/embeddings-avg{i}  
  
first dictionary has the structure: {verb lemma : \[\[list of representations of all negated occurrences\],  
                                                   \[list of representations of all non negated occurrences\]\]}  
  
second dictionary has the structure: {verb lemma : \[\[average representation of all negated occurrences\],  
                                                    \[average representation of all non negated occurrences\],  
                                                    number of negated occurrences,  
                                                    number of non negated occurrences\]}  
  
### Calculating the cosinuses and creating .csv files
cos_and_csv.py takes two integers as arguments (first, last) in the same manner as the previous two scripts (see examples).  
  
The .csv files are created incrementally as follows:  
  
for(i = first, i < last, i += 10000):  
    load dictionary from file embeddings-avg/embeddings-avg{i}  
    open the file csv_files/from{first}-from{i}.csv, then:  
        for each key in the dictionary:  
            calculate the cosine between the average negated and non negated representations  
            append row: key, total number of occurrences, number of non negated occurrences,  
                        num of negated occurrences, percentage of negated occurrences,  
                        cosine between the average of negated and non negated occurrences  
  
A folder named "csv_files" has to exist in the current directory, otherwise Python will throw an error.  
  
In the end, we end up with the following files in "csv_files": from{first}-from{first}.csv,  
                                                               from{first}-from{first + 10000}.csv,  
                                                               ...  
                                                               from{first}-from{last - (last % 10000)}.csv,  
containing respectively information from pages \[first:first + 10000\], \[first:first + 20000\] etc.  
    
