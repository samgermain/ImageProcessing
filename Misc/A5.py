def index_responses (list):
    '''
    Takes a list of questions and answers and returns a dictionary pairing them together
    :param list: list of questions and answers
    :return: a dictionary pairing the answers to the questions
    '''
    dictionary = {}
    for count in range(len(list)):
        dictionary["Q" + str(count + 1)] = list[count]
    return dictionary

print ('Testing index_responses () ')
print ( index_responses ([ 'a', 'b', 'c']))
print ( index_responses ([ 'a','a','c','b']))
print ( index_responses ([ 'd','d','b','e','e','e','d','a']))
print("\n")

def index_student (list):
    '''
    Takes a list with a student name, student number, and their questions and answers and returns a dictionary
    :param list: list of a student name, student number, and their answers and questions
    :return: a dictonary pairing the student information
    '''
    dictionary = {'ID':list[0], 'Name':list[1]}
    new_list = []
    for count in range(2,len(list)):
        new_list.append(list[count])
    dictionary['Responses'] = index_responses(new_list)
    return dictionary

print ('Testing index_student () ')
print ( index_student ([ '345 ','xyzzy ','a','a','c','b']))
print ( index_student ([ '10021795 ','Samden Cross ','d','d','b','e','e','e','d','a']))
print("\n")

def index_class (list):
    '''
    Takes a list of multiple students and pairs their test information into a dictionary
    :param list: The list of students with their name, student number and test answers
    :return: A dictionary with student test information
    '''
    dictionary = {}
    for count in range(len(list)):
        in_list = list[count]
        dictionary[in_list[0]] = index_student(in_list)
    return dictionary

print ('Testing index_class () ')
print ( index_class ([[ '123 ','foo ', 'a','b','c','a'],['234 ','bar ', 'a','b','c','b'],['345 ','xyzzy ','a','a','c','b']]))
print ( index_class ([[ '10021795 ','Samden Cross ', 'd','d','b','e','e','e','d','a'],['11051158 ','Jenni Nuxulon ','d','d','b','e','e','d','d','a']]))
print("\n")

def grade_student(right_answers, student_answers):
    '''
    Takes an answer key and student answers and returns the score the student got
    :param right_answers: The answer key as a dicitonary
    :param student_answers: The student answers as a dictionary
    :return:
    '''
    score = 0
    for key,value in right_answers.items():
        if student_answers[key] == value:
            score = score + 1
    return score

print('Testing grade_student ')
answers = index_responses([ 'a', 'b', 'c'])
resp1 = index_responses([ 'a', 'b', 'b'])
resp2 = index_responses([ 'a', 'b', 'c'])
print('Correct responses for first example: ', grade_student(answers, resp1))
print('Correct responses for second example: ', grade_student(answers, resp2))
print("\n")

def grade (right_answers, response_db):
    '''
    Takes an answer and a database of student information and returns the database with the students test scores
    :param right_answers: The answer key as a dictionary
    :param response_db: A dictionary of student information
    :return: The database with the student test scores
    '''
    new_db = {}
    for num,data in response_db.items():
        data['Score'] = grade_student(right_answers,data['Responses'])
        new_db[num] = data
    return new_db

print('Testing grade ')
answers = index_responses([ 'a', 'b', 'c', 'b'])
response_db = index_class([[ '123 ', 'foo ', 'a', 'b', 'c', 'a'],['234 ', 'bar ', 'a', 'b', 'c', 'b'],['345 ', 'xyzzy ', 'a', 'a', 'c', 'b']])
print('Response DB before ')
print(response_db)
grade(answers, response_db)
print('Response DB after ')
print(response_db)
print("\n")

def read_response_file(file):
    '''
    Takes a file of information separated by commas and returns the information as lists
    :param file: The file of information
    :return: The information as lists
    '''
    answers = []
    with open(file) as f:
        for line in f:
            nline = str.rstrip(line)
            answers.append(nline.split(","))
    return answers

print ('Testing grade ')
data = read_response_file ('cmpt181_midterm.txt')
print ( data [0:3])

def write_score_file(file, response_db):
    '''
    Takes a file of information separated by commas and a database of student information and writes the students test scores
    :param file: The file of test score information
    :param response_db: The student information about the tests
    :return: The test scores of the student
    '''
    answers = read_response_file(file)

print ('Testing write_score_file ')
answers = index_responses (['a','b','c','b'])
response_db = index_class ([[ '123 ','foo ', 'a','b','c','a'],
['234 ','bar ', 'a','b','c','b'],
['345 ','xyzzy ','a','a','c','b']])
grade ( answers , response_db )
write_score_file (' score_file_example .txt ', response_db )