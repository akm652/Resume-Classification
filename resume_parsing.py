#!/usr/bin/env python
# coding: utf-8

# In[30]:


# import libraries from PyPDF
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from io import StringIO
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser


# In[5]:

# function for parsing pdf files and outputs string
def pdf_parsing(pdf_filepath):
    output_string = StringIO()
    with open(pdf_filepath, 'rb') as in_file:
        parser = PDFParser(in_file)
        doc = PDFDocument(parser)
        rsrcmgr = PDFResourceManager()
        device = TextConverter(rsrcmgr, output_string, laparams=LAParams())
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        for page in PDFPage.create_pages(doc):
            interpreter.process_page(page)

    pdf_text = output_string.getvalue()

    return pdf_text


# In[46]:

# function for removing any brackets inside text
def remove_nested_brackets(test_str):
    ret = ''
    skip1c = 0
    skip2c = 0
    for i in test_str:
        if i == '[':
            skip1c += 1
        elif i == '(':
            skip2c += 1
        elif i == ']' and skip1c > 0:
            skip1c -= 1
        elif i == ')' and skip2c > 0:
            skip2c -= 1
        elif skip1c == 0 and skip2c == 0:
            ret += i
    return ret


# In[49]:

# function for preprocessing text obtained from parsing the pdf file
def custom_preprocessing(text):
    text_tokens = word_tokenize(text)

    months = ['Jan', 'Feb', 'Mac', 'Apr', 'May', 'Jun', 'Jul', 'July', 'Aug', 'Sept', 'Oct', 'Nov', 'Dis']

    tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]

    tokens_without_months = [word for word in tokens_without_sw if not word in months]

    new_text = ' '.join(map(str, tokens_without_months))

    new_text = new_text.replace('•', '.')

    new_text = new_text.replace('. .', '.')

    new_text = new_text.replace('GPA', ' ')

    new_text = new_text.replace(':', '')

    new_text = new_text.replace('–', '')

    new_text = new_text.replace('/', '')

    new_text = new_text.replace('  ', ' ')

    new_text = new_text.replace('   ', ' ')

    new_text = re.sub('[0-9]+', '', new_text)

    new_text = remove_nested_brackets(new_text)

    return new_text
