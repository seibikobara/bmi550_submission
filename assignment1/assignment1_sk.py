'''
Assignment 1
Seibi Kobara
'''




import re
import os
import numpy as np
import pandas as pd
from nltk import ngrams
from fuzzywuzzy import fuzz
import Levenshtein
from nltk.tokenize import sent_tokenize
import nltk




# prepare annotation database
# 1. negation list, which was used in regular expression 

path = "./neg_trigs.txt"
negs = []
with open(path, "r") as input:
    for i in input:
        w = i.strip()
        negs.append(w) 
negs = '|'.join(negs)


# 2. create a reference by which a model identify symptoms
# 2-1. All manually annotated file by students were used


path = "/Users/seibi/projects/bmi550/assignment1/annots"
dir_list = os.listdir(path)

anno_dic = {}

for i in dir_list:
    path = "./annots/" + i
    annotated_file = pd.read_excel(path)
    ref_dic = {}
    for index,row in annotated_file.iterrows():
        used_expr = re.sub("\$+","$$$", row["Symptom Expressions"]).strip("$$$").split("$$$")
        standard_sym = re.sub("\$+","$$$", row["Standard Symptom"]).strip("$$$").split("$$$")
        cui = re.sub("\$+","$$$", row["Symptom CUIs"]).strip("$$$").split("$$$")
        
        # lower
        used_expr_lowered = []
        for i in used_expr:
            used_expr_lowered.append(i.lower())

        standard_sym_lowered = []
        for i in standard_sym:
            standard_sym_lowered.append(i.lower())


        if len(used_expr_lowered) == len(standard_sym_lowered)  ==  len(cui):
            temp = pd.DataFrame({
                "used_expr": used_expr_lowered,
                "standard_sym": standard_sym_lowered,
                #"cui": cui
            })
            ref_dic[index] = temp

    temp_sum = pd.concat(ref_dic.values())
    anno_dic[path] = temp_sum

anno_ref = pd.concat(anno_dic.values())

# remove duplicated expression
anno_ref = anno_ref.drop_duplicates(subset = ['used_expr',"standard_sym"])


# 2-2. Previously annotated data

path = "./COVID-Twitter-Symptom-Lexicon.txt"
ref = pd.read_csv(path, delimiter = "\t", names = ["standard_sym","cui","used_expr"])
ref["standard_sym"] = ref["standard_sym"].str.lower()
ref["used_expr"] = ref["used_expr"].str.lower()


# manually curated data has multiple incorrect labels for CUI, so the previously annotated data was used to match CUIs

cuis = []
for index,row in anno_ref.iterrows():
    standard = row["standard_sym"]
    cui_all = ref.loc[ref["standard_sym"] == standard, "cui"]
    cui = "".join(list(set(cui_all)))
    cuis.append(cui)
anno_ref["cui"] = cuis


# combine
ref_com = pd.concat([anno_ref, ref])

# remove unavailable cui
ref_com = ref_com[~ref_com["cui"].isin([""])]

# remove duplication
ref_com = ref_com.drop_duplicates(subset = ["used_expr"])





# function to read unlebeld test data
def post_to_string(post):
    post = str(post)
    censored = ""
    for i in post:
        # lower the case
        temp = i.strip('\n').lower().replace(")"," ").replace("(", " ").replace(":", " ").replace(";", " ")\
            .replace("]", " ").replace("[", " ").replace("*", " ").replace("/", " ")\
            .replace("\\", " ").replace(",", "").replace("?", "")
        censored = censored + temp
        
    return(censored)






# function to read a post and identify symptoms and negation context
def detect_system_from_post(string_post, reference):
    # param: string_post is a post 
    # param: reference is a annotated symptoms and corresponding standard symptoms and CUIs.

    # curated symptom, by which I caclulated levenshtein ratio
    curated_sym_list = reference["used_expr"]


    symptom_exps = "$$$"
    standards = "$$$"
    cuis = "$$$"
    negations = "$$$"
    if  string_post=="|nan":
        symptom_exps += "$$$"
        standards += "$$$"
        cuis +=  "$$$"
        negations += "$$$"
    else:
        # for this post, check curated symptoms exist
        sym_in_post_and_in_curated_sym = {}

        for curated_sym in curated_sym_list:
            
            # obtain ngrams by the length of this symptom    
            #print("curated_sym1: " + curated_sym)
            length = len([curated_sym])
            
            # ngram    
            words_post= ngrams(string_post.split(" "), length)
            for w_post in words_post:
                
                w_post = " ".join(w_post)
                
                normalized_levenshtein = Levenshtein.ratio(w_post, curated_sym)
                if normalized_levenshtein > 0.8:
                    # expr used in this post: w_post
                    # corresponding curated sym: curated_sym

                    sym_in_post_and_in_curated_sym[w_post] = curated_sym
                    
        # check if this symptoms is negated
        is_negated = False
        for sym_used in sym_in_post_and_in_curated_sym.keys():
            
            # pattern
            # 0: [not help symptom] -> not negation
            # 1: [not any word . symptom] -> not negation
            # 2. [not, symptom, w]
            # 3. [w, not, symptom]
            # 4. [not w symptom]
            

            pattern0 = "not help\s" + sym_used
            pattern1 = negs + "\s(\S*\s*){0,2}\\.\s" + sym_used
            pattern2_4 = "\S*\s*(" + negs +  ")\s(\S*\s*){0,1}" + sym_used

            # pattern 0        
            if re.search(pattern0, string_post):
               # not negation 
                is_negated = False
            
            # pattern 1
            elif re.search(pattern1, string_post):
                is_negated = False

            # 2, 3, 4
            elif re.search(pattern2_4, string_post):
                is_negated = True

            else: 
                is_negated = False

            
            # record
            symptom_exps += sym_used + "$$$"
            curated_exps = sym_in_post_and_in_curated_sym[sym_used]
            standard = reference.loc[reference["used_expr"] == curated_exps, "standard_sym"].values 
            standard = "".join(standard)
            standards += standard + "$$$"
            
            cui = reference.loc[reference["used_expr"] == curated_exps, "cui"].values 
            cui = "".join(cui)
            cuis += cui + "$$$"
            
            if is_negated:
                negations += str(1) + "$$$"
            else:
                negations += str(0) + "$$$"

    # final output
    symptom_exps += "$$$"
    standards += "$$$"
    cuis +=  "$$$"
    negations += "$$$"
    
    return(
            symptom_exps, 
            standards,
            cuis,
            negations)



# system performance evaluation
data = pd.read_excel("./goldstandard_for_test.xlsx")

n,m = data.shape
symptom_exps = []
standards = []
cuis = []
negations = []

for i in range(n):
    print(i)
    post = data["TEXT"][i]
    string = post_to_string(post)
    a,b,c,d = detect_system_from_post(string, ref_com)
    symptom_exps.append(a)
    standards.append(b)
    cuis.append(c)
    negations.append(d)


# outout excel
data["Symptom Expressions"] = symptom_exps
data["Standard Symptom"]    = standards
data["Symptom CUIs"] = cuis
data["Negation Flag"] = negations

data.to_excel("./goldstandard_for_test_annotated.xlsx")






# testing in the unlabeled data
# read data
data = pd.read_excel("./UnlabeledSet.xlsx")


n,m = data.shape
symptom_exps = []
standards = []
cuis = []
negations = []

for i in range(n):
    print(i)
    post = data["TEXT"][i]
    string = post_to_string(post)
    a,b,c,d = detect_system_from_post(string, ref_com)
    symptom_exps.append(a)
    standards.append(b)
    cuis.append(c)
    negations.append(d)


# output
data["Symptom Expressions"] = symptom_exps
data["Standard Symptom"]    = standards
data["Symptom CUIs"] = cuis
data["Negation Flag"] = negations


# final output
data.to_excel("./UnlabeledSet_annotated_SK.xlsx")