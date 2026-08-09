#!/usr/bin/python3

import stanza

class Statement:
    def __init__(self):
        self.subject = ""
        self.verb = ""
        self.object = ""
        self.noun = ""

    def print(self):
#        print(self.subject+" "+self.verb+" "+self.object+" "+self.noun)
        out = ""
        if self.verb != "" and (self.subject != "" or self.object != ""):
            if self.subject == "":
                if self.noun != "":
                    out = self.verb+"("+self.object+","+self.noun+")."
                else:
                    out = self.verb+"("+self.object+")."
            elif self.object == "":
                if self.noun != "":
                    out = self.verb+"("+self.subject+","+self.noun+")."
                else:
                    out = self.verb+"("+self.subject+")."
            else: 
                out = self.verb+"("+self.subject+","+self.object+self.noun+")."
            out = out.replace(" ","").replace("’","")
            print(out)
            return out+"\n"
        
        return ""

text = 'A black cat tears a white mouse.'


nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma')
doc = nlp(text)
res = []

for s in doc.sentences:
    st = Statement()
    sts = []

    for w in s.words:
        #print(w)
        if w.feats is not None:
            if "Case=Nom" in w.feats:
                if w.upos == "PRON" or w.upos == "NOUN":
                    st.subject = st.subject + w.text + " "

            if "Case=Acc" in w.feats:
                if w.upos == "PRON" or w.upos == "NOUN":
                    st.object = st.object + w.text + " "

        if w.upos == "AUX" or w.upos == "VERB" or w.upos == "PART":
            st.verb = st.verb + w.text + " "

        if w.upos == "NOUN":
            st.noun = st.noun + w.text + " "

        if w.upos == "PUNCT":
            res.append(st) 
            st = Statement()      

f = ""
for st in res:
    f = f + st.print()

print(*[f'word: {word.text}\tupos: {word.upos}\txpos: {word.xpos}\tfeats: {word.feats if word.feats else "_"}' for sent in doc.sentences for word in sent.words], sep='\n')
print(*[f'entity: {ent.text}\ttype: {ent.type}' for sent in doc.sentences for ent in sent.ents], sep='\n')
