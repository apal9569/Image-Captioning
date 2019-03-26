import numpy as np
import pandas as pd
import keras
import os
import matplotlib.pyplot as plt
from keras import layers
from keras.layers import Dense, LSTM, Dropout,Input,Embedding
from keras.applications.vgg19 import VGG19,preprocess_input
from keras.models import Model
from keras.preprocessing.image import load_img,img_to_array
#from keras.applications.vgg19 import preprocess_input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import pandas as pd
import string
from collections import Counter
from copy import copy



def loadFile(flname):
    file=open(flname,'r')
    text=file.read()
    file.close()
    return text
flname="/home/ankit/data set/Flickr8k_text/Flickr8k.token.txt"
lemmatext=loadFile(flname)
#print(lemmatext)




def load_desc(lemmatext):
    intface=[]
    for line in lemmatext.split('\n'):
        word=line.split('\t')
        if len(line)<2:
            continue
        ids=word[0].split('#')
        intface.append(ids+[word[1].lower()])
        #ids=ids.split('.')[0]
        #desc=' '.join(desc)
        #if ids not in intface:
        #    intface[ids]=list()
        #intface[ids].append(desc)
        
    return intface
description=load_desc(lemmatext)
act_desc=pd.DataFrame(description,columns=["file","index","caption"])
filename=np.unique(act_desc.file.values)
Counter(Counter(act_desc.file.values).values())
#print(act_desc)

#print(description)

from pickle import dumps,loads
#res=dumps(description)
#open('/home/ankit/data set/dump.txt').write(res)

#description=loads(open('/home/ankit/data set/dump.txt').read())
#print(description.type)
#description=str(description)



def clean(description):
    descrip=description.translate(string.punctuation)
    descript=""
    for word in descrip.split():
        if len(word)>1:
            descript+=" "+word
    descriptions=""
    for word in descript.split():
        alpha=word.isalpha()
        if alpha:
            descriptions+=" "+word
    return(description)


for i,caption in enumerate(act_desc.caption.values):
    captions=clean(caption)
    act_desc["caption"].iloc[i]=captions
#print(act_desc.caption)



def start_end(cap):
    captions=[]
    for sentence in cap:
        sentence='startseq '+sentence+' endseq'
        captions.append(sentence)
    return(captions)
act_desc1=copy(act_desc)
act_desc1["caption"]=start_end(act_desc["caption"])
del act_desc
print(act_desc1["caption"])
    


model1=VGG19(weights='imagenet')
model=Model(model1.input,model1.layers[-2].output)
model.summary()



images=os.listdir("/home/ankit/data set/Flickr8k_Dataset/Flicker8k_Dataset")



encoding={}
for x,img in enumerate(images):
    file="/home/ankit/data set/Flickr8k_Dataset/Flicker8k_Dataset"+"/"+img
    image=load_img(file, target_size=(224,224,3))
    image=img_to_array(image)
    pro_image=preprocess_input(image)
    features=model.predict(pro_image.reshape((1,)+pro_image.shape[:3]))
    #features=np.reshape(features,features.shape[1])
    encoding[img]=features.flatten()



imgs,indx=[],[]
act_desc1=act_desc1.loc[act_desc1["index"].values=="0",:]
for i,file in enumerate(act_desc1.file):
    if file in encoding.keys():
        imgs.append(encoding[file])
        indx.append(i)

files=act_desc1["file"].iloc[indx].values
capt=act_desc1["caption"].iloc[indx].values
imgs=np.array(imgs)



tokens=Tokenizer(nb_words=8000)
tokens.fit_on_texts(capt)
vocab_size=len(tokens.word_index)+1
texts=tokens.texts_to_sequences(capt)



train=int(len(texts)*.75)
def tra_test(text_image_file,train):
    return(text_image_file[:train],text_image_file[train:])
text_train,text_test=tra_test(texts,train)
imgs_train,imgs_test=tra_test(imgs,train)
files_train,files_test=tra_test(files,train)



maxlen=np.max([len(i) for i in texts])
def preprocess(texts,imgs):
    assert(len(texts)==len(imgs))
    Xtxt, Ximg, Ytxt=[],[],[]
    for txt,im in zip(texts,imgs):
        in_txt,out_txt=txt[:i],txt[i]
        in_txt=pad_sequences([in_txt],maxlen=maxlen).flatten()
        out_txt=to_categorical(out_txt,num_classes=vocab_size)
        Xtxt.append(in_txt)
        Ximg.append(im)
        Ytxt.append(out_txt)
    Xtxt=np.array(Xtxt)
    Ximg=np.array(Ximg)
    Ytxt=np.array(Ytxt)
    return(Xtxt,Ximg,Yimg)
Xtxt_train,Ximg_train,Ytxt_train=preprocess(text_train,imgs_train)


emb_dim=64

img_in=Input(shape=(Ximg_train.shape[1],))
layer_1=Dense(512,activation='relu',name="ImageFeature")(img_in)
txt_in=Input(shape=(maxlen,))
text_1=Embedding(vocab_size,emb_dim,mask_zero=True)(txt_in)
lstm_1=LSTM(512,name="CaptionFeature")(text_1)

Decode_1=layers.add([text_1,layer_1])
Decode_1=Dense(512,activation='relu')(Decode_1)
out=Dense(vocab_size,activation='softmax')(Decode_1)
model=Model(inputs=[img_in,txt_in],outputs=out)

model.compile(loss='categorical_crossentropy',optimizer='RMSprop')


training=model.fit([Ximg_train,Xtxt_train],Ytxt_train,epochs=10,verbose=1,batch_size=64)

ind_word=dict([(indx,word) for word,indx in tokenize.word_index.items()])
def predict(img):
    in_text='startseq'
    for iword in range(maxlen):
        seq=tokenizer.texts_to_sequences([in_text])
        seq=pad_sequences([sequence],maxlen)
        y_pred=model.predict([img,seq],verbose=1);
        y_pred=np.argmax(y_pred)
        nword=ind_word[y_pred]
        y_txt+=" "+nword
        if nword=="endseq":
            break
    return(y_txt)


count=1
fig=plt.figure(figsize=(10,20))
for filejpg, img_feat in zip(files_test[:10],imgs_test[:10]):
    img_file="/home/ankit/data set/Flickr8k_Dataset/Flicker8k_Dataset"+'/'+filejpg
    img_load=load_img(img_file,target_size=(224,224,3))
    ax=fig.add_subplot(npic,2,count,xticks=[],yticks=[])
    ax.imshow(img_load)
    count+=1

    caption=predict(img_feat.reshape(1,len(img_feat)))
    ax=fig.add_subplot(npic,2,count)
    plt.axis('off')
    ax.plot()
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.text(0,0.5,caption,fontsize=20)
    count+=1
plt.show()


