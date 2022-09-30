from unicodedata import category
#from attr import field
from django.contrib import admin
from django.contrib.auth.models import User
from .models import Diagnosis, Item, Prediction , Category
import admin_thumbnails
from django.contrib import messages
from django.http import HttpResponseRedirect
from django.urls import reverse
# Register your models here.
from django.urls import path
from django.shortcuts import render
from django import forms
import pandas as pd
import numpy as np

#import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
#import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import cv2
from django.core.files.storage import FileSystemStorage
import os 




class CsvImportForm(forms.Form):
    image_upload = forms.FileField()



EMBEDDING_SIZE = 50

class RecommenderNet(keras.Model):

    def __init__(self, num_users, num_items, embedding_size, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_size = embedding_size
        self.user_embedding = layers.Embedding(
            num_users,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.user_bias = layers.Embedding(num_users, 1)
        self.item_embedding = layers.Embedding(
            num_items,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.item_bias = layers.Embedding(num_items, 1)
        
        #model.add(layers.LSTM(10 , dropout= 0.1))
     
    
    

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        item_vector = self.item_embedding(inputs[:, 1])
        item_bias = self.item_bias(inputs[:, 1])
        dot_user_item = tf.tensordot(user_vector, item_vector, 2)
        # Add all the components (including bias)
        x = dot_user_item + user_bias + item_bias
        # The sigmoid activation forces the rating to between 0 and 1
        return tf.nn.sigmoid(x)

    def getModel(self , num_users , num_items , EMBEDDING_SIZE = 50):

        model = RecommenderNet(num_users, num_items, EMBEDDING_SIZE)
        model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
        )

        return model


    

class ItemAdmin(admin.ModelAdmin):
    list_display = ('category', 'content')



 

    def get_urls(self):
        urls = super().get_urls()
        new_urls = [path('upload-csv/', self.upload_csv),]
        return new_urls + urls

    def upload_csv(self, request):

        if request.method == "POST":

            ##print("current user =  ",request.user.is_authenticated)
            ##print("current user ID =  ",request.user.id)
    

            upload = request.FILES['image_upload']
            fss = FileSystemStorage()
            #file = fss.save(upload.name, upload)
            #file_url = fss.url(file)
            image_path = upload.name
            if not os.path.exists('media/'+image_path):
                file = fss.save(upload.name, upload)
                file_url = fss.url(file)




            trained_model = load_model('static/brain_tumor_detection.h5', compile=False) #or compile = False
            #print(trained_model.summary())
            img = cv2.imread('media/'+image_path)

            opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            img = cv2.resize(opencvImage,(150,150))
            img = img.reshape(1,150,150,3)
            p = trained_model.predict(img)
            p = np.argmax(p,axis=1)[0]
            message = ""
            drap = -1
            if p==0:
                p='Glioma'
                drap = 0
            elif p==1:
                message = 'MRI - The model predicts that there is No Tumor'
                drap = -1
            elif p==2:
                p='Meningioma'
                drap = 0
            else:
                p='Pituitary'
                drap = 0
            if p!=1:
                message = 'MRI image / The Model predicts that it is a  - '+p


            #get all objects
            a = Category.objects.all()
            for ii in a:
                print(ii.category_id)

            #get users dataframe  
            users = User.objects.all()
            user_df = pd.DataFrame(columns = ['user_id' , 'name' , 'email' , 'password'])
            for user in users:
                user_df = user_df.append({'user_id':user.id , 'name': user.username ,'email': user.email ,'password' : user.password  }, ignore_index=True)
    
            #print(user_df.head(10))
            #get items dataframe
            items = Item.objects.all()
            item_df = pd.DataFrame(columns = ['item_id' , 'category' , 'content'])
            for item in items:
                item_df = item_df.append({'item_id':item.item_id , 'category': item.category ,'content': item.content }, ignore_index=True)
    
            #print(item_df.head(10))

            #diagnosis dataframe
            diagnosises = Diagnosis.objects.all()
            disgnosis_df = pd.DataFrame(columns = ['diagnosis_id' , 'user_id' , 'item_id' , 'nbr_propositions'])
            for diag in diagnosises:
                disgnosis_df = disgnosis_df.append({'diagnosis_id':diag.diagnosis_id , 'user_id': diag.user_id ,'item_id': diag.item_id ,'nbr_propositions': diag.nbr_propositions }, ignore_index=True)
    
            #print(disgnosis_df.head(10))

            #get user and item ids 
            user_ids = disgnosis_df["user_id"].unique().tolist()
            item_ids = disgnosis_df["item_id"].unique().tolist()


            #encoded users
            user2user_encoded = {x: i for i, x in enumerate(user_ids)}
            userencoded2user = {i: x for i, x in enumerate(user_ids)}

            #print(userencoded2user)

            #encoded items
            item2item_encoded = {x: i for i, x in enumerate(item_ids)}
            item_encoded2item = {i: x for i, x in enumerate(item_ids)}

            #print(item2item_encoded)

            #set disgnosis_df dataframe
            disgnosis_df["user"] = disgnosis_df["user_id"].map(user2user_encoded)
            disgnosis_df["item"] = disgnosis_df["item_id"].map(item2item_encoded)


            #get users len
            num_users = len(user2user_encoded)
            #get items len 
            num_items = len(item_encoded2item)


            #set nbr_propositions value 
            disgnosis_df["nbr_propositions"] = disgnosis_df["nbr_propositions"].values.astype(np.float32)


            # min and max ratings will be used to normalize the nbre of propositions later
            min_propositions = min(disgnosis_df["nbr_propositions"])
            max_propositions = max(disgnosis_df["nbr_propositions"])

            print(
                "Number of users: {}, Number of Items: {}, Min Proposition: {}, Max Proposition: {}".format(
                    num_users, num_items, min_propositions, max_propositions
                )
            )


            x = disgnosis_df[["user", "item"]].values
            # Normalize the targets between 0 and 1. Makes it easy to train.
            y = disgnosis_df["nbr_propositions"].apply(lambda x: (x - min_propositions) / (max_propositions - min_propositions)).values
            # Assuming training on 90% of the data and validating on 10%.
            train_indices = int(0.9 * disgnosis_df.shape[0])
            x_train, x_val, y_train, y_val = (
                x[:train_indices],
                x[train_indices:],
                y[:train_indices],
                y[train_indices:],
            )

            #print(x_train, x_val, y_train, y_val)

            #train model 
            m_instance = RecommenderNet(num_users , num_items , 50)
            model = m_instance.getModel(num_users , num_items)
            #print(model)
            history = model.fit(
                x=x_train,
                y=y_train,
                batch_size=64,
                epochs=30,
                verbose=1,
                validation_data=(x_val, y_val),
            )

            #print(history)
            #end train model 

            #test model 
            #user_id = request.user.id
            user_id = 4

            df_random = disgnosis_df.sample(frac=1, random_state=42)
    

            #print("user id : ", user_id)

            items_proposed_by_user = disgnosis_df[df_random.user_id == user_id]


            items_not_proposed = item_df[~item_df["item_id"].isin(items_proposed_by_user.item_id.values)]["item_id"]



            items_not_proposed = list(set(items_not_proposed).intersection(set(item2item_encoded.keys())))


            items_not_proposed = [[item2item_encoded.get(x)] for x in items_not_proposed]



            user_encoder = user2user_encoded.get(user_id)


            user_item_array = np.hstack(([[user_encoder]] * len(items_not_proposed), items_not_proposed))


            propositions = model.predict(user_item_array).flatten()


            #top 10 propositions
            top_propositions_indices = propositions.argsort()[-5:][::-1]


            #recommended item ids

            recommended_items_ids = [
                item_encoded2item.get(items_not_proposed[x][0]) for x in top_propositions_indices
            ]


            #print(recommended_items_ids)

            '''print("Showing recommendations for user: {}".format(user_id))
            print("====" * 9)
            print("Items with high ratings from user")
            print("----" * 8)'''

            top_items_user = (
                items_proposed_by_user.sort_values(by="nbr_propositions", ascending=False)
                .head(5)
                .item_id.values
            )
            item_df_rows = item_df[item_df["item_id"].isin(top_items_user)]

            for row in item_df_rows.itertuples():
                print(row.category , ':', row.content, 'Rating:',(disgnosis_df[(disgnosis_df['item_id'] == row.item_id) & (disgnosis_df['user_id'] == user_id)].nbr_propositions), "\n" )
                print(row)

            '''print("----" * 8)
            print("Top 10 items cancer diagnosis recommendations")
            print("----" * 8)'''

            top_diag = ""
            tab = []
            recommended_items = item_df[item_df["item_id"].isin(recommended_items_ids)]
            ind = 0
            for row in recommended_items.itertuples():
                ind = ind + 1
                tab.append({'category' : row.category , 'content' : row.content})
                #print(row.category, ":", row.content,'\n')
                if ind == 1:
                    top_diag =  row.content
                print(row)



            #showing recommendations 
 

            csv_file = request.FILES["image_upload"]


            if not csv_file.name.endswith('.jpg'):
                messages.warning(request, 'The wrong file type was uploaded')
                return HttpResponseRedirect(request.path_info)

            #file_data = csv_file.read().decode("utf-8")
            #csv_data = file_data.split("\n")

            if request.POST.get('type') == "Items":

                dataTsv = pd.read_csv('static/items.csv', sep = ",")
                df =  pd.DataFrame(dataTsv)
                #print(df.head(20))

                for x in df.index:
                    print(df['item_id'][x] , df['category'][x] , df['content'][x])
                    if Category.objects.filter(la_category__icontains= df['category'][x]):
                        pass
                    else:
                        pass
                        '''category_object, created_cat = Category.objects.get_or_create(
                            category_id= df['item_id'][x],
                            la_category = df['category'][x],
                        )'''
                    '''created = Item.objects.update_or_create(
                                item_id= df['item_id'][x],
                                category = df['category'][x],
                                content = df['content'][x])'''

            elif request.POST.get('type') == "Diagnosis":
                dataDiag = pd.read_csv('static/diagnosis.csv', sep = ",")
                df_diag =  pd.DataFrame(dataDiag)
                for i in df_diag.index:

                    '''created_diag = Diagnosis.objects.update_or_create(
                                diagnosis_id= df_diag['diagnosis_id'][i],
                                item_id=  df_diag['item_id'][i],
                                user_id= df_diag['user_id'][i] ,
                                nbr_propositions= df_diag['nbr_propositions'][i])'''

            url = reverse('admin:index')
            form = CsvImportForm()
            if drap == 0:
                dataRecommendation = {'recommendations' : tab , "form": form , 'message' : message 
                , 'drap' : drap , 'image_path' : image_path , 'top_diag' : top_diag}
            else:
                dataRecommendation = {'recommendations' : [] , "form": form , 'message' : message 
                , 'drap' : drap , 'image_path' : image_path , 'top_diag' : top_diag}
            #return HttpResponseRedirect(url)
            return render(request, "admin/csv_upload.html", dataRecommendation)
        form = CsvImportForm()
        data = {"form": form}
        return render(request, "admin/csv_upload.html", data)
     


@admin_thumbnails.thumbnail('image')
class PredictionAdmin(admin.ModelAdmin):
    list_display = ('image', 'image_tag', 'content')
    fields = ['image' ,'content','image_thumbnail']
    editable = ('content')
 
      
class DiagnosisAdmin(admin.ModelAdmin):
    list_display = ('user_id','item_id','nbr_propositions')



class CategoryAdmin(admin.ModelAdmin):
    list_display = ('category_id','la_category')



admin.site.register(Category , CategoryAdmin)
admin.site.register(Item  , ItemAdmin)
admin.site.register(Diagnosis , DiagnosisAdmin)
admin.site.register(Prediction, PredictionAdmin)
