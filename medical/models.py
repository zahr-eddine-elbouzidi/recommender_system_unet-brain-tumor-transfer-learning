from unicodedata import category
from django.conf import settings
from django.db import models
from django.conf.urls.static import static
from django.contrib.auth.models import User
from django.utils.html import mark_safe

# Create your models here.


class Category(models.Model):
    category_id = models.AutoField(primary_key=True)
    la_category = models.CharField(max_length=255)

class Item(models.Model):
    item_id = models.AutoField(primary_key=True)
    category = models.CharField(max_length=255)
    content = models.CharField(max_length=1000)



class Diagnosis(models.Model):
    diagnosis_id = models.AutoField(primary_key=True)
    user_id = models.IntegerField()
    item_id = models.IntegerField()
    nbr_propositions = models.IntegerField()

    def get_username(self):
        return ",".join([str(p.username) for p in self.user_id.all()])
    

    def get_category_item(self):
        return ",".join([str(p.category) for p in self.item_id.all()])


    def __unicode__(self):
        return "{0}".format(self.nbr_propositions)



class Prediction(models.Model):
    pred_id = models.AutoField(primary_key=True)
    content = models.CharField(max_length=1000)
    image = models.ImageField(upload_to='photos/')

    def image_tag(self):
        return mark_safe('<img src="{}" width="100" />'.format(self.image.url))
    image_tag.short_description = 'Image'
    image_tag.allow_tags = True

    def __str__(self):
        return self.content