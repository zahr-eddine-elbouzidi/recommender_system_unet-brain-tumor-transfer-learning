

from django.contrib.admin.widgets import AdminFileWidget
from django.utils.safestring import mark_safe
from django.contrib import admin

class ImageWidget(AdminFileWidget):
    def render(self, name, value, attrs=None):
        output = []
        if value and getattr(value, "url", None):
            image_url = value.url
            file_name = str(value)
            output.append(u' <a href="%s" target="_blank"><img src="%s" alt="%s" /></a>' % \
                          (image_url, image_url, file_name))
        output.append(super(AdminFileWidget, self).render(name, value, attrs))
        return mark_safe(u''.join(output))


class ImageWidgetAdmin(admin.ModelAdmin):
    image_fields = []

    def formfield_for_dbfield(self, db_field, **kwargs):
        if db_field.name in self.image_fields:
            request = kwargs.pop("request", None)
            kwargs['widget'] = ImageWidget
            return db_field.formfield(**kwargs)
        return super(ImageWidgetAdmin, self).formfield_for_dbfield(db_field, **kwargs)