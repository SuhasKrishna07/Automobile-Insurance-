# Generated by Django 5.2 on 2025-04-07 12:04

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('myapp', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='candidate',
            name='Police_reports',
        ),
    ]
