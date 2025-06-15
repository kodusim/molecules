#!/bin/bash

# Run migrations
python manage.py migrate --settings=config.settings_fly

# Collect static files
python manage.py collectstatic --noinput --settings=config.settings_fly

# Create superuser if it doesn't exist
python manage.py shell --settings=config.settings_fly << END
from django.contrib.auth import get_user_model
User = get_user_model()
if not User.objects.filter(username='admin').exists():
    User.objects.create_superuser('admin', 'admin@example.com', 'adminpass123')
    print('Superuser created.')
else:
    print('Superuser already exists.')
END

# Start Gunicorn
exec gunicorn config.wsgi:application \
  --bind 0.0.0.0:8080 \
  --workers 2 \
  --threads 4 \
  --timeout 120 \
  --access-logfile - \
  --error-logfile -