# Použití oficiálního Python image
FROM python:3.11

# Create the app directory
RUN mkdir /app

# Nastavení pracovního adresáře
WORKDIR /mysite/mysite

# Zkopírování souborů aplikace
COPY requirements.txt .

# Instalace závislostí
RUN pip install --no-cache-dir -r requirements.txt

# Zkopírování celého projektu
COPY . /mysite

# Exponování portu
EXPOSE 8000

# Spuštění serveru
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
