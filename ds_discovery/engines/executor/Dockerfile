FROM python:3.9

WORKDIR /app/hadron
COPY requirements.txt /app/hadron
RUN pip install --upgrade -r requirements.txt

COPY src /app/hadron

RUN echo "hadron:x:1001:0:Default Application User:/app:/sbin/nologin" >> /etc/passwd && chown -R 1001 /app
USER hadron

CMD ["python", "hadron_executor.py"]