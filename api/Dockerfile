FROM public.ecr.aws/lambda/python:3.10

# Install pysqlite3-binary to override the default sqlite3 library with SQLite >= 3.35.0
RUN pip install pysqlite3-binary

COPY handler.py rag_pipeline.py prepare.py requirements.txt /var/task/
COPY VectorDB /var/task/VectorDB

RUN pip install -r requirements.txt

CMD ["handler.lambda_handler"]
