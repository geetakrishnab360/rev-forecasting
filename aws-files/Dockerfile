FROM public.ecr.aws/lambda/python:3.12
COPY requirements.txt ${LAMBDA_TASK_ROOT}
RUN pip install -r requirements.txt
COPY aws-checkrunmodels.py ${LAMBDA_TASK_ROOT}
CMD ["aws-checkrunmodels.lambda_handler"]