from celery import Celery


app = Celery(
        'milvus_test',
        broker='redis://localhost:6379/0',
        backend='redis://localhost:6379/0',
        include=['deployments.tasks']
    )

app.autodiscover_tasks()

app.conf.update(
    result_expires=3600,
)
