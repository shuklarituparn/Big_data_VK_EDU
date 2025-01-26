FROM ubuntu:20.04


COPY task.sh .
RUN chmod +x task.sh

ENTRYPOINT ["/opt/hadoop/task.sh"]