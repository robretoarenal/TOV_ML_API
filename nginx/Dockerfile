FROM nginx:1.19.6

RUN echo "nginx Before"
RUN rm /etc/nginx/nginx.conf
COPY nginx.conf /etc/nginx/
RUN echo "nginx Copy"
RUN rm /etc/nginx/conf.d/default.conf
COPY project.conf /etc/nginx/conf.d/
RUN echo "nginx End"