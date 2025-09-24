FROM python:3.11-slim AS build
WORKDIR /app

COPY . /app

RUN pip install --upgrade pip   && \
    pip install -r requirements

RUN mkdocs build

FROM nginx:alpine

COPY --from=build --chown=185 /app/site/ /usr/share/nginx/html/

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
