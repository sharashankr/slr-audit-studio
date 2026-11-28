FROM python:3.11-slim

WORKDIR /app
COPY requirements-render.txt .
RUN pip install --no-cache-dir -r requirements-render.txt

COPY . .
ENV SLR_PDF_PARSER=pdfium
EXPOSE 10000
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}"]
