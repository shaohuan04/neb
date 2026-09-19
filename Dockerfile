FROM python:3.11-slim

WORKDIR /app

# Install deps first so this layer caches across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Cloud Run injects PORT; default to 8080 for local docker runs.
ENV PORT=8080
EXPOSE 8080

# XSRF protection is disabled because Cloud Run terminates TLS at its proxy,
# which breaks Streamlit's file-upload token check. This app takes uploads and
# holds no user accounts or secrets, so the tradeoff is acceptable here.
CMD streamlit run app.py \
    --server.port=$PORT \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --browser.gatherUsageStats=false
