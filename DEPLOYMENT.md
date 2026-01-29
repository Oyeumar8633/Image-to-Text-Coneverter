# Deployment Guide for Image-to-Word Converter

This guide provides step-by-step instructions for deploying the Streamlit application to various platforms.

## Prerequisites

- Your code is in a Git repository (GitHub, GitLab, etc.)
- You have accounts for the deployment platform of your choice

---

## Option 1: Streamlit Cloud (Recommended - Free & Easy)

### Why Streamlit Cloud?
- ✅ Free tier available
- ✅ Automatic deployments from GitHub
- ✅ No server management needed
- ✅ Easy to set up

### Steps:

1. **Push your code to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account
   - Click **"New app"**
   - Fill in:
     - **Repository**: Select your GitHub repo
     - **Branch**: `main` (or your default branch)
     - **Main file path**: `app.py`
   - Click **"Deploy"**

3. **Wait for deployment** (usually 1-2 minutes)

4. **Access your app** at: `https://your-app-name.streamlit.app`

### Important Notes:
- Models (PaddleOCR/EasyOCR) will download on first use (~100MB each)
- Free tier has resource limits but sufficient for most use cases
- App may sleep after inactivity but wakes up automatically

---

## Option 2: Heroku

### Steps:

1. **Install Heroku CLI**: [heroku.com/cli](https://devcenter.heroku.com/articles/heroku-cli)

2. **Create `Procfile`** (in project root):
   ```
   web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```

3. **Create `setup.sh`** (in project root):
   ```bash
   mkdir -p ~/.streamlit/
   echo "\
   [server]\n\
   headless = true\n\
   port = $PORT\n\
   enableCORS = false\n\
   \n\
   " > ~/.streamlit/config.toml
   ```

4. **Update `requirements.txt`** to include:
   ```
   gunicorn
   ```

5. **Deploy**:
   ```bash
   heroku login
   heroku create your-app-name
   git push heroku main
   heroku open
   ```

---

## Option 3: Docker Deployment

### Steps:

1. **Create `Dockerfile`** (in project root):
   ```dockerfile
   FROM python:3.9-slim

   WORKDIR /app

   # Install system dependencies
   RUN apt-get update && apt-get install -y \
       tesseract-ocr \
       && rm -rf /var/lib/apt/lists/*

   # Copy requirements and install Python packages
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt

   # Copy application code
   COPY . .

   # Expose port
   EXPOSE 8501

   # Health check
   HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

   # Run the app
   ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
   ```

2. **Create `.dockerignore`**:
   ```
   venv/
   __pycache__/
   *.pyc
   .git/
   .gitignore
   ```

3. **Build and run**:
   ```bash
   docker build -t image-to-word-converter .
   docker run -p 8501:8501 image-to-word-converter
   ```

4. **Access at**: `http://localhost:8501`

### For Cloud Docker Services:
- **AWS ECS/Fargate**: Use the Dockerfile above
- **Google Cloud Run**: Deploy container directly
- **Azure Container Instances**: Use the Dockerfile above

---

## Option 4: Local Network Sharing

### Steps:

1. **Find your local IP address**:
   - **macOS/Linux**: `ifconfig | grep "inet "`
   - **Windows**: `ipconfig`

2. **Run Streamlit with network access**:
   ```bash
   streamlit run app.py --server.address=0.0.0.0
   ```

3. **Access from other devices** on the same network:
   - Use: `http://YOUR_IP_ADDRESS:8501`
   - Example: `http://192.168.1.100:8501`

---

## Option 5: VPS/Cloud Server (AWS, DigitalOcean, etc.)

### Steps:

1. **Set up your server** (Ubuntu/Debian recommended)

2. **Install dependencies**:
   ```bash
   sudo apt-get update
   sudo apt-get install -y python3 python3-pip tesseract-ocr
   ```

3. **Clone your repository**:
   ```bash
   git clone <your-repo-url>
   cd PPIT-Project
   ```

4. **Set up virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

5. **Run with systemd service** (create `/etc/systemd/system/streamlit-app.service`):
   ```ini
   [Unit]
   Description=Streamlit Image-to-Word Converter
   After=network.target

   [Service]
   Type=simple
   User=your-username
   WorkingDirectory=/path/to/PPIT-Project
   Environment="PATH=/path/to/PPIT-Project/venv/bin"
   ExecStart=/path/to/PPIT-Project/venv/bin/streamlit run app.py --server.port=8501 --server.address=0.0.0.0
   Restart=always

   [Install]
   WantedBy=multi-user.target
   ```

6. **Start the service**:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable streamlit-app
   sudo systemctl start streamlit-app
   ```

7. **Set up Nginx reverse proxy** (optional, for HTTPS):
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;

       location / {
           proxy_pass http://localhost:8501;
           proxy_http_version 1.1;
           proxy_set_header Upgrade $http_upgrade;
           proxy_set_header Connection "upgrade";
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
       }
   }
   ```

---

## Important Deployment Considerations

### 1. Model Storage
- PaddleOCR models: `~/.paddlex/` (~100MB)
- EasyOCR models: `~/.EasyOCR/` (~100MB)
- **For Docker**: Pre-download models in the image or use volumes
- **For Cloud**: Models download on first use (may take a few minutes)

### 2. Environment Variables
Create a `.streamlit/config.toml` for production:
```toml
[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false
```

### 3. Security
- Add file size limits in `app.py`:
  ```python
  st.file_uploader(..., max_upload_size=10)  # 10MB limit
  ```
- Implement authentication if needed
- Use HTTPS in production
- Sanitize file inputs

### 4. Performance
- First OCR request is slower (model loading)
- Consider using GPU for faster processing
- Implement caching for frequently processed images
- Use CDN for static assets

### 5. Monitoring
- Set up logging
- Monitor resource usage
- Track errors and performance metrics

---

## Quick Comparison

| Platform | Cost | Ease | Best For |
|----------|------|------|----------|
| Streamlit Cloud | Free | ⭐⭐⭐⭐⭐ | Quick deployment, demos |
| Heroku | Free tier | ⭐⭐⭐⭐ | Small projects |
| Docker | Varies | ⭐⭐⭐ | Full control, scaling |
| VPS | $5-20/mo | ⭐⭐ | Production, custom setup |

---

## Troubleshooting

### Models not downloading:
- Check internet connection
- Verify disk space
- Check firewall settings

### App crashes on first OCR:
- Increase memory allocation
- Check logs for specific errors
- Verify all dependencies installed

### Slow performance:
- Use GPU if available
- Reduce image size before processing
- Implement caching

---

## Support

For issues or questions:
1. Check the main `PHASE1_REPORT.md`
2. Review Streamlit documentation: [docs.streamlit.io](https://docs.streamlit.io)
3. Check PaddleOCR/EasyOCR documentation

---

**Last Updated**: Phase 1 Completion
