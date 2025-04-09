# 🚀 Zyeta.io - Agentic Ecosystem

Zyeta is a powerful platform that empowers developers to create, deploy and monetize AI agents and tools. Our ecosystem connects:

- 🛠️ **Creators**: Build sophisticated AI agents and tools using our developer-friendly framework
- 💼 **Developers**: Monetize your AI creations through our marketplace
- 🔍 **Clients**: Discover and utilize high-quality AI solutions for your specific needs

With Zyeta, we're building the bridge between AI innovation and practical application, creating opportunities for developers while delivering powerful solutions to businesses and individuals.

## 🌟Key Features

- Intuitive agent & tool creation framework
- Secure deployment and testing environment
- Integrated marketplace with revenue opportunities
- Quality-assured AI solutions for diverse needs

Join Zyeta today and become part of the future of AI agent development and utilization!

## 🔧 Initial Setup

1. 💻 Use **Cursor** for development or any AI editor you like
2. 🧹 Install two extensions:
   - **Ruff** for linting
   - **Mypy** for type checking
3. 🐍 Create a virtual environment:

   ```bash
   python3.10 -m venv venv
   ```

   > 💡 Your Python version should be ≥ 3.10

4. ⚡ Activate the virtual environment:

   ```bash
   source venv/bin/activate
   ```

   > ℹ️ Different for Windows - please check online

5. 📦 Install dependencies:

   ```bash
   pip install poetry
   poetry install
   ```

## 🔍 Code Quality Setup

1. 🔄 Install pre-commits:

   ```bash
   pre-commit install
   ```

2. ✅ This project enforces rules via `.pre-commit-config.yaml`
3. ⚠️ Pre-commit **must run** on every commit or PRs will be rejected
4. 📝 Check configuration in:
   - `ruff.toml` for linting
   - `mypy.ini` for typing

## 🗃️ Database Setup

1. 🐳 Setup local database with Docker -> [link](https://docs.docker.com/engine/install/):

    ```bash
    docker run --name zyeta -e POSTGRES_PASSWORD=mysecretpassword -d postgres
    ```

2. 📄 Create configuration files:

    ```bash
    cp .env.local .env
    cp .alembic.copy.ini alembic.ini
    ```

3. ⚙️ Configure your `.env` and `alembic.ini` files
4. 🔗 Your `DATABASE_URL` should be:

    ```bash
    postgresql+asyncpg://postgres:mysecretpassword@localhost:5432/postgres
    ```

5. 🚀 Run database migrations:

    ```bash
    alembic upgrade head
    ```

6. 🎉 Start the server:

    ```bash
    fastapi dev src/app.py
    ```

## 🏝️ Sandbox Servers Setup

Sandbox servers let you test and run dynamically generated code for agents and tools. Find them in `src/servers`.

1. 📥 Install NVM:

   ```bash
   curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh | bash
   ```

2. 🟢 Install and use Node.js:

   ```bash
   nvm install 20.17
   nvm use 20.17
   ```

3. 📂 Navigate to the Python tool tester:

   ```bash
   cd src/servers/python_tool_tester
   ```

4. 📦 Install Node.js dependencies:

   ```bash
   npm install
   npm install -g tsx
   ```

5. 🚀 Start the sandbox server:

   ```bash
   tsx index.ts
   ```

   > ✨ Your server will start at <http://localhost:3000>

## 🚀 Basic Application Configuration

| Variable | Description | Example |
|----------|-------------|---------|
| `APP_NAME` | Name of the application | `zyeta.backend` |
| `ENVIRONMENT` | Current environment (dev/beta/prod) | `dev` |
| `JWT_SECRET` | Secret key for JWT authentication | *(secret value)* |
| `JWT_EXPIRE_MINUTES` | JWT token expiration time in minutes | `1400` |
| `MASTER_API_KEY` | Master key for API access | *(secret value)* |

## 🔑 API Keys

### 🤖 LLM Services

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | Authentication for OpenAI API services |
| `ANTHROPIC_API_KEY` | Authentication for Anthropic AI services |

### 📧 Communication

| Variable | Purpose |
|----------|---------|
| `RESEND_API_KEY` | For email delivery services |
| `FRONTEND_URL` | URL for frontend application |

## 🗄️ Database Configuration

```bash
DATABASE_URL=postgresql+asyncpg://postgres:mysecretpassword@localhost:5432/postgres
```

This connection string follows the format:
`postgresql+asyncpg://[username]:[password]@[host]:[port]/[database_name]`

## 💳 Payment Processing (Stripe)

| Variable | Purpose |
|----------|---------|
| `STRIPE_SECRET_KEY` | Server-side Stripe API authentication |
| `STRIPE_PUBLISHABLE_KEY` | Client-side Stripe API authentication |
| `STRIPE_WEBHOOK_SECRET` | Verifies Stripe webhook events |

## 📚 Storage Configuration

### 🪣 S3 Storage

| Variable | Description |
|----------|-------------|
| `S3_BUCKET` | Main storage bucket name (`zyeta-dev`) |
| `S3_ACCESS_KEY` | S3 access credentials |
| `S3_SECRET_KEY` | S3 secret credentials |
| `S3_ENDPOINT` | S3 service endpoint |
| `PUBLIC_S3_BUCKET` | Public assets bucket |

## 🔄 Task Processing

### 🥬 Celery Configuration

| Variable | Purpose |
|----------|---------|
| `CELERY_BROKER_URL` | Message broker URL for Celery tasks |
| `CELERY_RESULT_BACKEND` | Backend storage for Celery results |

## 🏖️ Testing Environment

| Variable | Purpose |
|----------|---------|
| `PYTHON_SANDBOX_TESTING_URL` | URL for Python sandbox testing service |
| `KB_SETTINGS_VERSION` | Knowledge base settings version |
| `FIRECRAWL_API_KEY` | Authentication for Firecrawl service |

## 🛠️ Setting Up Your Environment

1. Copy `.env.local` to create your own `.env` file
2. Fill in all the required values
3. Make sure your database connection string matches your setup
4. Keep your API keys secure and never commit them to version control!

> 💡 **Pro Tip**: Make sure your local PostgreSQL instance is running before starting the application!
