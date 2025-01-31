# dolbo.backend

## Quick Start guide

### Env Setup:
- Create a python virtual environment
```cmd
python -m venv venv
```

- For env setup, rename the `.env.local` file to `env`.

```
APP_NAME=dolbo-backend
DATABASE_URL=<DATABASE URL>
ENVIRONMENT=development
FRONTEND_URL=<FRONTEND URL>
JWT_SECRET=<JWT SECRET>
MASTER_API_KEY=<MASTER API KEY>
OPENAI_API_KEY=<OPENAI API KEY>
RESEND_API_KEY=<RESEND API KEY>
```

### Database Setup:
Prefer using Postgres:16.

**via Local**
- Refer to [postgres installation guide](https://www.postgresql.org/download/)
- After installation, replace the DATABASE_URL in the `.env` file with this: `postgresql+asyncpg://<user>:<password>@<host>:5432/<db-name>`

> **Note:** 
> - You may experience database connectivity issues while running the application when using special characters like '@' in the database passoword. In such case rememeer to use the encoded character.\
> `For e.g. for '@' use '%40'. If your database password is 'Test@123' then use 'Test%40123'`
> - Ensure that the PostgreSQL server is running and accessible before starting the application.

**via Docker**
- Refer to the [docker installation guide](https://docs.docker.com/get-started/get-docker/)
- After installation, head over to the root directory of the project.
- Run the command:

```
docker compose up
```
 > **Note:** Ensure `docker-compose.yml` is present in the root directory. If not, then create one. Copy the following:
```yaml
services:
postgres:
    image: postgres:16.2
    container_name: postgres-container
    environment:
    POSTGRES_USER: <database-user>
    POSTGRES_PASSWORD: <database-password>
    POSTGRES_DB: <database-name>
    ports:
    - "5432:5432"
    volumes:
    - postgres_data:/var/lib/postgresql/data
    networks:
    - db_network

pgadmin:
    image: dpage/pgadmin4:latest
    container_name: pgadmin-container
    environment:
    PGADMIN_DEFAULT_EMAIL: <your-pgadmin-user-email>
    PGADMIN_DEFAULT_PASSWORD: <your-pgadmin-user-password>
    ports:
    - "8080:80"
    depends_on:
    - postgres
    volumes:
    - ./pgadmin-data:/var/lib/pgadmin
    - ./servers.json:/pgadmin4/servers.json
    networks:
    - db_network

volumes:
postgres_data:
    driver: local

networks:
db_network:
    driver: bridge
```

 > - Also, to access the database for testing purpose use default pgAdmin. For that, create a `servers.json` file in the root.
 > - Copy / paste the following configuration:
 ```json
 {
  "Servers": {
    "1": {
      "Name": "<your-preferred-name>",
      "Group": "Servers",
      "Host": "<docker-container-name-mentioned-in-docker-compose-file>",
      "Port": 5432,
      "MaintenanceDB": "<database-name>",
      "Username": "<database-username>",
      "Password": "<database-password>",
      "SSLMode": "prefer",
      "Connected": true
    }
  }
}
```


### **Application Setup**

- It is recommended to set up the application in a virtual environment to avoid conflicts with other external dependencies.

- The dependency manager used is `poetry`.

- **To install dependencies:**
  ```bash
  pip install poetry
  poetry install
  ```

- Setting up the migration tool:
    - Rename the `alembic.copy.ini` file to `alembic.ini`.
    - Set the `sqlalchemy.url`  to match the database URL in the `.env` file.
    - Run the following command to apply migrations:
    ```bash
    alembic upgrade head
    ```

- Starting the application:
    - **Using uvicorn:**
    ```bash
    uvicorn src/app:app --reload
    ```

    - **Using fastapi:**
    ```bash
    fastapi dev src/app.py
    ```

- Access application API's via this [link](http://localhost:8000/docs).

> **NOTE:**
> This quick start guide covers only the steps for setting up the application in the development phase.\
> For `SECRET_KEYS`, you can create your own custom key for `.env` file except for `RESEND_API_KEY` contact IT team.

> ⚠️ **WARNING:**
> NOT FOR PRODUCTION.