# Zyeta Backend Testing

This directory contains the test suite for the Zyeta backend. The tests are organized to ensure the application's reliability, functionality, and performance.

## Test Directory Structure

```
tests/
├── services/                      # Service layer tests
│   ├── test_auth.py               # Authentication service tests
│   ├── test_billing.py            # Billing service tests
│   ├── test_llm.py                # LLM model service tests
│   ├── test_org.py                # Organization service tests
│   ├── test_prompts.py            # Prompts service tests
│   ├── test_public_upload.py      # File upload service tests
│   ├── test_roles.py              # Role management service tests
│   ├── test_tools.py              # Tools service tests
│   ├── test_users.py              # User management service tests
│   └── test_ws.py                 # WebSocket service tests
├── conftest.py                    # Shared test fixtures
├── .env.test                      # Test environment variables
├── .env.test.sample               # Sample test environment file
└── README.md                      # This documentation
```

## Test Types

Our testing strategy includes:

1. **Unit Tests**: Testing individual service functions in isolation with mocked dependencies
2. **Integration Tests**: Testing service interactions with the database 
3. **Error Handling Tests**: Verifying error cases and proper exception handling
4. **Performance Tests**: Testing bulk operations and efficiency

## Running Tests

### Prerequisites

- PostgreSQL database for integration tests
- Python 3.10 or higher
- Virtual environment with dependencies installed

### Environment Setup

1. Create a test environment file:
   ```bash
   cp .env.test.sample .env.test
   ```

2. Configure your test database in `.env.test`

### Running Tests

#### All Tests
```bash
pytest
```

#### Service Tests Only
```bash
pytest tests/services/
```

#### Specific Service Tests
```bash
pytest tests/services/test_llm.py
```

#### Running Integration Tests
Some tests require a database connection and are skipped by default. To run these integration tests:

```bash
INTEGRATION_TEST=1 pytest tests/services/test_name.py
```

#### Verbose Output
For detailed test output:

```bash
pytest -v tests/services/test_name.py
```

## Test Fixtures

The main test fixtures are defined in `conftest.py`:

- `event_loop`: Event loop for async tests
- `setup_test_db`: Creates and cleans up test database schema
- `db_session`: Provides an async SQLAlchemy session
- `app`: FastAPI application instance
- `client`: FastAPI TestClient instance
- `auth_headers`: Simulated authentication headers
- `generate_id`: Utility to generate random IDs for tests

## LLM Service Tests

The LLM service tests (`tests/services/test_llm.py`) demonstrate our testing approach:

### Test Categories

1. **Unit Tests** (`TestLLMService`): 
   - Tests service functions with mocked database
   - Verifies CRUD operations, validation, and error handling

2. **Integration Tests** (`TestLLMServiceIntegration`):
   - Tests with real database operations
   - Validates data persistence and retrieval
   - Only runs when `INTEGRATION_TEST=1` is set

3. **Error Handling Tests** (`TestLLMServiceErrorHandling`):
   - Verifies input validation
   - Tests transaction rollback on errors
   - Ensures proper exception handling

4. **Performance Tests** (`TestLLMServicePerformance`):
   - Tests bulk operations
   - Validates database efficiency

### Database Testing Approach

- **Connection Management**: Sessions are properly acquired and released
- **Transaction Control**: Explicit transaction handling with proper rollback
- **Database Cleanup**: Tests clean up after themselves
- **Async/Await Patterns**: Proper handling of async database operations

## Best Practices

Our tests follow these best practices:

1. **Test Isolation**: 
   - Each test runs independently
   - Tests don't depend on state from other tests
   - Database is cleaned between test runs

2. **Error Handling**:
   - Test both happy paths and error scenarios
   - Validate exception handling
   - Ensure proper transaction management

3. **Async Testing**:
   - Use pytest-asyncio for asynchronous tests
   - Properly manage async database sessions
   - Close connections to prevent resource leaks

4. **Performance**:
   - Test bulk operations
   - Validate query efficiency
   - Measure response times for critical operations

5. **Clean Setup and Teardown**:
   - Create test data before tests
   - Clean up all test data after tests complete
   - Use fixtures for common setup operations

## Debugging Tests

For detailed logs when testing:

```bash
pytest tests/services/test_llm.py -v --log-cli-level=INFO
```

To focus on a specific test:

```bash
pytest tests/services/test_llm.py::TestLLMService::test_post_add_success -v
```

To debug database operations, enable SQL echo:

```bash
pytest tests/services/test_llm.py --echo-sql
``` 