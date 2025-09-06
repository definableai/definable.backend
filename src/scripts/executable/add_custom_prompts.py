#!/usr/bin/env python3
"""
This script scrapes cursor.directory for AI prompts and imports them into the database.
It populates both prompt_categories and prompts tables with structured, categorized data.
"""

import asyncio
import os
import sys
from typing import List, Optional, Tuple
from dataclasses import dataclass

import aiohttp
from bs4 import BeautifulSoup, Tag
from sqlalchemy.sql import text
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import uuid4


# Add the parent directory to the path so we can import from src
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if parent_dir not in sys.path:
  sys.path.insert(0, parent_dir)

from scripts.core.base_script import BaseScript
from common.logger import log as logger

# CONFIGURATION CONSTANTS

# Target website configuration
BASE_URL = "https://cursor.directory"
RULES_ENDPOINT = "/rules"

# Performance tuning constants (conservative for external site politeness)
REQUEST_TIMEOUT = 15  # Seconds to wait for HTTP responses
PARALLEL_REQUESTS = 2  # Concurrent HTTP requests (REDUCED for server politeness)
RETRY_ATTEMPTS = 3  # Number of retry attempts for failed requests
BACKOFF_FACTOR = 2  # Exponential backoff multiplier (seconds)
REQUEST_DELAY = 0.5  # Minimum delay between requests (seconds) for politeness
USE_SEQUENTIAL_FALLBACK = True  # Fall back to sequential processing if too many failures

# Content validation settings
MIN_PROMPT_LENGTH = 10  # Minimum characters for valid prompt content
MAX_EMPTY_PROMPTS_RATIO = 0.5  # Maximum ratio of empty prompts before warning

# CSS Selectors for web scraping (cursor.directory specific)
SELECTORS = {
  "category_links": "body > div.flex.w-full.h-full > div.hidden > aside > div > div > div > div > a",
  "prompt_container": "main > section > div:nth-of-type(2)",
  "code_elements": "div > div > a > div > code",
}


@dataclass
class CategoryInfo:
  """Data class for category information."""

  name: str
  description: str
  icon_url: str
  display_order: int


@dataclass
class PromptInfo:
  """Data class for prompt information."""

  title: str
  content: str
  description: str
  category_id: int


async def check_existing_data(db: AsyncSession) -> Tuple[bool, bool]:
  """
  Check if tables exist and contain data to prevent duplicates.

  Returns:
      Tuple[bool, bool]: (categories_exist, prompts_exist)
  """
  try:
    # Check if tables exist and have data
    category_result = await db.execute(text("SELECT COUNT(*) FROM prompt_categories"))
    prompt_result = await db.execute(text("SELECT COUNT(*) FROM prompts"))

    categories_count = category_result.scalar() or 0
    prompts_count = prompt_result.scalar() or 0

    return categories_count > 0, prompts_count > 0

  except Exception as e:
    logger.warning(f"Tables may not exist yet: {e}")
    return False, False


async def fetch_html_content(session: aiohttp.ClientSession, url: str, max_retries: int = RETRY_ATTEMPTS) -> Optional[str]:
  """
  Fetch HTML content with retry logic and proper error handling.

  Args:
      session: aiohttp ClientSession for connection reuse
      url: URL to fetch
      max_retries: Maximum number of retry attempts

  Returns:
      Optional[str]: HTML content or None if failed
  """
  for attempt in range(max_retries):
    try:
      async with session.get(url, timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)) as response:
        if response.status == 200:
          return await response.text()
        elif response.status == 429:  # Rate limited
          retry_after = response.headers.get("Retry-After", str(BACKOFF_FACTOR * (2**attempt)))
          logger.warning(f"Rate limited by {url}. Retry after {retry_after}s, attempt {attempt + 1}/{max_retries}")
          if attempt < max_retries - 1:
            await asyncio.sleep(float(retry_after))
            continue
        else:
          logger.warning(f"HTTP {response.status} for {url}, attempt {attempt + 1}/{max_retries}")

    except asyncio.TimeoutError:
      logger.warning(f"Timeout fetching {url}, attempt {attempt + 1}/{max_retries}")
    except aiohttp.ClientError as e:
      logger.warning(f"Client error fetching {url}: {e}, attempt {attempt + 1}/{max_retries}")
    except Exception as e:
      logger.error(f"Unexpected error fetching {url}: {e}")
      break

    if attempt < max_retries - 1:
      # Exponential backoff with longer delays for rate limiting respect
      backoff_delay = BACKOFF_FACTOR * (2**attempt)  # 2, 4, 8 seconds
      logger.info(f"Retrying {url} in {backoff_delay} seconds...")
      await asyncio.sleep(backoff_delay)

  logger.error(f"Failed to fetch {url} after {max_retries} attempts")
  return None


async def create_new_organization_user(db: AsyncSession) -> Tuple[Optional[str], Optional[str]]:
  """
  Create a new organization and user.

  Returns:
      Tuple[Optional[str], Optional[str]]: (organization_id, user_id)
  """

  try:
    # 1. Get the org_id and role_id
    result = await db.execute(text("SELECT id FROM organizations WHERE name = 'Default Org' UNION SELECT id FROM roles WHERE name = 'admin'"))
    row = result.scalars().all()

    if row:
      org_id = str(row[0])
      role_id = str(row[1])

      # 2. Create a new user
      admin_email = "temp-admin@definable.ai"
      admin_stytch_id = f"user-test-{uuid4()}"
      admin_first_name = "Admin"

      query = text("""
          INSERT INTO users (email, stytch_id, first_name) 
          VALUES (:email, :stytch_id, :first_name)
          RETURNING id
      """)

      result = await db.execute(query, {"email": admin_email, "stytch_id": admin_stytch_id, "first_name": admin_first_name})
      admin_user_id = result.scalar_one()
      await db.commit()

      # 3. Add the user into organization table
      query = text("""
        INSERT INTO organization_members (organization_id, user_id, role_id, status)
        VALUES
        (:organization_id, :user_id, :role_id, :status)
      """)

      await db.execute(query, {"organization_id": org_id, "user_id": admin_user_id, "role_id": role_id, "status": "active"})
      await db.commit()

      return org_id, admin_user_id

    else:
      logger.error("No organization found in database")
      return None, None

  except Exception as e:
    logger.error(f"Failed to create new organization and user: {e}")
    return None, None


async def extract_categories_from_html(html_content: str) -> List[CategoryInfo]:
  """
  Parse HTML content and extract category information.

  Args:
      html_content: HTML content from the rules page

  Returns:
      List[CategoryInfo]: List of parsed category information
  """
  soup = BeautifulSoup(html_content, "html5lib")
  directories = soup.select(SELECTORS["category_links"])

  categories: List[CategoryInfo] = []
  for index, directory in enumerate(directories):
    try:
      # Safely extract button content
      if not (directory.button and directory.button.contents):
        logger.debug(f"Skipping directory element with missing button content at index {index}")
        continue

      name = str(directory.button.contents[0]).strip()
      href_raw = directory.get("href", "")

      # Ensure href is a string (BeautifulSoup can return AttributeValueList)
      href = str(href_raw) if href_raw else ""

      if not href:
        logger.debug(f"Skipping directory element with missing href at index {index}")
        continue

      categories.append(
        CategoryInfo(name=name, description=f"Custom instructions for {href.split('/')[-1]}", icon_url=href, display_order=len(categories) + 1)
      )

    except Exception as e:
      logger.warning(f"Error processing directory element at index {index}: {e}")
      continue

  logger.info(f"Successfully extracted {len(categories)} categories")
  return categories


async def extract_prompts_from_category(session: aiohttp.ClientSession, category: CategoryInfo, base_url: str) -> Optional[str]:
  """
  Extract prompts content for a specific category.

  Args:
      session: aiohttp ClientSession for HTTP requests
      category: Category information
      base_url: Base URL for requests

  Returns:
      Optional[str]: Combined prompt content or None if failed
  """
  prompt_url = base_url + category.icon_url
  logger.info(f"Processing category: {category.name}")

  html_content = await fetch_html_content(session, prompt_url)
  if not html_content:
    return None

  soup = BeautifulSoup(html_content, "html5lib")
  directory_elements = soup.select(SELECTORS["prompt_container"])

  if not directory_elements:
    logger.warning(f"No prompt container found for category: {category.name}")
    return None

  directory = directory_elements[0]
  final_prompts = []

  # Extract prompts from all child elements
  for prompt in directory.children:
    try:
      # Only process Tag elements that have the select method
      if not isinstance(prompt, Tag):
        continue

      code_elements = prompt.select(SELECTORS["code_elements"])
      if code_elements and code_elements[0].string:
        prompt_content = code_elements[0].string.strip()
        # Only add prompts that meet minimum length requirement
        if len(prompt_content) >= MIN_PROMPT_LENGTH:
          final_prompts.append(prompt_content)
    except Exception as e:
      logger.debug(f"Error extracting prompt in category {category.name}: {e}")
      continue

  if final_prompts:
    logger.info(f"Extracted {len(final_prompts)} prompts from category: {category.name}")
    return "\n".join(final_prompts)
  else:
    logger.warning(f"No prompts found in category: {category.name}")
    return None


async def populate_prompt_categories_and_prompts(base_url: str, db: AsyncSession) -> None:
  """
  Main function to populate prompt categories and prompts with improved performance.

  Uses async HTTP requests and parallel processing for better performance.
  Includes duplicate checking and proper error handling.

  Args:
      base_url: Base URL for cursor.directory
      db: Database session
  """

  # Check for existing data to prevent duplicates
  categories_exist, prompts_exist = await check_existing_data(db)
  if categories_exist and prompts_exist:
    logger.info("Prompt categories and prompts already exist. Skipping population.")
    return

  # Get organization information
  organization_id, user_id = await create_new_organization_user(db)
  if not organization_id or not user_id:
    raise Exception("No organization_id or user_id found in database")

  # Create aiohttp session with requests-like headers for compatibility
  headers = {
    "User-Agent": "Mozilla/5.0 (compatible; DefinableAI-Scripts/2.0)"  # Identify ourselves politely
  }
  connector = aiohttp.TCPConnector(limit=PARALLEL_REQUESTS, limit_per_host=PARALLEL_REQUESTS)
  timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)

  async with aiohttp.ClientSession(
    connector=connector,
    timeout=timeout,
    headers=headers,
    trust_env=True,  # Respect proxy settings like requests
  ) as session:
    # Step 1: Fetch and parse categories
    logger.info("Fetching categories from cursor.directory...")
    html_content = await fetch_html_content(session, base_url + RULES_ENDPOINT)

    if not html_content:
      raise Exception(f"Failed to fetch categories from {base_url}{RULES_ENDPOINT}")

    categories = await extract_categories_from_html(html_content)
    if not categories:
      raise Exception("No categories found on the page")

    # Step 2: Insert categories into database (if not exists)
    if not categories_exist:
      logger.info(f"Inserting {len(categories)} categories into database...")
      category_data = [
        {"name": cat.name, "description": cat.description, "icon_url": cat.icon_url, "display_order": cat.display_order} for cat in categories
      ]

      await db.execute(
        text("INSERT INTO prompt_categories (name, description, icon_url, display_order) VALUES (:name, :description, :icon_url, :display_order)"),
        category_data,
      )
      await db.commit()  # Commit categories before processing prompts
      logger.info("Categories inserted successfully")

    # Step 3: Process prompts with parallel HTTP but sequential database access
    if not prompts_exist:
      logger.info("Processing prompts with parallel HTTP requests...")

      # Step 3a: Parallel HTTP requests to fetch prompt content
      async def fetch_category_content(category: CategoryInfo) -> Optional[Tuple[CategoryInfo, str]]:
        """Fetch content for a single category (HTTP only, no DB access)."""
        prompt_content = await extract_prompts_from_category(session, category, base_url)
        if prompt_content:
          return (category, prompt_content)
        return None

      # Use semaphore to limit concurrent HTTP requests
      semaphore = asyncio.Semaphore(PARALLEL_REQUESTS)

      async def fetch_with_semaphore(category: CategoryInfo) -> Optional[Tuple[CategoryInfo, str]]:
        async with semaphore:
          # Add polite delay between requests to avoid overwhelming the server
          await asyncio.sleep(REQUEST_DELAY)
          return await fetch_category_content(category)

      # Process all HTTP requests in parallel
      logger.info(f"Fetching prompt content from {len(categories)} categories...")
      fetch_tasks = [fetch_with_semaphore(category) for category in categories]
      fetch_results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

      # Step 3b: Sequential database operations to get category IDs and prepare data
      successful_prompts = []
      failed_count = 0

      for i, fetch_result in enumerate(fetch_results):
        if isinstance(fetch_result, Exception):
          category_name = categories[i].name if i < len(categories) else f"category_{i}"
          logger.error(f"Error fetching content for category {category_name}: {fetch_result}")
          failed_count += 1
          continue

        if fetch_result is None:
          category_name = categories[i].name if i < len(categories) else f"category_{i}"
          logger.debug(f"No content found for category: {category_name}")
          continue

        # At this point, fetch_result should be a tuple, but mypy needs assurance
        if not isinstance(fetch_result, tuple) or len(fetch_result) != 2:
          logger.error(f"Unexpected fetch_result type: {type(fetch_result)}")
          failed_count += 1
          continue

        category, prompt_content = fetch_result

        try:
          # Sequential database access (one at a time)
          result = await db.execute(
            text("SELECT id FROM prompt_categories WHERE display_order = :display_order"), {"display_order": category.display_order}
          )
          category_id = result.scalar_one_or_none()

          if not category_id:
            logger.error(f"Category ID not found for: {category.name}")
            failed_count += 1
            continue

          successful_prompts.append(
            PromptInfo(title=category.name, content=prompt_content, description=category.description, category_id=category_id)
          )

        except Exception as e:
          logger.error(f"Database error processing category {category.name}: {e}")
          failed_count += 1

      logger.info(f"Successfully processed {len(successful_prompts)} categories, {failed_count} failed")

      # Check if we should fall back to sequential processing due to too many failures
      failure_rate = failed_count / len(categories) if categories else 0
      if USE_SEQUENTIAL_FALLBACK and failure_rate > 0.5 and len(successful_prompts) < 10:
        logger.warning(f"High failure rate ({failure_rate:.1%}). Consider reducing PARALLEL_REQUESTS or checking for rate limiting.")
        logger.info("Tip: Set PARALLEL_REQUESTS=1 in the script for fully sequential processing if needed.")

      # Step 4: Insert prompts into database
      if successful_prompts:
        logger.info(f"Inserting {len(successful_prompts)} prompts into database...")
        prompt_data = [
          {
            "title": prompt.title,
            "content": prompt.content,
            "description": prompt.description,
            "category_id": prompt.category_id,
            "creator_id": user_id,
            "organization_id": organization_id,
          }
          for prompt in successful_prompts
        ]

        await db.execute(
          text("""
            INSERT INTO prompts (title, content, description, category_id, creator_id, organization_id)
            VALUES (:title, :content, :description, :category_id, :creator_id, :organization_id)
          """),
          prompt_data,
        )
        logger.info("Prompts inserted successfully")
      else:
        logger.warning("No prompts were successfully processed")

    logger.info("Prompt population completed successfully")


class AddCustomPromptsScript(BaseScript):
  """
  Script to populate prompt_categories and prompts tables with cursor.directory data.

  Features:
  - Async HTTP requests for improved performance
  - Parallel processing of categories
  - Duplicate detection and prevention
  - Comprehensive error handling and logging
  - Proper rollback and verification capabilities
  """

  def __init__(self):
    super().__init__("add_custom_prompts")

  async def execute(self, db: AsyncSession) -> None:
    """
    Main script execution logic.

    Fetches prompt categories and prompts from cursor.directory and populates
    the database with structured data. Uses async HTTP requests and parallel
    processing for optimal performance.
    """
    logger.info("Starting add_custom_prompts script execution...")

    try:
      # Populate prompt_categories and prompts tables with data
      await populate_prompt_categories_and_prompts(BASE_URL, db)

      # Commit the transaction
      await db.commit()
      logger.info("add_custom_prompts script execution completed successfully.")

    except Exception as e:
      logger.error(f"Script execution failed: {e}")
      await db.rollback()
      raise

  async def rollback(self, db: AsyncSession) -> None:
    """
    Rollback logic for the script.

    Removes all prompt categories and prompts that were added by this script.
    This is determined by checking for categories that match the cursor.directory
    pattern and their associated prompts.
    """
    logger.info("Rolling back add_custom_prompts script...")

    try:
      # Get count of records before rollback for logging
      category_count_result = await db.execute(text("SELECT COUNT(*) FROM prompt_categories"))
      prompt_count_result = await db.execute(text("SELECT COUNT(*) FROM prompts"))

      initial_categories = category_count_result.scalar() or 0
      initial_prompts = prompt_count_result.scalar() or 0

      # Delete prompts associated with cursor.directory categories first (foreign key constraint)
      delete_prompts_query = text("""
                DELETE FROM prompts
                WHERE category_id IN (
                    SELECT id FROM prompt_categories
                    WHERE description LIKE 'Custom instructions for %'
                    OR icon_url LIKE '/rules/%'
                )
            """)
      prompts_result = await db.execute(delete_prompts_query)
      prompts_deleted_count = getattr(prompts_result, "rowcount", 0) or 0

      # Delete cursor.directory categories
      delete_categories_query = text("""
                DELETE FROM prompt_categories
                WHERE description LIKE 'Custom instructions for %'
                OR icon_url LIKE '/rules/%'
            """)
      categories_result = await db.execute(delete_categories_query)
      categories_deleted_count = getattr(categories_result, "rowcount", 0) or 0

      await db.commit()

      logger.info(f"Rollback completed: removed {prompts_deleted_count} prompts and {categories_deleted_count} categories")
      logger.info(
        f"Database state: {initial_categories - categories_deleted_count} categories, {initial_prompts - prompts_deleted_count} prompts remaining"
      )

    except Exception as e:
      logger.error(f"Rollback failed: {e}")
      await db.rollback()
      raise

  async def verify(self, db: AsyncSession) -> bool:
    """
    Verify script execution was successful.

    Checks that:
    1. Prompt categories table has cursor.directory categories
    2. Each category has associated prompts
    3. All prompts have valid content
    4. Data integrity is maintained

    Returns:
        bool: True if verification passes, False otherwise
    """
    logger.info("Verifying add_custom_prompts script execution...")

    try:
      # Check 1: Verify categories were inserted
      category_result = await db.execute(
        text("""
                SELECT COUNT(*) FROM prompt_categories
                WHERE description LIKE 'Custom instructions for %'
            """)
      )
      category_count = category_result.scalar() or 0

      if category_count == 0:
        logger.error("Verification failed: No cursor.directory categories found")
        return False

      logger.info(f"Found {category_count} cursor.directory categories")

      # Check 2: Verify prompts were inserted
      prompt_result = await db.execute(
        text("""
                SELECT COUNT(*) FROM prompts p
                JOIN prompt_categories pc ON p.category_id = pc.id
                WHERE pc.description LIKE 'Custom instructions for %'
            """)
      )
      prompt_count = prompt_result.scalar() or 0

      if prompt_count == 0:
        logger.error("Verification failed: No prompts found for cursor.directory categories")
        return False

      logger.info(f"Found {prompt_count} prompts across all categories")

      # Check 3: Verify content quality (using configurable minimum length)
      empty_content_result = await db.execute(
        text(f"""
                SELECT COUNT(*) FROM prompts p
                JOIN prompt_categories pc ON p.category_id = pc.id
                WHERE pc.description LIKE 'Custom instructions for %'
                AND (p.content IS NULL OR p.content = '' OR LENGTH(TRIM(p.content)) < {MIN_PROMPT_LENGTH})
            """)
      )
      empty_content_count = empty_content_result.scalar() or 0

      if empty_content_count > prompt_count * MAX_EMPTY_PROMPTS_RATIO:
        logger.warning(f"Verification warning: {empty_content_count} prompts have minimal content (< {MIN_PROMPT_LENGTH} chars)")

      # Check 4: Verify referential integrity
      orphaned_prompts_result = await db.execute(
        text("""
                SELECT COUNT(*) FROM prompts p
                LEFT JOIN prompt_categories pc ON p.category_id = pc.id
                WHERE pc.id IS NULL
            """)
      )
      orphaned_count = orphaned_prompts_result.scalar() or 0

      if orphaned_count > 0:
        logger.error(f"Verification failed: {orphaned_count} orphaned prompts found")
        return False

      # Check 5: Verify organization and user references
      invalid_refs_result = await db.execute(
        text("""
                SELECT COUNT(*) FROM prompts p
                JOIN prompt_categories pc ON p.category_id = pc.id
                WHERE pc.description LIKE 'Custom instructions for %'
                AND (p.creator_id IS NULL OR p.organization_id IS NULL)
            """)
      )
      invalid_refs_count = invalid_refs_result.scalar() or 0

      if invalid_refs_count > 0:
        logger.error(f"Verification failed: {invalid_refs_count} prompts missing creator or organization references")
        return False

      logger.info("All verification checks passed successfully!")
      logger.info(f"Summary: {category_count} categories, {prompt_count} prompts, {empty_content_count} with minimal content")

      return True

    except Exception as e:
      logger.error(f"Verification failed with error: {e}")
      return False


def main():
  """Entry point for backward compatibility."""
  script = AddCustomPromptsScript()
  script.main()


if __name__ == "__main__":
  script = AddCustomPromptsScript()
  script.run_cli()
