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


parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if parent_dir not in sys.path:
  sys.path.insert(0, parent_dir)

from scripts.core.base_script import BaseScript
from common.logger import log as logger

# CONFIGURATION CONSTANTS

# Target website configuration
BASE_URL = "https://cursor.directory"
RULES_ENDPOINT = "/rules"

# Performance and validation constants
REQUEST_TIMEOUT = 15
PARALLEL_REQUESTS = 2
RETRY_ATTEMPTS = 3
BACKOFF_FACTOR = 2
REQUEST_DELAY = 0.5
USE_SEQUENTIAL_FALLBACK = True
MIN_PROMPT_LENGTH = 10
MAX_EMPTY_PROMPTS_RATIO = 0.5

# CSS Selectors for web scraping
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


async def check_existing_cursor_data(db: AsyncSession) -> bool:
  """Check if cursor.directory data already exists to prevent duplicates."""
  try:
    result = await db.execute(
      text("""
      SELECT COUNT(*) FROM prompt_categories
      WHERE description LIKE 'Custom instructions for %'
      OR icon_url LIKE '/rules/%'
    """)
    )
    return (result.scalar() or 0) > 0

  except Exception as e:
    logger.warning(f"Tables may not exist yet: {e}")
    return False


async def fetch_html_content(session: aiohttp.ClientSession, url: str, max_retries: int = RETRY_ATTEMPTS) -> Optional[str]:
  """Fetch HTML content with retry logic."""
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
      backoff_delay = BACKOFF_FACTOR * (2**attempt)
      await asyncio.sleep(backoff_delay)

  logger.error(f"Failed to fetch {url} after {max_retries} attempts")
  return None


async def create_new_organization_user(db: AsyncSession) -> Tuple[Optional[str], Optional[str]]:
  """Create a new organization and user."""

  try:
    result = await db.execute(text("SELECT id FROM organizations WHERE name = 'Default Org' UNION SELECT id FROM roles WHERE name = 'admin'"))
    row = result.scalars().all()

    if row:
      org_id = str(row[0])
      role_id = str(row[1])

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
  """Parse HTML content and extract category information."""
  soup = BeautifulSoup(html_content, "html5lib")
  directories = soup.select(SELECTORS["category_links"])

  categories: List[CategoryInfo] = []
  for index, directory in enumerate(directories):
    try:
      if not (directory.button and directory.button.contents):
        continue

      name = str(directory.button.contents[0]).strip()
      href_raw = directory.get("href", "")

      href = str(href_raw) if href_raw else ""

      if not href:
        continue

      categories.append(
        CategoryInfo(name=name, description=f"Custom instructions for {href.split('/')[-1]}", icon_url=href, display_order=len(categories) + 1)
      )

    except Exception as e:
      logger.warning(f"Error processing directory element at index {index}: {e}")
      continue

  logger.info(f"Successfully extracted {len(categories)} categories")
  return categories


async def extract_prompts_from_category(session: aiohttp.ClientSession, category: CategoryInfo, base_url: str) -> Optional[List[str]]:
  """Extract prompts content for a specific category."""
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

  for prompt in directory.children:
    try:
      if not isinstance(prompt, Tag):
        continue

      code_elements = prompt.select(SELECTORS["code_elements"])
      if code_elements and code_elements[0].string:
        prompt_content = code_elements[0].string.strip()
        if len(prompt_content) >= MIN_PROMPT_LENGTH:
          final_prompts.append(prompt_content)
    except Exception:
      continue

  if final_prompts:
    logger.info(f"Extracted {len(final_prompts)} prompts from category: {category.name}")
    return final_prompts
  else:
    logger.warning(f"No prompts found in category: {category.name}")
    return None


async def populate_prompt_categories_and_prompts(base_url: str, db: AsyncSession) -> None:
  """Main function to populate prompt categories and prompts."""

  # Check for existing cursor.directory data to prevent duplicates
  cursor_data_exists = await check_existing_cursor_data(db)
  if cursor_data_exists:
    logger.info("Cursor.directory data already exists. Skipping population.")
    return

  # Get organization information
  organization_id, user_id = await create_new_organization_user(db)
  if not organization_id or not user_id:
    raise Exception("No organization_id or user_id found in database")

  headers = {"User-Agent": "Mozilla/5.0 (compatible; DefinableAI-Scripts/2.0)"}
  connector = aiohttp.TCPConnector(limit=PARALLEL_REQUESTS, limit_per_host=PARALLEL_REQUESTS)
  timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)

  async with aiohttp.ClientSession(
    connector=connector,
    timeout=timeout,
    headers=headers,
    trust_env=True,
  ) as session:
    logger.info("Fetching categories from cursor.directory...")
    html_content = await fetch_html_content(session, base_url + RULES_ENDPOINT)

    if not html_content:
      raise Exception(f"Failed to fetch categories from {base_url}{RULES_ENDPOINT}")

    categories = await extract_categories_from_html(html_content)
    if not categories:
      raise Exception("No categories found on the page")

    logger.info(f"Inserting {len(categories)} categories into database...")
    category_data = [
      {"name": cat.name, "description": cat.description, "icon_url": cat.icon_url, "display_order": cat.display_order} for cat in categories
    ]

    await db.execute(
      text("INSERT INTO prompt_categories (name, description, icon_url, display_order) VALUES (:name, :description, :icon_url, :display_order)"),
      category_data,
    )
    await db.commit()
    logger.info("Categories inserted successfully")

    logger.info("Processing prompts with parallel HTTP requests...")

    async def fetch_category_content(category: CategoryInfo) -> Optional[Tuple[CategoryInfo, List[str]]]:
      """Fetch content for a single category."""
      prompt_content = await extract_prompts_from_category(session, category, base_url)
      if prompt_content:
        return (category, prompt_content)
      return None

    semaphore = asyncio.Semaphore(PARALLEL_REQUESTS)

    async def fetch_with_semaphore(category: CategoryInfo) -> Optional[Tuple[CategoryInfo, List[str]]]:
      async with semaphore:
        await asyncio.sleep(REQUEST_DELAY)
        return await fetch_category_content(category)

    logger.info(f"Fetching prompt content from {len(categories)} categories...")
    fetch_tasks = [fetch_with_semaphore(category) for category in categories]
    fetch_results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

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
        continue

      if not isinstance(fetch_result, tuple) or len(fetch_result) != 2:
        failed_count += 1
        continue

      category, prompt_content_list = fetch_result

      try:
        result = await db.execute(
          text("SELECT id FROM prompt_categories WHERE display_order = :display_order"), {"display_order": category.display_order}
        )
        category_id = result.scalar_one_or_none()

        if not category_id:
          logger.error(f"Category ID not found for: {category.name}")
          failed_count += 1
          continue

        for i, individual_prompt_content in enumerate(prompt_content_list, 1):
          prompt_title = f"{category.name} - Prompt {i}"
          successful_prompts.append(
            PromptInfo(title=prompt_title, content=individual_prompt_content, description=category.description, category_id=category_id)
          )

      except Exception as e:
        logger.error(f"Database error processing category {category.name}: {e}")
        failed_count += 1

    logger.info(f"Successfully processed {len(successful_prompts)} categories, {failed_count} failed")

    failure_rate = failed_count / len(categories) if categories else 0
    if USE_SEQUENTIAL_FALLBACK and failure_rate > 0.5 and len(successful_prompts) < 10:
      logger.warning(f"High failure rate ({failure_rate:.1%}). Consider reducing PARALLEL_REQUESTS or checking for rate limiting.")

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
  """Script to populate prompt_categories and prompts tables with cursor.directory data."""

  def __init__(self):
    super().__init__("add_custom_prompts")

  async def execute(self, db: AsyncSession) -> None:
    """Main script execution logic."""
    logger.info("Starting add_custom_prompts script execution...")

    try:
      await populate_prompt_categories_and_prompts(BASE_URL, db)

      await db.commit()
      logger.info("add_custom_prompts script execution completed successfully.")

    except Exception as e:
      logger.error(f"Script execution failed: {e}")
      await db.rollback()
      raise

  async def rollback(self, db: AsyncSession) -> None:
    """Rollback logic - removes all data added by this script."""
    logger.info("Rolling back add_custom_prompts script...")

    try:
      category_count_result = await db.execute(text("SELECT COUNT(*) FROM prompt_categories"))
      prompt_count_result = await db.execute(text("SELECT COUNT(*) FROM prompts"))
      initial_categories = category_count_result.scalar() or 0
      initial_prompts = prompt_count_result.scalar() or 0
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

      delete_categories_query = text("""
                DELETE FROM prompt_categories
                WHERE description LIKE 'Custom instructions for %'
                OR icon_url LIKE '/rules/%'
            """)
      categories_result = await db.execute(delete_categories_query)
      categories_deleted_count = getattr(categories_result, "rowcount", 0) or 0

      delete_org_members_query = text("""
                DELETE FROM organization_members
                WHERE user_id IN (
                    SELECT id FROM users
                    WHERE email = 'temp-admin@definable.ai'
                    AND stytch_id LIKE 'user-test-%'
                )
            """)
      org_members_result = await db.execute(delete_org_members_query)
      org_members_deleted_count = getattr(org_members_result, "rowcount", 0) or 0

      delete_user_query = text("""
                DELETE FROM users
                WHERE email = 'temp-admin@definable.ai'
                AND stytch_id LIKE 'user-test-%'
            """)
      users_result = await db.execute(delete_user_query)
      users_deleted_count = getattr(users_result, "rowcount", 0) or 0

      await db.commit()

      logger.info(
        f"Rollback completed: removed {prompts_deleted_count} prompts, {categories_deleted_count} categories, "
        f"{users_deleted_count} temp admin users, and {org_members_deleted_count} organization memberships"
      )
      logger.info(
        f"Database state: {initial_categories - categories_deleted_count} categories, {initial_prompts - prompts_deleted_count} prompts remaining"
      )

    except Exception as e:
      logger.error(f"Rollback failed: {e}")
      await db.rollback()
      raise

  async def verify(self, db: AsyncSession) -> bool:
    """Verify script execution was successful."""
    logger.info("Verifying add_custom_prompts script execution...")

    try:
      # Single query to get all verification metrics
      result = await db.execute(
        text(f"""
          WITH cursor_data AS (
            SELECT p.id as prompt_id, p.content, p.creator_id, p.organization_id
            FROM prompts p
            JOIN prompt_categories pc ON p.category_id = pc.id
            WHERE pc.description LIKE 'Custom instructions for %'
          )
          SELECT
            (SELECT COUNT(*) FROM prompt_categories WHERE description LIKE 'Custom instructions for %') as categories,
            (SELECT COUNT(*) FROM cursor_data) as prompts,
            (SELECT COUNT(*) FROM cursor_data WHERE content IS NULL OR content = '' OR LENGTH(TRIM(content)) < {MIN_PROMPT_LENGTH}) as empty_content,
            (SELECT COUNT(*) FROM prompts p LEFT JOIN prompt_categories pc ON p.category_id = pc.id WHERE pc.id IS NULL) as orphaned,
            (SELECT COUNT(*) FROM cursor_data WHERE creator_id IS NULL OR organization_id IS NULL) as invalid_refs
        """)
      )

      row = result.fetchone()
      if not row:
        logger.error("Verification failed: Unable to fetch verification data")
        return False

      categories, prompts, empty_content, orphaned, invalid_refs = row

      # Check all conditions
      failures = [
        (categories == 0, "No cursor.directory categories found"),
        (prompts == 0, "No prompts found for cursor.directory categories"),
        (orphaned > 0, f"{orphaned} orphaned prompts found"),
        (invalid_refs > 0, f"{invalid_refs} prompts missing creator or organization references"),
      ]

      for condition, message in failures:
        if condition:
          logger.error(f"Verification failed: {message}")
          return False

      if empty_content > prompts * MAX_EMPTY_PROMPTS_RATIO:
        logger.warning(f"Verification warning: {empty_content} prompts have minimal content (< {MIN_PROMPT_LENGTH} chars)")

      logger.info("All verification checks passed successfully!")
      logger.info(f"Summary: {categories} categories, {prompts} prompts, {empty_content} with minimal content")
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
