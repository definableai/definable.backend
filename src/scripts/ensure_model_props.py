#!/usr/bin/env python3
"""
Ensures the 'props' column exists in the models table and populates it with data.
Uses the script_run_tracker table to track execution.
"""

import asyncio
import json
import os
import platform
import sys
import time
from typing import Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

# Add the parent directory to the path so we can import from src
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from common.logger import log as logger
from database.postgres import async_session


async def check_script_executed(db: AsyncSession, script_name: str) -> bool:
  """Checks if this script has already been executed successfully."""
  result = await db.execute(
    text("SELECT status FROM script_run_tracker WHERE script_name = :script_name ORDER BY updated_at DESC LIMIT 1"),
    {"script_name": script_name},
  )
  row = result.fetchone()
  return row[0] == "success" if row else False


async def log_script_execution(db: AsyncSession, script_name: str, status: str, error_message: Optional[str] = None):
  """Logs the script execution with status."""
  try:
    # Check if entry exists
    result = await db.execute(text("SELECT 1 FROM script_run_tracker WHERE script_name = :script_name"), {"script_name": script_name})
    exists = result.scalar()

    if exists:
      # Update existing record
      await db.execute(
        text("""
                    UPDATE script_run_tracker
                    SET status = :status,
                        error_message = :error_message,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE script_name = :script_name
                """),
        {"script_name": script_name, "status": status, "error_message": error_message},
      )
    else:
      # Insert new record
      await db.execute(
        text("""
                    INSERT INTO script_run_tracker (script_name, status, error_message)
                    VALUES (:script_name, :status, :error_message)
                """),
        {"script_name": script_name, "status": status, "error_message": error_message},
      )
    await db.commit()
    logger.info(f"Logged execution of script '{script_name}' with status: {status}")
  except Exception as e:
    await db.rollback()
    logger.error(f"Failed to log script execution: {e}")
    raise


async def check_props_column_exists(db: AsyncSession) -> bool:
  """Checks if the 'props' column exists in the models table."""
  result = await db.execute(
    text("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'models' AND column_name = 'props'
        """)
  )
  if not result.scalar():
    raise ValueError("'props' column does not exist in models table. Run the migration first.")
  return True


async def update_model_props(db: AsyncSession, props_data: dict, llm_models: list):
  """Updates the 'props' column for each model."""
  try:
    for name, provider, version, is_active in llm_models:
      # Get props for this model
      model_props = props_data.get(version, props_data.get(name, {}))

      # Check if model exists
      result = await db.execute(
        text("""
                    SELECT 1 FROM models
                    WHERE name = :name AND provider = :provider AND version = :version
                """),
        {"name": name, "provider": provider, "version": version},
      )
      exists = result.scalar()

      if exists:
        # Update existing model
        await db.execute(
          text("""
                        UPDATE models
                        SET props = :props
                        WHERE name = :name AND provider = :provider AND version = :version
                    """),
          {"props": json.dumps(model_props), "name": name, "provider": provider, "version": version},
        )
        logger.info(f"Updated props for model {name} ({version})")
      else:
        # Insert new model
        await db.execute(
          text("""
                        INSERT INTO models
                        (name, provider, version, is_active, config, props, created_at, updated_at)
                        VALUES
                        (:name, :provider, :version, :is_active, :config, :props, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    """),
          {
            "name": name,
            "provider": provider,
            "version": version,
            "is_active": is_active,
            "config": json.dumps({}),
            "props": json.dumps(model_props),
          },
        )
        logger.info(f"Inserted new model {name} ({version})")
  except Exception as e:
    logger.error(f"Error updating model properties: {e}")
    raise


def get_models_data():
  """Returns props data and model definitions."""
  # Define props data directly in the migration
  props_data = {
    "gpt-4o": {
      "features": [
        "Multimodal (text, vision, audio)",
        "High reasoning ability",
        "Fast and cost-effective compared to GPT-4",
        "Supports long contexts (up to 128k tokens)",
      ],
      "examples": [
        "Summarize this PDF and extract action items. (Attach file)",
        "Write a professional email replying to a job offer.",
        "Explain quantum entanglement in simple terms for a 10-year-old.",
        "Analyze this image and describe the objects in it. (upload image)",
        "Convert this business idea into a pitch deck outline.",
      ],
    },
    "gpt-4o-mini": {
      "features": [
        "Lightweight version of GPT-4o",
        "Fast, affordable, good for basic reasoning and conversation",
        "Optimized for real-time interaction",
      ],
      "examples": [
        "What's a good birthday gift for a 25-year-old software engineer?",
        "Give me 3 ideas for a weekend trip near San Francisco.",
        "Summarize Moby Dick in a more childish way",
        "Help me brainstorm YouTube video titles for a tech channel.",
        "Summarize this blog post into bullet points.",
      ],
    },
    "gpt-3.5-turbo": {
      "features": [
        "Very fast and cheap",
        "Good for casual tasks, basic coding, summarization, chatbots",
        "Limited reasoning depth compared to GPT-4-class models",
      ],
      "examples": [
        "Generate a to-do list for moving to a new apartment.",
        "Create a basic HTML page with a contact form.",
        "What are the pros and cons of remote work?",
        "Explain how HTTP works in simple terms.",
        "Draft a thank-you message for a job interview.",
      ],
    },
    "o1-preview": {
      "features": [
        "Competitive performance with GPT-4-class models",
        "Open-weight / open inference (pending official confirmation)",
        "Suited for research, experiments, and transparency",
      ],
      "examples": [
        "Generate code for a Python web scraper.",
        "Compare three cloud platforms: AWS, Azure, GCP.",
        "What are ethical considerations in AI deployment?",
        "Summarize the latest research on LLM agents.",
        "Write a tutorial for using Docker with Django.",
      ],
    },
    "o1": {
      "features": [
        "Released by OpenAI for open-weight access",
        "Optimized for researchers and transparent use cases",
        "Balanced between performance and efficiency",
      ],
      "examples": [
        "Write a shell script to backup my documents folder.",
        "Explain how transformers work in neural networks.",
        "Suggest 5 project ideas for learning computer vision.",
        "Summarize the book 'The Lean Startup'.",
        "Generate mock interview questions for a backend developer.",
      ],
    },
    "o3-mini": {
      "features": ["Lightweight model in the O series", "Likely optimized for cost-effective inference", "Ideal for low-latency applications"],
      "examples": [
        "Tell me a bedtime story for a 7-year-old.",
        "Convert a sentence to passive voice.",
        "Give me 3 tweet ideas about AI ethics.",
        "Create a simple Python program that rolls a dice.",
        "Explain recursion using a pizza analogy.",
      ],
    },
    "gpt-4.1": {
      "features": [
        "Hypothetical next-gen GPT-4 with enhancements",
        "Better reasoning and memory (if implemented)",
        "Potential for real-time multi-agent interactions",
      ],
      "examples": [
        "Act as a mentor and help me improve my resume.",
        "Debate the pros and cons of crypto regulation.",
        "Write a short story collaboratively with me — I'll take turns.",
        "Plan a 7-day keto meal plan with shopping list.",
        "Simulate a Socratic dialogue on consciousness.",
      ],
    },
    "o4-mini": {
      "features": ["Compact version in the O4 line", "Prioritizes speed and lower costs", "Great for embedded agents or API-heavy applications."],
      "examples": [
        "Generate onboarding instructions for new employees.",
        "Give me a list of productivity hacks for remote work.",
        "Write a regex to extract emails from a text file.",
        "Create an outline for a blog post on AI agents.",
        "Simplify this technical paragraph for a general audience.",
      ],
    },
    "gpt-4.5-preview": {
      "features": [
        "Intermediate between GPT-4 and GPT-5.",
        "Possibly better memory, coherence, and cost efficiency.",
        "Useful for advanced tasks not needing full GPT-5-level performance.",
      ],
      "examples": [
        "Generate a report on competitors in the fintech industry.",
        "Convert a Figma design description into React code.",
        "Summarize and critique this academic abstract.",
        "Help me design a multi-agent system for content moderation.",
        "Translate legal jargon into plain English.",
      ],
    },
    "claude-3-7-sonnet-latest": {
      "features": [
        "Most advanced reasoning capabilities in the Claude family",
        "Complex problem solving and nuanced analysis",
        "Strong at multi-step tasks requiring deep understanding",
        "Advanced coding abilities with sophisticated logic",
        "Handles subtle context and complex instructions",
      ],
      "examples": [
        "Analyze the potential economic impact of quantum computing on cybersecurity markets over the next decade, considering both technological evolution and regulatory responses.",  # noqa: E501
        "Design a Python application that uses machine learning to analyze customer feedback data, extract sentiment patterns, and generate visualized insights with appropriate explanations.",  # noqa: E501
        "Create a comprehensive business plan for a sustainable fashion startup, including market analysis, financial projections, and addressing potential challenges in the supply chain.",  # noqa: E501
        "Develop a detailed architectural proposal for a smart city infrastructure that balances privacy concerns with technological innovation.",
        "Write a research paper outline examining the intersection of behavioral economics and climate change policy, including key research questions and methodological approaches.",  # noqa: E501
      ],
    },
    "claude-3-5-sonnet-latest": {
      "features": [
        "Strong balance of reasoning and efficiency",
        "Good at handling nuanced instructions",
        "Creative content generation with quality outputs",
        "Effective data analysis and summarization",
        "Balanced performance for everyday complex tasks",
      ],
      "examples": [
        "Write a detailed marketing strategy for a new plant-based protein product targeting health-conscious consumers.",
        "Create a JavaScript function that processes user input from a form, validates it, and stores the data in a structured format.",
        "Summarize the key arguments from this academic paper on climate change mitigation strategies and explain their practical implications.",
        "Draft a compelling cover letter for a senior project manager position that highlights leadership experience and communication skills.",
        "Compare and contrast three different approaches to implementing authentication in a web application, including security considerations for each.",  # noqa: E501
      ],
    },
    "claude-3-5-haiku-latest": {
      "features": [
        "Fastest response time among Claude models",
        "Optimized for brief, concise outputs",
        "Efficient handling of straightforward tasks",
        "Quick information retrieval and summarization",
        "Cost-effective for high-volume interactions",
        "Ideal for customer service and simple queries",
      ],
      "examples": [
        "Provide a quick explanation of how blockchain technology works in simple terms.",
        "Draft a brief email confirming a client meeting scheduled for next Tuesday at 2pm.",
        "What are three effective ways to improve team communication in a remote work environment?",
        "Create a simple HTML template for a contact form with name, email, and message fields.",
        "Summarize these meeting notes into 3-5 key action items for the team.",
      ],
    },
    "deepseek-chat": {
      "features": [
        "General-purpose conversational AI",
        "Natural language understanding",
        "Multi-turn dialogue handling",
        "Creative content generation",
        "Basic Q&A and recommendations",
        "Casual conversation capabilities",
        "Multilingual support",
      ],
      "examples": [
        "Help me draft a friendly email to a client about project delays",
        "What are some fun weekend activities for a family with young kids?",
        "Explain quantum computing in simple terms to a high school student",
        "Generate a short story about a robot learning to bake cookies",
        "Translate 'Where is the nearest hospital?' to Spanish and French",
      ],
    },
    "deepseek-reason": {
      "features": [
        "Complex problem solving",
        "Logical reasoning and analysis",
        "Mathematical computations",
        "Algorithm design assistance",
        "Data interpretation and pattern recognition",
        "Technical troubleshooting",
        "Scientific concept explanation",
      ],
      "examples": [
        "Solve this differential equation: dy/dx = 3x² + 2x - 5",
        "Analyze the logical fallacies in this political speech transcript",
        "Design an algorithm to find the shortest path in a weighted graph",
        "Explain the thermodynamic principles behind refrigeration systems",
        "Interpret these sales data trends and predict Q4 performance",
      ],
    },
  }

  llm_models = [
    ("gpt-4.5-preview", "openai", "gpt-4.5-preview", True),
    ("gpt-4.1", "openai", "gpt-4.1", True),
    ("gpt-4o", "openai", "gpt-4o", True),
    ("gpt-4o-mini", "openai", "gpt-4o-mini", True),
    ("gpt-3.5-turbo", "openai", "gpt-3.5-turbo", True),
    ("o4-mini", "openai", "o4-mini", True),
    ("o3-mini", "openai", "o3-mini", True),
    ("o1-preview", "openai", "o1-preview", True),
    ("o1", "openai", "o1", True),
    ("claude-3.7-sonnet", "anthropic", "claude-3-7-sonnet-latest", True),
    ("claude-3.5-sonnet", "anthropic", "claude-3-5-sonnet-latest", True),
    ("claude-3.5-haiku", "anthropic", "claude-3-5-haiku-latest", True),
    ("deepseek-chat", "deepseek", "deepseek-chat", True),
    ("deepseek-reason", "deepseek", "deepseek-reason", True),
  ]
  return props_data, llm_models


async def main():
  """Main function to execute the script."""
  script_name = "ensure_model_props"
  logger.info(f"Starting {script_name} script...")

  async with async_session() as db:
    try:
      # Check if script has already been executed successfully
      if await check_script_executed(db, script_name):
        logger.info(f"Script '{script_name}' has already been executed successfully. Skipping.")
        return

      # Log script execution as pending
      await log_script_execution(db, script_name, "pending")

      # Verify props column exists
      await check_props_column_exists(db)

      # Update models with props data
      props_data, llm_models = get_models_data()
      await update_model_props(db, props_data, llm_models)

      # Log successful script execution
      await log_script_execution(db, script_name, "success")
      logger.info(f"Script '{script_name}' completed successfully.")

    except Exception as e:
      error_message = str(e)
      logger.error(f"Error executing script: {error_message}")

      # Log failed execution
      await log_script_execution(db, script_name, "failed", error_message)
      raise


if __name__ == "__main__":
  try:
    asyncio.run(main())

    # Give connections time to close properly
    if platform.system() == "Windows":
      time.sleep(1)

  except KeyboardInterrupt:
    logger.info("Script interrupted by user")
  except Exception as e:
    logger.error(f"Script failed: {e}")
