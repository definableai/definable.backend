"""006_insert_default_roles_permissions

Revision ID: e080a53fbcbd
Revises: 0bb3780d56dd
Create Date: 2025-01-29 13:18:01.249505
"""

import json
from typing import Sequence, Union
from uuid import uuid4

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e080a53fbcbd"
down_revision: Union[str, None] = "0bb3780d56dd"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
  permissions = [
    # Billing permissions
    ("billing_read", "Read billing information", "billing", "read"),
    ("billing_write", "Modify billing settings", "billing", "write"),
    ("billing_delete", "Delete billing records", "billing", "delete"),
    # User permissions
    ("users_read", "View users", "users", "read"),
    ("users_write", "Modify users", "users", "write"),
    ("users_delete", "Delete users", "users", "delete"),
    # Agent permissions
    ("agents_read", "View agents", "agents", "read"),
    ("agents_write", "Modify agents", "agents", "write"),
    ("agents_delete", "Delete agents", "agents", "delete"),
    # Prompt permissions
    ("prompts_read", "View prompts", "prompts", "read"),
    ("prompts_write", "Create/modify prompts", "prompts", "write"),
    ("prompts_delete", "Delete prompts", "prompts", "delete"),
    # Model permissions
    ("models_read", "View models", "models", "read"),
    ("models_write", "Configure models", "models", "write"),
    ("models_delete", "Delete models", "models", "delete"),
    # Chats permissions
    ("chats_read", "View chats", "chats", "read"),
    ("chats_write", "Create/modify chats", "chats", "write"),
    ("chats_delete", "Delete chats", "chats", "delete"),
    # Organization permissions
    ("org_read", "View organization", "org", "read"),
    ("org_write", "Modify organization", "org", "write"),
    ("org_delete", "Delete organization", "org", "delete"),
    # Role permissions
    ("role_read", "View roles", "roles", "read"),
    ("role_write", "Create/modify roles", "roles", "write"),
    ("role_delete", "Delete roles", "roles", "delete"),
    # Knowledge base permissions
    ("kb_read", "View knowledge bases", "kb", "read"),
    ("kb_write", "Create/modify knowledge bases", "kb", "write"),
    ("kb_delete", "Delete knowledge bases", "kb", "delete"),
    # Tools permissions
    ("tools_read", "View tools", "tools", "read"),
    ("tools_write", "Create/modify tools", "tools", "write"),
    ("tools_delete", "Delete tools", "tools", "delete"),
    # all access
    ("*_*", "Access to all resources", "*", "*"),
  ]

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
        "Create a comprehensive business plan for a sustainable fashion startup, including market analysis, financial projections, and addressing potential challenges in the supply chain.", # noqa: E501
        "Develop a detailed architectural proposal for a smart city infrastructure that balances privacy concerns with technological innovation.",
        "Write a research paper outline examining the intersection of behavioral economics and climate change policy, including key research questions and methodological approaches.", # noqa: E501
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
        "Compare and contrast three different approaches to implementing authentication in a web application, including security considerations for each.", # noqa: E501
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

  for name, provider, version, is_active in llm_models:
    # Get props for this model
    model_props = {}
    if version in props_data:
      model_props = props_data[version]
    elif name in props_data:
      model_props = props_data[name]

    # Convert to JSON string and escape single quotes for SQL
    props_json = json.dumps(model_props).replace("'", "''")

    op.execute(f"""
      INSERT INTO models (id, name, provider, version, is_active, config, props, created_at, updated_at)
      VALUES ('{str(uuid4())}', '{name}', '{provider}', '{version}', {is_active}, '{{}}', '{props_json}', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
    """)

  # Rest of the function remains unchanged
  permission_ids = {}
  for name, desc, resource, action in permissions:
    perm_id = str(uuid4())
    op.execute(f"""
      INSERT INTO permissions (id, name, description, resource, action, created_at)
      VALUES ('{perm_id}', '{name}', '{desc}', '{resource}', '{action}', CURRENT_TIMESTAMP)
    """)
    permission_ids[name] = perm_id

  org_id = str(uuid4())
  op.execute(f"""
    INSERT INTO organizations (id, name, slug, created_at, updated_at)
    VALUES ('{org_id}', 'Default', 'default', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
  """)

  roles = [
    ("owner", "Organization owner with full access", True, 90),
    ("admin", "Administrator with management access", True, 60),
    ("dev", "Developer with limited access", True, 30),
  ]

  for role_name, desc, is_system, level in roles:
    role_id = str(uuid4())
    op.execute(f"""
      INSERT INTO roles (id, organization_id, name, description, is_system_role, hierarchy_level, created_at, updated_at)
      VALUES (
        '{role_id}',
        '{org_id}',
        '{role_name}',
        '{desc}',
        {is_system},
        {level},
        CURRENT_TIMESTAMP,
        CURRENT_TIMESTAMP
      )
    """)
    if role_name == "owner":
      # Owner gets all permissions
      for perm_id in permission_ids.values():
        op.execute(f"""
          INSERT INTO role_permissions (id, role_id, permission_id, created_at)
          VALUES ('{str(uuid4())}', '{role_id}', '{perm_id}', CURRENT_TIMESTAMP)
        """)

    elif role_name == "admin":
      # Admin gets all read/write but no delete
      for perm_name, perm_id in permission_ids.items():
        if "delete" not in perm_name:
          op.execute(f"""
                    INSERT INTO role_permissions (id, role_id, permission_id, created_at)
                    VALUES ('{str(uuid4())}', '{role_id}', '{perm_id}', CURRENT_TIMESTAMP)
                """)

    elif role_name == "dev":
      # Developer gets only read permissions
      for perm_name, perm_id in permission_ids.items():
        if perm_name.endswith("_read"):
          op.execute(f"""
                    INSERT INTO role_permissions (id, role_id, permission_id, created_at)
                    VALUES ('{str(uuid4())}', '{role_id}', '{perm_id}', CURRENT_TIMESTAMP)
                """)


def downgrade() -> None:
  # Delete in reverse order to respect foreign key constraints

  op.execute("""
    DELETE FROM role_permissions
    WHERE role_id IN (
      SELECT id FROM roles
      WHERE is_system_role = true
    )
  """)

  op.execute("""
    DELETE FROM roles
    WHERE is_system_role = true
  """)

  op.execute("DELETE FROM permissions")
  op.execute("UPDATE messages SET model_id = NULL WHERE model_id IS NOT NULL")
  op.execute("DELETE FROM models")
