MODEL_CONFIGS = {
  "openai": {
    "gpt-4": {
      "versions": {
        "turbo": {
          "name": "gpt-4-0125-preview",
          "context_window": 128000,
        },
        "vision": {"name": "gpt-4-vision-preview", "context_window": 128000, "vision": True},
      }
    },
    "gpt-3.5": {
      "versions": {
        "latest": {
          "name": "gpt-3.5-turbo-0125",
          "context_window": 16385,
        }
      }
    },
  },
  "anthropic": {
    "claude-3": {
      "versions": {
        "opus": {
          "name": "claude-3-opus-20240229",
          "context_window": 200000,
        },
        "sonnet": {
          "name": "claude-3-sonnet-20240229",
          "context_window": 200000,
        },
        "haiku": {
          "name": "claude-3-haiku-20240307",
          "context_window": 200000,
        },
      }
    }
  },
}
