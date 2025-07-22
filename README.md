# Crypto Trading Bot - Telegram Token Tracker

A Telegram bot that tracks and analyzes cryptocurrency tokens, providing alerts and analysis for promising opportunities.

## Heroku Deployment Instructions

1. Create a new Heroku app:
```bash
heroku create your-app-name
```

2. Set up the required environment variable:
```bash
heroku config:set TELEGRAM_BOT_TOKEN=your_telegram_bot_token
```

3. Deploy to Heroku:
```bash
git push heroku main
```

4. Ensure the worker dyno is running:
```bash
heroku ps:scale worker=1
```

## Required Files

The following files must exist in the project root:
- `logo.svg`
- `active_chats.json`
- `shared_tokens_log.csv`

Create these files before deploying:
```bash
touch active_chats.json
touch shared_tokens_log.csv
```

## Environment Variables

- `TELEGRAM_BOT_TOKEN`: Your Telegram bot token from BotFather

## Configuration

The bot's behavior is configured through `config.yaml`, which includes settings for:
- Token filtering criteria
- Scoring weights
- Scan intervals
- Message rate limiting
- Health check parameters

## Monitoring

Monitor your bot's logs using:
```bash
heroku logs --tail
