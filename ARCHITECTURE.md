# The core loop

```
input -> slop-dome defence loop -> final outcome / summary if any
```

# The filtering algorithm

```mermaid
flowchart TD
    A[input from adapter] --> B{is it AI message?}
    B -- No --> C[allow message through + summary if any]
    B -- Yes --> D[slop-dome defence loop]
    D --> E[synethesised summary + any actionables]


```

## Logic: Is AI message

# Interception mechanism (adapters)

- Slack adapter
- Whatsapp adapter
- Email adapter
