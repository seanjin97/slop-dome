# The core loop

```
input -> slop-dome defence loop -> final outcome / summary if any
```

# The filtering algorithm

```mermaid
flowchart TD
    A[input from adapter] --> B[cloudflared ingress to local pc]
    B --> C{is it AI message?}
    C -- No --> D[allow message through + summary if any]
    C -- Yes --> E[slop-dome defence loop]
    E --> F[synethesised summary + any actionables]


```

## Logic: Is AI message

# Interception mechanism (adapters)

- Slack adapter
- Whatsapp adapter
- Email adapter
