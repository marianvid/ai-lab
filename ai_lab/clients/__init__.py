"""Who may call AI-Lab: the registered clients, their tokens, the web session.

Every client the owner registers gets one token (`registry`). The door checks it
on every call (`door`) and remembers each client's last call and last heartbeat
(`presence`). The web interface is let in by a session cookie instead (`sessions`),
given at login or, with login switched off, when the page is opened.

Nothing in here decides anything about models or the card.
"""
