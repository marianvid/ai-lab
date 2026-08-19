"""The web server: HTTP in, service calls out, JSON back.

This layer makes no decisions. If you find an `if` here that is about models,
engines or formats, it is in the wrong place — that rule is what keeps the
business logic testable without HTTP, and it is the rule the previous version
broke.
"""
