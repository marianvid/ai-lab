"""The card's front door: which model answers a name, the queue, and loading.

`core.Gateway` is the entry point; `scheduler` decides who goes next;
`catalogue`, `loading`, `limits`, `state`, `control`, `resources`, `stats`
and `errors` are its parts; `eviction` chooses what leaves the card;
`card_sharing` is what a card load asks of the models that borrow the card.
"""
